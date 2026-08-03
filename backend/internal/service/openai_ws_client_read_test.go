package service

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	coderws "github.com/coder/websocket"
	"github.com/stretchr/testify/require"
)

func TestReadOpenAIWSClientMessage_ControlCloseFrames(t *testing.T) {
	tests := []struct {
		name          string
		timeout       time.Duration
		timeoutStatus coderws.StatusCode
		timeoutReason string
		cancelCause   error
		wantStatus    coderws.StatusCode
		wantReason    string
	}{
		{
			name:          "inter-turn idle sends normal close",
			timeout:       25 * time.Millisecond,
			timeoutStatus: coderws.StatusNormalClosure,
			timeoutReason: "websocket idle timeout",
			wantStatus:    coderws.StatusNormalClosure,
			wantReason:    "websocket idle timeout",
		},
		{
			name:          "first message timeout sends policy close",
			timeout:       25 * time.Millisecond,
			timeoutStatus: coderws.StatusPolicyViolation,
			timeoutReason: "missing first response.create message",
			wantStatus:    coderws.StatusPolicyViolation,
			wantReason:    "missing first response.create message",
		},
		{
			name:        "lease loss sends retry close",
			cancelCause: ErrOpenAIWSIngressLeaseLost,
			wantStatus:  coderws.StatusTryAgainLater,
			wantReason:  "websocket ingress capacity lease lost; please reconnect",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			controlCtx, cancelControl := context.WithCancelCause(context.Background())
			defer cancelControl(context.Canceled)
			serverResult := make(chan error, 1)
			readStarted := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, err := coderws.Accept(w, r, nil)
				if err != nil {
					serverResult <- err
					return
				}
				defer func() { _ = conn.CloseNow() }()
				close(readStarted)
				_, _, err = ReadOpenAIWSClientMessage(
					controlCtx,
					conn,
					tt.timeout,
					tt.timeoutStatus,
					tt.timeoutReason,
				)
				serverResult <- err
			}))
			defer server.Close()

			dialCtx, cancelDial := context.WithTimeout(context.Background(), time.Second)
			clientConn, _, err := coderws.Dial(dialCtx, "ws"+strings.TrimPrefix(server.URL, "http"), nil)
			cancelDial()
			require.NoError(t, err)
			defer func() { _ = clientConn.CloseNow() }()
			<-readStarted
			if tt.cancelCause != nil {
				cancelControl(tt.cancelCause)
			}

			readCtx, cancelRead := context.WithTimeout(context.Background(), time.Second)
			_, _, err = clientConn.Read(readCtx)
			cancelRead()
			var clientClose coderws.CloseError
			require.ErrorAs(t, err, &clientClose)
			require.Equal(t, tt.wantStatus, clientClose.Code)
			require.Equal(t, tt.wantReason, clientClose.Reason)

			select {
			case serverErr := <-serverResult:
				var closeErr *OpenAIWSClientCloseError
				require.ErrorAs(t, serverErr, &closeErr)
				require.Equal(t, tt.wantStatus, closeErr.StatusCode())
				require.Equal(t, tt.wantReason, closeErr.Reason())
			case <-time.After(time.Second):
				t.Fatal("server read goroutine did not exit after close handshake")
			}
		})
	}
}

func TestReadOpenAIWSClientMessage_ParentCancellationStillJoinsRead(t *testing.T) {
	controlCtx, cancelControl := context.WithCancelCause(context.Background())
	serverResult := make(chan error, 1)
	readStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := coderws.Accept(w, r, nil)
		if err != nil {
			serverResult <- err
			return
		}
		defer func() { _ = conn.CloseNow() }()
		close(readStarted)
		_, _, err = ReadOpenAIWSClientMessage(controlCtx, conn, 0, 0, "")
		serverResult <- err
	}))
	defer server.Close()

	dialCtx, cancelDial := context.WithTimeout(context.Background(), time.Second)
	clientConn, _, err := coderws.Dial(dialCtx, "ws"+strings.TrimPrefix(server.URL, "http"), nil)
	cancelDial()
	require.NoError(t, err)
	defer func() { _ = clientConn.CloseNow() }()
	<-readStarted
	cancelControl(errors.New("server shutting down"))
	readCtx, cancelRead := context.WithTimeout(context.Background(), time.Second)
	_, _, err = clientConn.Read(readCtx)
	cancelRead()
	var clientClose coderws.CloseError
	require.ErrorAs(t, err, &clientClose)
	require.Equal(t, coderws.StatusGoingAway, clientClose.Code)
	require.Equal(t, "websocket request canceled", clientClose.Reason)

	select {
	case <-serverResult:
	case <-time.After(time.Second):
		t.Fatal("server read goroutine leaked after parent cancellation")
	}
}

func TestReadOpenAIWSClientMessage_StalledPeerCleanupIsBounded(t *testing.T) {
	serverResult := make(chan error, 1)
	readStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := coderws.Accept(w, r, nil)
		if err != nil {
			serverResult <- err
			return
		}
		defer func() { _ = conn.CloseNow() }()
		close(readStarted)
		_, _, err = ReadOpenAIWSClientMessage(
			context.Background(),
			conn,
			250*time.Millisecond,
			coderws.StatusPolicyViolation,
			"missing first response.create message",
		)
		serverResult <- err
	}))
	defer server.Close()

	rawConn, rawReader := dialRawWebSocket(t, server)
	defer func() { _ = rawConn.Close() }()
	<-readStarted

	writeRawPartialTextFrame(t, rawConn, 1<<20, []byte(`{"type":"response.create","input":"incomplete`))
	// Keep the socket open after declaring a 1 MiB frame but sending only the
	// prefix. This reproduces a client that stalls inside the first frame body.

	startedAt := time.Now()
	select {
	case serverErr := <-serverResult:
		var closeErr *OpenAIWSClientCloseError
		require.ErrorAs(t, serverErr, &closeErr)
		require.Equal(t, coderws.StatusPolicyViolation, closeErr.StatusCode())
		require.ErrorIs(t, serverErr, context.DeadlineExceeded)
		require.NotErrorIs(t, serverErr, ErrOpenAIWSClientCleanupTimeout)
		require.Less(t, time.Since(startedAt), 3*time.Second)
	case <-time.After(3 * time.Second):
		t.Fatal("stalled websocket peer kept the server reader blocked")
	}
	requireRawSocketClosed(t, rawConn, rawReader)
}

func TestReadOpenAIWSClientMessage_ReadErrorCleanupIsBounded(t *testing.T) {
	serverResult := make(chan error, 1)
	readStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := coderws.Accept(w, r, nil)
		if err != nil {
			serverResult <- err
			return
		}
		defer func() { _ = conn.CloseNow() }()
		conn.SetReadLimit(32)
		close(readStarted)
		_, _, err = ReadOpenAIWSClientMessage(
			context.Background(),
			conn,
			5*time.Second,
			coderws.StatusPolicyViolation,
			"missing first response.create message",
		)
		serverResult <- err
	}))
	defer server.Close()

	rawConn, rawReader := dialRawWebSocket(t, server)
	defer func() { _ = rawConn.Close() }()
	<-readStarted
	writeRawPartialTextFrame(t, rawConn, 1<<20, []byte(strings.Repeat("x", 64)))

	startedAt := time.Now()
	select {
	case serverErr := <-serverResult:
		require.ErrorIs(t, serverErr, coderws.ErrMessageTooBig)
		require.NotErrorIs(t, serverErr, ErrOpenAIWSClientCleanupTimeout)
		require.Less(t, time.Since(startedAt), 2*time.Second)
	case <-time.After(2 * time.Second):
		t.Fatal("oversized partial frame kept the server reader blocked")
	}
	requireRawSocketClosed(t, rawConn, rawReader)
}

func dialRawWebSocket(t *testing.T, server *httptest.Server) (net.Conn, *bufio.Reader) {
	t.Helper()
	rawConn, err := net.DialTimeout("tcp", server.Listener.Addr().String(), time.Second)
	require.NoError(t, err)
	rawReader := bufio.NewReader(rawConn)
	_, err = fmt.Fprintf(
		rawConn,
		"GET / HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
		server.Listener.Addr().String(),
	)
	require.NoError(t, err)
	response, err := http.ReadResponse(rawReader, &http.Request{Method: http.MethodGet})
	require.NoError(t, err)
	require.Equal(t, http.StatusSwitchingProtocols, response.StatusCode)
	require.NoError(t, response.Body.Close())
	return rawConn, rawReader
}

func writeRawPartialTextFrame(t *testing.T, rawConn net.Conn, declaredPayloadBytes uint64, partialPayload []byte) {
	t.Helper()
	maskKey := [4]byte{0x11, 0x22, 0x33, 0x44}
	framePrefix := make([]byte, 14)
	framePrefix[0] = 0x81 // FIN + text frame
	framePrefix[1] = 0xff // masked + 64-bit payload length
	binary.BigEndian.PutUint64(framePrefix[2:10], declaredPayloadBytes)
	copy(framePrefix[10:14], maskKey[:])
	_, err := rawConn.Write(framePrefix)
	require.NoError(t, err)
	maskedPayload := append([]byte(nil), partialPayload...)
	for i := range maskedPayload {
		maskedPayload[i] ^= maskKey[i%len(maskKey)]
	}
	_, err = rawConn.Write(maskedPayload)
	require.NoError(t, err)
}

func requireRawSocketClosed(t *testing.T, rawConn net.Conn, rawReader *bufio.Reader) {
	t.Helper()
	require.NoError(t, rawConn.SetReadDeadline(time.Now().Add(time.Second)))
	for {
		_, readErr := rawReader.ReadByte()
		if readErr == nil {
			continue
		}
		require.Error(t, readErr)
		if netErr, ok := readErr.(net.Error); ok {
			require.False(t, netErr.Timeout(), "server left the partial-frame socket open")
		}
		break
	}
}
