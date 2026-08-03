package service

import (
	"context"
	"errors"
	"time"

	coderws "github.com/coder/websocket"
)

type openAIWSClientReadResult struct {
	messageType coderws.MessageType
	payload     []byte
	err         error
}

const (
	openAIWSClientCloseHandshakeGrace = time.Second
	openAIWSClientCleanupJoinTimeout  = time.Second
)

// ErrOpenAIWSClientCleanupTimeout marks a client connection whose bounded
// cleanup did not finish before the handler was allowed to release its lease.
var ErrOpenAIWSClientCleanupTimeout = errors.New("openai websocket client cleanup timed out")

// ReadOpenAIWSClientMessage owns connection cleanup for every returned read
// error. Control events get a brief close-handshake grace period; stalled or
// invalid reads are force-closed without indefinitely holding the caller.
func ReadOpenAIWSClientMessage(
	controlCtx context.Context,
	conn *coderws.Conn,
	timeout time.Duration,
	timeoutStatus coderws.StatusCode,
	timeoutReason string,
) (coderws.MessageType, []byte, error) {
	return readOpenAIWSClientMessageWithTimeoutStart(
		controlCtx,
		conn,
		timeout,
		timeoutStatus,
		timeoutReason,
		nil,
		nil,
	)
}

// readOpenAIWSClientMessageWithTimeoutStart supports readers whose timeout
// starts after a state transition, such as a completed passthrough turn. When
// timeoutActive is nil, a positive timeout starts immediately.
func readOpenAIWSClientMessageWithTimeoutStart(
	controlCtx context.Context,
	conn *coderws.Conn,
	timeout time.Duration,
	timeoutStatus coderws.StatusCode,
	timeoutReason string,
	timeoutStart <-chan struct{},
	timeoutActive func() bool,
) (coderws.MessageType, []byte, error) {
	if conn == nil {
		return 0, nil, errors.New("openai websocket client connection is nil")
	}
	if controlCtx == nil {
		controlCtx = context.Background()
	}

	readCtx, cancelRead := context.WithCancel(context.Background())
	defer cancelRead()

	readDone := make(chan openAIWSClientReadResult, 1)
	go func() {
		messageType, payload, err := conn.Read(readCtx)
		readDone <- openAIWSClientReadResult{messageType: messageType, payload: payload, err: err}
	}()

	var timer *time.Timer
	var timeoutCh <-chan time.Time
	startTimeout := func() {
		if timeout <= 0 || (timeoutActive != nil && !timeoutActive()) {
			return
		}
		if timer == nil {
			timer = time.NewTimer(timeout)
		} else {
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer.Reset(timeout)
		}
		timeoutCh = timer.C
	}
	if timeoutActive == nil || timeoutActive() {
		startTimeout()
	}
	defer func() {
		if timer != nil {
			timer.Stop()
		}
	}()

	closeAndJoin := func(status coderws.StatusCode, reason string, cause error) (coderws.MessageType, []byte, error) {
		closeDone := make(chan struct{}, 1)
		go func() {
			_ = conn.Close(status, reason)
			closeDone <- struct{}{}
		}()

		closeWait := (<-chan struct{})(closeDone)
		graceTimer := time.NewTimer(openAIWSClientCloseHandshakeGrace)
		select {
		case <-closeWait:
			closeWait = nil
		case <-graceTimer.C:
		}
		if !graceTimer.Stop() {
			select {
			case <-graceTimer.C:
			default:
			}
		}

		// coder/websocket binds Read context cancellation to the underlying
		// transport. This force-closes a peer that does not answer the close
		// handshake and prevents a partial first frame from pinning the reader.
		cancelRead()

		readWait := (<-chan openAIWSClientReadResult)(readDone)
		joinTimer := time.NewTimer(openAIWSClientCleanupJoinTimeout)
		defer joinTimer.Stop()
		for readWait != nil || closeWait != nil {
			select {
			case <-readWait:
				readWait = nil
			case <-closeWait:
				closeWait = nil
			case <-joinTimer.C:
				return 0, nil, NewOpenAIWSClientCloseError(
					status,
					reason,
					errors.Join(cause, ErrOpenAIWSClientCleanupTimeout),
				)
			}
		}
		return 0, nil, NewOpenAIWSClientCloseError(status, reason, cause)
	}
	forceCloseBounded := func(cause error) error {
		cancelRead()
		closeDone := make(chan struct{}, 1)
		go func() {
			_ = conn.CloseNow()
			closeDone <- struct{}{}
		}()

		joinTimer := time.NewTimer(openAIWSClientCleanupJoinTimeout)
		defer joinTimer.Stop()
		select {
		case <-closeDone:
			return cause
		case <-joinTimer.C:
			return errors.Join(cause, ErrOpenAIWSClientCleanupTimeout)
		}
	}

	for {
		select {
		case result := <-readDone:
			if result.err != nil {
				result.err = forceCloseBounded(result.err)
			}
			return result.messageType, result.payload, result.err
		case <-timeoutStart:
			startTimeout()
		case <-timeoutCh:
			return closeAndJoin(timeoutStatus, timeoutReason, context.DeadlineExceeded)
		case <-controlCtx.Done():
			cause := context.Cause(controlCtx)
			if errors.Is(cause, ErrOpenAIWSIngressLeaseLost) {
				return closeAndJoin(
					coderws.StatusTryAgainLater,
					"websocket ingress capacity lease lost; please reconnect",
					cause,
				)
			}
			return closeAndJoin(coderws.StatusGoingAway, "websocket request canceled", cause)
		}
	}
}
