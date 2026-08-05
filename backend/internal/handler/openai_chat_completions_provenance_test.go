//go:build unit

package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestBeginOpenAIChatCompletionsAttemptClearsPriorUpstreamProvenanceBeforeSelection(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	provenanceHeaders := []string{
		"X-Sub2API-Requested-Model",
		"X-Sub2API-Selected-Account-Platform",
		"X-Sub2API-Sent-Upstream-Model",
		"X-Sub2API-Provenance-Level",
		"X-Sub2API-Physical-Platform",
		"X-Sub2API-Actual-Model",
		"X-Sub2API-Upstream-Platform",
		"X-Upstream-Model",
		"X-Actual-Model",
		"X-Sub2API-Upstream-Model",
		"X-Model-Mapping-Chain",
		"X-Sub2API-Model-Mapping-Chain",
		"X-Upstream-Request-Id",
	}
	for _, name := range provenanceHeaders {
		c.Writer.Header().Set(name, "attempt-one")
	}

	beginOpenAIChatCompletionsAttempt(c)
	c.JSON(http.StatusBadGateway, gin.H{"error": "selection exhausted"})

	for _, name := range provenanceHeaders {
		require.Empty(t, rec.Header().Values(name), name)
	}
}
