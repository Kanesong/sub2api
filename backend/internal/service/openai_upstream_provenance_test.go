//go:build unit

package service

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestWriteOpenAIUpstreamProvenanceIdentityGrok(t *testing.T) {
	t.Parallel()

	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{
		Matched:        true,
		PublicModel:    "grok-4.5",
		TargetPlatform: PlatformGrok,
		UpstreamModel:  "grok-4.5",
	})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok}, "grok-4.5", "grok-4.5", "grok-4.5",
		"https://api.x.ai/v1/responses", http.Header{
			"Xai-Request-Id": []string{"xai_req_123"},
		})
	promoteOpenAIActualModel(c, "grok-4.5")

	require.Equal(t, "grok-4.5", rec.Header().Get(headerSub2APIRequestedModel))
	require.Equal(t, PlatformGrok, rec.Header().Get(headerSub2APISelectedAccountPlatform))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerSub2APISentUpstreamModel))
	require.Equal(t, provenanceLevelDirectOfficial, rec.Header().Get(headerSub2APIProvenanceLevel))
	require.Equal(t, PlatformGrok, rec.Header().Get(headerSub2APIPhysicalPlatform))
	require.Equal(t, PlatformGrok, rec.Header().Get(headerSub2APIUpstreamPlatform))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerActualModel))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerSub2APIActualModel))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerSub2APIUpstreamModel))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerSub2APIModelMappingChain))
	require.Equal(t, "xai_req_123", rec.Header().Get(headerUpstreamRequestID))
}

func TestWriteOpenAIUpstreamProvenanceExposesTransparentMapping(t *testing.T) {
	t.Parallel()

	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{
		Matched:        true,
		PublicModel:    "grok-4.5",
		TargetPlatform: PlatformOpenAI,
		UpstreamModel:  "gpt-5.6-sol",
	})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformOpenAI}, "gpt-5.6-sol", "gpt-5.6-sol", "gpt-5.6-sol",
		"https://chatgpt.com/backend-api/codex/responses", http.Header{
			"X-Request-Id": []string{"openai_req_456"},
		})
	promoteOpenAIActualModel(c, "gpt-5.6-sol")

	require.Equal(t, "grok-4.5", rec.Header().Get(headerSub2APIRequestedModel))
	require.Equal(t, PlatformOpenAI, rec.Header().Get(headerSub2APIUpstreamPlatform))
	require.Equal(t, "gpt-5.6-sol", rec.Header().Get(headerActualModel))
	require.Equal(t, "grok-4.5→gpt-5.6-sol", rec.Header().Get(headerModelMappingChain))
	require.Equal(t, "openai_req_456", rec.Header().Get(headerUpstreamRequestID))
}

func TestWriteOpenAIUpstreamProvenanceDoesNotClaimUnknownModel(t *testing.T) {
	t.Parallel()

	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok}, "grok-4.5", "grok-4.5", "",
		"https://api.x.ai/v1/responses", http.Header{})

	require.Equal(t, PlatformGrok, rec.Header().Get(headerSub2APIUpstreamPlatform))
	require.Empty(t, rec.Header().Get(headerActualModel))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerModelMappingChain))
	require.Empty(t, rec.Header().Get(headerUpstreamRequestID))
}

func TestWriteOpenAIUpstreamProvenanceSanitizesAndBoundsUntrustedHeaderValues(t *testing.T) {
	t.Parallel()

	publicModel := "grok-4.5\r\nX-Spoofed: yes" + strings.Repeat("x", 256)
	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{
		Matched:        true,
		PublicModel:    publicModel,
		TargetPlatform: PlatformGrok,
		UpstreamModel:  "grok-4.5",
	})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: "grok\r\nX-Spoofed: yes"},
		"grok-4.5", "grok-4.5", "grok-4.5", "https://relay.example/v1/chat/completions", http.Header{
			"X-Request-Id": []string{"rid\r\nX-Spoofed: yes" + strings.Repeat("r", 512)},
		})

	require.NotContains(t, rec.Header().Get(headerSub2APIRequestedModel), "\r")
	require.NotContains(t, rec.Header().Get(headerSub2APIRequestedModel), "\n")
	require.LessOrEqual(t, len([]rune(rec.Header().Get(headerSub2APIRequestedModel))), 128)
	require.Equal(t, "grokX-Spoofed: yes", rec.Header().Get(headerSub2APISelectedAccountPlatform))
	require.Empty(t, rec.Header().Get(headerSub2APIUpstreamPlatform))
	require.LessOrEqual(t, len([]rune(rec.Header().Get(headerUpstreamRequestID))), 256)
	require.Empty(t, rec.Header().Get("X-Spoofed"))
}

func TestWriteOpenAIUpstreamProvenanceRejectsAmbiguousRequestIDs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		headers http.Header
	}{
		{
			name:    "same value duplicated in one header",
			headers: http.Header{"X-Request-Id": []string{"rid-1", "rid-1"}},
		},
		{
			name:    "conflicting values duplicated in one header",
			headers: http.Header{"X-Request-Id": []string{"rid-1", "rid-2"}},
		},
		{
			name: "same value repeated across aliases",
			headers: http.Header{
				"X-Request-Id":   []string{"rid-1"},
				"Xai-Request-Id": []string{"rid-1"},
			},
		},
		{
			name: "conflicting values repeated across aliases",
			headers: http.Header{
				"X-Request-Id": []string{"rid-1"},
				"Request-Id":   []string{"rid-2"},
			},
		},
		{
			name:    "empty and non-empty values are ambiguous",
			headers: http.Header{"X-Request-Id": []string{"", "rid-1"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{
				Matched:        true,
				PublicModel:    "grok-4.5",
				TargetPlatform: PlatformGrok,
				UpstreamModel:  "grok-4.5",
			})
			writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok},
				"grok-4.5", "grok-4.5", "grok-4.5", "https://api.x.ai/v1/responses", tt.headers)

			require.Empty(t, rec.Header().Get(headerUpstreamRequestID))
		})
	}
}

func TestNewStreamHeaderWriterRejectsSpoofedUpstreamProvenance(t *testing.T) {
	t.Parallel()

	cfg := &config.Config{Security: config.SecurityConfig{ResponseHeaders: config.ResponseHeaderConfig{
		Enabled: true,
		AdditionalAllowed: []string{
			"x-sub2api-requested-model",
			"x-sub2api-selected-account-platform",
			"x-sub2api-sent-upstream-model",
			"x-sub2api-provenance-level",
			"x-sub2api-physical-platform",
			"x-sub2api-actual-model",
			"x-upstream-request-id",
			"x-actual-model",
			"x-upstream-model",
			"x-sub2api-upstream-model",
			"x-model-mapping-chain",
			"x-sub2api-model-mapping-chain",
			"x-sub2api-upstream-platform",
		},
	}}}
	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{
		Matched:        true,
		PublicModel:    "grok-4.5",
		TargetPlatform: PlatformGrok,
		UpstreamModel:  "grok-4.5",
	})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok}, "grok-4.5", "grok-4.5", "grok-4.5",
		"https://api.x.ai/v1/responses", http.Header{"Xai-Request-Id": []string{"trusted-request"}})
	promoteOpenAIActualModel(c, "grok-4.5")

	svc := &OpenAIGatewayService{cfg: cfg, responseHeaderFilter: compileResponseHeaderFilter(cfg)}
	writeHeaders := svc.newStreamHeaderWriter(c, http.Header{
		"X-Sub2api-Requested-Model":           []string{"spoof-public"},
		"X-Sub2api-Selected-Account-Platform": []string{PlatformOpenAI},
		"X-Sub2api-Sent-Upstream-Model":       []string{"gpt-5.6-sol"},
		"X-Sub2api-Provenance-Level":          []string{provenanceLevelLogicalOnly},
		"X-Sub2api-Physical-Platform":         []string{PlatformOpenAI},
		"X-Sub2api-Actual-Model":              []string{"gpt-5.6-sol"},
		"X-Upstream-Request-Id":               []string{"spoof-request"},
		"X-Actual-Model":                      []string{"gpt-5.6-sol"},
		"X-Upstream-Model":                    []string{"gpt-5.6-sol"},
		"X-Sub2api-Upstream-Model":            []string{"gpt-5.6-sol"},
		"X-Model-Mapping-Chain":               []string{"grok-4.5→gpt-5.6-sol"},
		"X-Sub2api-Model-Mapping-Chain":       []string{"grok-4.5→gpt-5.6-sol"},
		"X-Sub2api-Upstream-Platform":         []string{PlatformOpenAI},
	})
	writeHeaders()

	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "grok-4.5", rec.Header().Get(headerActualModel))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerUpstreamModel))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerModelMappingChain))
	require.Equal(t, PlatformGrok, rec.Header().Get(headerSub2APIUpstreamPlatform))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerSub2APIRequestedModel))
	require.Equal(t, PlatformGrok, rec.Header().Get(headerSub2APISelectedAccountPlatform))
	require.Equal(t, provenanceLevelDirectOfficial, rec.Header().Get(headerSub2APIProvenanceLevel))
	require.Equal(t, "trusted-request", rec.Header().Get(headerUpstreamRequestID))
}

func TestWriteOpenAIUpstreamProvenanceCustomRelayStaysLogicalOnly(t *testing.T) {
	t.Parallel()

	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok}, "grok-4.5", "grok-4.5", "grok-4.5",
		"https://relay.example/v1/chat/completions", http.Header{})
	promoteOpenAIActualModel(c, "grok-4.5")

	require.Equal(t, PlatformGrok, rec.Header().Get(headerSub2APISelectedAccountPlatform))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerSub2APISentUpstreamModel))
	require.Equal(t, provenanceLevelLogicalOnly, rec.Header().Get(headerSub2APIProvenanceLevel))
	require.Empty(t, rec.Header().Get(headerSub2APIPhysicalPlatform))
	require.Empty(t, rec.Header().Get(headerSub2APIUpstreamPlatform))
	require.Empty(t, rec.Header().Get(headerSub2APIActualModel))
	require.Empty(t, rec.Header().Get(headerActualModel))
}

func TestWriteOpenAIUpstreamProvenanceDoesNotClaimActualBeforeProtocolSuccess(t *testing.T) {
	t.Parallel()

	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok}, "grok-4.5", "grok-4.5", "grok-4.5",
		"https://api.x.ai/v1/responses", http.Header{})

	require.Equal(t, provenanceLevelDirectOfficial, rec.Header().Get(headerSub2APIProvenanceLevel))
	require.Equal(t, PlatformGrok, rec.Header().Get(headerSub2APIPhysicalPlatform))
	require.Equal(t, "grok-4.5", rec.Header().Get(headerSub2APISentUpstreamModel))
	require.Empty(t, rec.Header().Get(headerSub2APIActualModel))
	require.Empty(t, rec.Header().Get(headerActualModel))
}

func TestWriteOpenAIUpstreamProvenancePreservesEveryMappingHop(t *testing.T) {
	t.Parallel()

	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{
		Matched:        true,
		PublicModel:    "public-a",
		TargetPlatform: PlatformOpenAI,
		UpstreamModel:  "composite-b",
	})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformOpenAI}, "channel-c", "gpt-5.6", "gpt-5.6-sol",
		"https://api.openai.com/v1/responses", http.Header{})

	require.Equal(t, "public-a", rec.Header().Get(headerSub2APIRequestedModel))
	require.Equal(t, "gpt-5.6-sol", rec.Header().Get(headerSub2APISentUpstreamModel))
	require.Equal(t, "public-a→composite-b→channel-c→gpt-5.6→gpt-5.6-sol", rec.Header().Get(headerModelMappingChain))
}

func TestWriteOpenAIUpstreamProvenanceUsesFinalResponseURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		finalURL     string
		wantLevel    string
		wantPhysical string
	}{
		{name: "official initial redirected to relay", finalURL: "https://relay.example/v1/responses", wantLevel: provenanceLevelLogicalOnly},
		{name: "relay initial redirected to official", finalURL: "https://api.x.ai/v1/responses", wantLevel: provenanceLevelDirectOfficial, wantPhysical: PlatformGrok},
		{name: "final destination is insecure", finalURL: "http://api.x.ai/v1/responses", wantLevel: provenanceLevelLogicalOnly},
		{name: "official lookalike", finalURL: "https://api.x.ai.attacker.example/v1/responses", wantLevel: provenanceLevelLogicalOnly},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{})
			resp := &http.Response{Request: mustNewProvenanceRequest(t, tt.finalURL)}
			writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok}, "grok-4.5", "grok-4.5", "grok-4.5",
				upstreamResponseRequestURL(resp), http.Header{})

			require.Equal(t, tt.wantLevel, rec.Header().Get(headerSub2APIProvenanceLevel))
			require.Equal(t, tt.wantPhysical, rec.Header().Get(headerSub2APIPhysicalPlatform))
		})
	}
}

func TestWriteOpenAIUpstreamProvenanceNilFinalResponseURLFailsClosed(t *testing.T) {
	t.Parallel()

	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok}, "grok-4.5", "grok-4.5", "grok-4.5",
		upstreamResponseRequestURL(&http.Response{}), http.Header{})
	promoteOpenAIActualModel(c, "grok-4.5")

	require.Equal(t, provenanceLevelLogicalOnly, rec.Header().Get(headerSub2APIProvenanceLevel))
	require.Empty(t, rec.Header().Get(headerSub2APIPhysicalPlatform))
	require.Empty(t, rec.Header().Get(headerActualModel))
}

func TestPromoteOpenAIActualModelAfterHeadersCommittedFailsClosed(t *testing.T) {
	t.Parallel()

	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok}, "grok-4.5", "grok-4.5", "grok-4.5",
		"https://api.x.ai/v1/responses", http.Header{})
	c.Writer.WriteHeader(http.StatusOK)
	c.Writer.WriteHeaderNow()
	promoteOpenAIActualModel(c, "grok-4.5")

	require.Empty(t, rec.Header().Get(headerActualModel))
}

func TestClearOpenAIUpstreamAttemptProvenanceRemovesPriorAttempt(t *testing.T) {
	t.Parallel()

	rec, c := newUpstreamProvenanceTestContext(t, CompositeRouteDecision{})
	writeOpenAIUpstreamProvenance(c, &Account{Platform: PlatformGrok}, "grok-4.5", "grok-4.5", "grok-4.5",
		"https://api.x.ai/v1/responses", http.Header{
			"Xai-Request-Id": []string{"first-attempt"},
		})
	clearOpenAIUpstreamAttemptProvenance(c)

	for _, name := range authoritativeUpstreamProvenanceHeaders {
		require.Empty(t, rec.Header().Values(name), name)
	}
}

func newUpstreamProvenanceTestContext(t *testing.T, decision CompositeRouteDecision) (*httptest.ResponseRecorder, *gin.Context) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	if decision.Matched {
		c.Request = c.Request.WithContext(WithCompositeRouteDecision(c.Request.Context(), decision))
	}
	return rec, c
}

func mustNewProvenanceRequest(t *testing.T, rawURL string) *http.Request {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, rawURL, nil)
	require.NoError(t, err)
	return req
}
