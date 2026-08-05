package service

import (
	"net/http"
	"net/url"
	"strings"
	"unicode"

	"github.com/Wei-Shaw/sub2api/internal/pkg/xai"
	"github.com/gin-gonic/gin"
)

const (
	headerSub2APIRequestedModel          = "X-Sub2API-Requested-Model"
	headerSub2APISelectedAccountPlatform = "X-Sub2API-Selected-Account-Platform"
	headerSub2APISentUpstreamModel       = "X-Sub2API-Sent-Upstream-Model"
	headerSub2APIProvenanceLevel         = "X-Sub2API-Provenance-Level"
	headerSub2APIPhysicalPlatform        = "X-Sub2API-Physical-Platform"
	headerSub2APIActualModel             = "X-Sub2API-Actual-Model"
	headerSub2APIUpstreamPlatform        = "X-Sub2API-Upstream-Platform"
	headerUpstreamModel                  = "X-Upstream-Model"
	headerActualModel                    = "X-Actual-Model"
	headerSub2APIUpstreamModel           = "X-Sub2API-Upstream-Model"
	headerModelMappingChain              = "X-Model-Mapping-Chain"
	headerSub2APIModelMappingChain       = "X-Sub2API-Model-Mapping-Chain"
	headerUpstreamRequestID              = "X-Upstream-Request-Id"

	provenanceLevelDirectOfficial = "direct-official"
	provenanceLevelLogicalOnly    = "logical-only"
)

var authoritativeUpstreamProvenanceHeaders = []string{
	headerSub2APIRequestedModel,
	headerSub2APISelectedAccountPlatform,
	headerSub2APISentUpstreamModel,
	headerSub2APIProvenanceLevel,
	headerSub2APIPhysicalPlatform,
	headerSub2APIActualModel,
	headerSub2APIUpstreamPlatform,
	headerUpstreamModel,
	headerActualModel,
	headerSub2APIUpstreamModel,
	headerModelMappingChain,
	headerSub2APIModelMappingChain,
	headerUpstreamRequestID,
}

// writeOpenAIUpstreamProvenance separates logical routing evidence from
// physical-provider evidence. A configured account platform or sent model is
// never promoted to physical/actual provenance for a custom relay.
func writeOpenAIUpstreamProvenance(
	c *gin.Context,
	account *Account,
	requestModel string,
	billingModel string,
	upstreamModel string,
	upstreamURL string,
	upstreamHeaders http.Header,
) {
	if c == nil || c.Writer == nil || c.Writer.Written() || account == nil {
		return
	}

	requestModel = normalizeProvenanceHeaderValue(requestModel, 128)
	billingModel = normalizeProvenanceHeaderValue(billingModel, 128)
	upstreamModel = normalizeProvenanceHeaderValue(upstreamModel, 128)
	requestedModel := requestModel
	if c.Request != nil {
		if publicModel, ok := RequestedPublicModelFromContext(c.Request.Context()); ok {
			requestedModel = normalizeProvenanceHeaderValue(publicModel, 128)
		}
	}

	headers := c.Writer.Header()
	setOrDeleteHeader(headers, headerSub2APIRequestedModel, requestedModel)
	selectedPlatform := normalizeProvenanceHeaderValue(account.Platform, 64)
	setOrDeleteHeader(headers, headerSub2APISelectedAccountPlatform, selectedPlatform)
	setOrDeleteHeader(headers, headerSub2APISentUpstreamModel, upstreamModel)

	level, physicalPlatform := classifyOpenAIUpstreamProvenance(account, upstreamURL)
	setOrDeleteHeader(headers, headerSub2APIProvenanceLevel, level)
	setOrDeleteHeader(headers, headerSub2APIPhysicalPlatform, physicalPlatform)
	// Legacy physical-platform header remains for existing strict consumers,
	// but is now emitted only when the TLS destination is an official provider.
	setOrDeleteHeader(headers, headerSub2APIUpstreamPlatform, physicalPlatform)

	// Receiving an HTTP response only proves the selected/sent route. Do not
	// claim an actual model until the caller has validated a protocol-level
	// success (for example response.completed or a valid Chat Completion body).
	setOrDeleteHeader(headers, headerSub2APIActualModel, "")
	setOrDeleteHeader(headers, headerUpstreamModel, "")
	setOrDeleteHeader(headers, headerActualModel, "")
	setOrDeleteHeader(headers, headerSub2APIUpstreamModel, "")

	resolvedCompositeModel := ""
	if c.Request != nil {
		if model, ok := ResolvedUpstreamModelFromContext(c.Request.Context()); ok {
			resolvedCompositeModel = normalizeProvenanceHeaderValue(model, 128)
		}
	}
	chain := buildUpstreamModelMappingChain(requestedModel, resolvedCompositeModel, requestModel, billingModel, upstreamModel)
	setOrDeleteHeader(headers, headerModelMappingChain, chain)
	setOrDeleteHeader(headers, headerSub2APIModelMappingChain, chain)

	requestID := optionalSingleUpstreamRequestID(upstreamHeaders)
	setOrDeleteHeader(headers, headerUpstreamRequestID, normalizeProvenanceHeaderValue(requestID, 256))
}

// promoteOpenAIActualModel publishes a provider-reported model only after a
// protocol-level success has been validated. Streaming callers deliberately do
// not promote after headers have been committed: sent/physical provenance is
// still available, while completed-model provenance remains fail closed.
func promoteOpenAIActualModel(c *gin.Context, actualModel string) {
	if c == nil || c.Writer == nil || c.Writer.Written() {
		return
	}
	headers := c.Writer.Header()
	if headers.Get(headerSub2APIProvenanceLevel) != provenanceLevelDirectOfficial ||
		strings.TrimSpace(headers.Get(headerSub2APIPhysicalPlatform)) == "" {
		return
	}
	actualModel = normalizeProvenanceHeaderValue(actualModel, 128)
	if actualModel == "" {
		return
	}
	setOrDeleteHeader(headers, headerSub2APIActualModel, actualModel)
	setOrDeleteHeader(headers, headerUpstreamModel, actualModel)
	setOrDeleteHeader(headers, headerActualModel, actualModel)
	setOrDeleteHeader(headers, headerSub2APIUpstreamModel, actualModel)
}

// classifyOpenAIUpstreamProvenance deliberately recognizes only official
// provider TLS hosts. Custom relays remain logical-only until an authenticated
// relay-attestation contract exists.
func classifyOpenAIUpstreamProvenance(account *Account, rawURL string) (string, string) {
	if account == nil {
		return provenanceLevelLogicalOnly, ""
	}
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || !strings.EqualFold(parsed.Scheme, "https") || parsed.Hostname() == "" {
		return provenanceLevelLogicalOnly, ""
	}
	host := strings.ToLower(strings.TrimSpace(parsed.Hostname()))
	switch account.Platform {
	case PlatformGrok:
		if xai.IsOfficialBaseURLHost(host) {
			return provenanceLevelDirectOfficial, PlatformGrok
		}
	case PlatformOpenAI:
		if host == "api.openai.com" || host == "chatgpt.com" {
			return provenanceLevelDirectOfficial, PlatformOpenAI
		}
	}
	return provenanceLevelLogicalOnly, ""
}

func upstreamResponseRequestURL(resp *http.Response) string {
	if resp == nil || resp.Request == nil || resp.Request.URL == nil {
		return ""
	}
	return resp.Request.URL.String()
}

// clearOpenAIUpstreamAttemptProvenance starts a fresh attempt. Requested model
// identity lives in request context and is reconstructed after a real HTTP
// response; writer headers must never leak a prior failover attempt.
func clearOpenAIUpstreamAttemptProvenance(c *gin.Context) {
	if c == nil || c.Writer == nil || c.Writer.Written() {
		return
	}
	headers := c.Writer.Header()
	for _, name := range authoritativeUpstreamProvenanceHeaders {
		headers.Del(name)
	}
}

// ClearOpenAIUpstreamAttemptProvenance lets the handler clear the previous
// HTTP attempt before account selection/slot admission. This prevents a final
// selection or admission error from inheriting provenance from an earlier
// upstream response.
func ClearOpenAIUpstreamAttemptProvenance(c *gin.Context) {
	clearOpenAIUpstreamAttemptProvenance(c)
}

// optionalSingleUpstreamRequestID publishes an audit identifier only when the
// upstream supplied exactly one non-empty value across all recognized aliases.
// Duplicate values are ambiguous even when their text matches: collapsing
// them here would hide evidence that strict downstream consumers need in order
// to reject polluted provenance.
func optionalSingleUpstreamRequestID(headers http.Header) string {
	aliases := map[string]struct{}{
		"x-request-id":   {},
		"xai-request-id": {},
		"request-id":     {},
	}
	values := make([]string, 0, 1)
	for key, rawValues := range headers {
		if _, ok := aliases[strings.ToLower(strings.TrimSpace(key))]; !ok {
			continue
		}
		for _, value := range rawValues {
			values = append(values, strings.TrimSpace(value))
		}
	}
	if len(values) != 1 || values[0] == "" {
		return ""
	}
	return values[0]
}

func buildUpstreamModelMappingChain(models ...string) string {
	chain := make([]string, 0, len(models))
	for _, model := range models {
		model = strings.TrimSpace(model)
		if model == "" || (len(chain) > 0 && chain[len(chain)-1] == model) {
			continue
		}
		chain = append(chain, model)
	}
	return strings.Join(chain, "→")
}

func setOrDeleteHeader(headers http.Header, name, value string) {
	value = normalizeProvenanceHeaderValue(value, 512)
	if value == "" {
		headers.Del(name)
		return
	}
	headers.Set(name, value)
}

func normalizeProvenanceHeaderValue(value string, maxRunes int) string {
	value = strings.TrimSpace(strings.Map(func(r rune) rune {
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, value))
	if value == "" || maxRunes <= 0 {
		return ""
	}
	runes := []rune(value)
	if len(runes) > maxRunes {
		return string(runes[:maxRunes])
	}
	return value
}

func snapshotAuthoritativeUpstreamProvenance(headers http.Header) http.Header {
	snapshot := make(http.Header, len(authoritativeUpstreamProvenanceHeaders))
	for _, name := range authoritativeUpstreamProvenanceHeaders {
		for _, value := range headers.Values(name) {
			snapshot.Add(name, value)
		}
	}
	return snapshot
}

func restoreAuthoritativeUpstreamProvenance(headers, snapshot http.Header) {
	for _, name := range authoritativeUpstreamProvenanceHeaders {
		headers.Del(name)
		for _, value := range snapshot.Values(name) {
			headers.Add(name, value)
		}
	}
}
