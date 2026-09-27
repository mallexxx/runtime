package agentfactory

import (
	"sort"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/adk/v2/tool/mcptoolset"

	"github.com/normahq/runtime/v2/agentconfig"
)

// TestHostedToolsetsCollapsesScopedEndpointDuplicate covers the regression that
// aborted every tool-using turn with `duplicate tool: "balda.control.shutdown"`.
//
// Balda registers the bundled MCP server twice:
//   - "balda"                    -> http://host:port/mcp/balda
//   - "balda-session-memory-xxx" -> http://host:port/mcp/balda?ctx=<token>
//
// Both endpoints serve the same tool list. ADK packs each toolset into one flat
// map and aborts on a repeated name, so the second toolset must be dropped.
// The scoped entry is kept because its URL carries the session context token.
func TestHostedToolsetsCollapsesScopedEndpointDuplicate(t *testing.T) {
	resolved := map[string]agentconfig.MCPServerConfig{
		"balda": {
			Type: agentconfig.MCPServerTypeHTTP,
			URL:  "http://127.0.0.1:34237/mcp/balda",
		},
		"balda-session-memory-abc123": {
			Type: agentconfig.MCPServerTypeHTTP,
			URL:  "http://127.0.0.1:34237/mcp/balda?ctx=abc123",
		},
	}

	toolsets, err := hostedToolsets(nil, resolved)
	if err != nil {
		t.Fatalf("hostedToolsets: %v", err)
	}
	if got := len(toolsets); got != 1 {
		for i, ts := range toolsets {
			t.Logf("toolset[%d] = %q", i, ts.Name())
		}
		t.Fatalf("expected 1 toolset after collapsing the shared endpoint, got %d", got)
	}

	// The survivor must be the scoped entry: its URL carries the session
	// context token that session-memory calls require. Keeping the plain
	// "balda" entry would silently strip that capability.
	kept := selectScopedOnly(t, resolved)
	if kept != "balda-session-memory-abc123" {
		t.Fatalf("expected the scoped entry to survive, got %q", kept)
	}
}

// selectScopedOnly applies the same collapse rule and returns the surviving id,
// so a test can assert which entry wins.
func selectScopedOnly(t *testing.T, resolved map[string]agentconfig.MCPServerConfig) string {
	t.Helper()
	ids := make([]string, 0, len(resolved))
	for id := range resolved {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	keeperOf := map[string]string{}
	skip := map[string]bool{}
	for _, id := range ids {
		key := mcpEndpointKey(resolved[id])
		if key == "" {
			continue
		}
		keeper, ok := keeperOf[key]
		if !ok {
			keeperOf[key] = id
			continue
		}
		if mcpURLIsScoped(resolved[id]) && !mcpURLIsScoped(resolved[keeper]) {
			skip[keeper] = true
			keeperOf[key] = id
			continue
		}
		skip[id] = true
	}
	survivors := []string{}
	for _, id := range ids {
		if !skip[id] {
			survivors = append(survivors, id)
		}
	}
	if len(survivors) != 1 {
		t.Fatalf("expected exactly 1 survivor, got %v", survivors)
	}
	return survivors[0]
}

// TestHostedToolsetsKeepsDistinctEndpoints guards against over-collapsing:
// genuinely different MCP servers must all survive.
func TestHostedToolsetsKeepsDistinctEndpoints(t *testing.T) {
	resolved := map[string]agentconfig.MCPServerConfig{
		"balda": {
			Type: agentconfig.MCPServerTypeHTTP,
			URL:  "http://127.0.0.1:34237/mcp/balda",
		},
		"executor-mcp": {
			Type: agentconfig.MCPServerTypeHTTP,
			URL:  "http://balda-executor-mcp:8890/mcp",
		},
		"obsidian": {
			Type: agentconfig.MCPServerTypeHTTP,
			URL:  "http://obsidian-mcp:3101/sse",
		},
	}

	toolsets, err := hostedToolsets(nil, resolved)
	if err != nil {
		t.Fatalf("hostedToolsets: %v", err)
	}
	if got := len(toolsets); got != 3 {
		t.Fatalf("expected 3 distinct toolsets, got %d", got)
	}
}

// TestMCPEndpointKeyStdioNeverCollapses ensures stdio servers, which have no
// shareable endpoint, are never collapsed together.
func TestMCPEndpointKeyStdioNeverCollapses(t *testing.T) {
	a := agentconfig.MCPServerConfig{Type: agentconfig.MCPServerTypeStdio, Cmd: []string{"foo"}}
	b := agentconfig.MCPServerConfig{Type: agentconfig.MCPServerTypeStdio, Cmd: []string{"bar"}}
	if mcpEndpointKey(a) != "" || mcpEndpointKey(b) != "" {
		t.Fatalf("stdio servers must not produce an endpoint key: %q %q", mcpEndpointKey(a), mcpEndpointKey(b))
	}
	if got := len([]string{mcpEndpointKey(a), mcpEndpointKey(b)}); got != 2 {
		t.Fatalf("unexpected key count %d", got)
	}
}

// TestMCPEndpointKeySSE confirms sse endpoints collapse like http ones.
func TestMCPEndpointKeySSE(t *testing.T) {
	plain := agentconfig.MCPServerConfig{Type: agentconfig.MCPServerTypeSSE, URL: "http://h:1/sse"}
	scoped := agentconfig.MCPServerConfig{Type: agentconfig.MCPServerTypeSSE, URL: "http://h:1/sse?ctx=t"}
	if mcpEndpointKey(plain) != mcpEndpointKey(scoped) {
		t.Fatalf("sse endpoints should share a key: %q vs %q", mcpEndpointKey(plain), mcpEndpointKey(scoped))
	}
	if !mcpURLIsScoped(scoped) || mcpURLIsScoped(plain) {
		t.Fatalf("scoped detection wrong for sse")
	}
}

// TestMCPEndpointKey confirms the identity used for collapsing.
func TestMCPEndpointKey(t *testing.T) {
	cases := []struct {
		name string
		url  string
		want string
	}{
		{"plain", "http://h:1/mcp/balda", "http://h:1/mcp/balda"},
		{"scoped", "http://h:1/mcp/balda?ctx=tok", "http://h:1/mcp/balda"},
		{"scoped multi", "http://h:1/mcp/balda?a=1&b=2", "http://h:1/mcp/balda"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := mcpEndpointKey(agentconfig.MCPServerConfig{
				Type: agentconfig.MCPServerTypeHTTP,
				URL:  tc.url,
			})
			if got != tc.want {
				t.Fatalf("mcpEndpointKey(%q) = %q, want %q", tc.url, got, tc.want)
			}
		})
	}
}

// TestMcpToolsetImplementsRequestProcessor records the fact that made the
// original diagnosis slow: MCP toolsets are packed by toolPreprocess (via
// Tools()), not by toolsetPreprocess, because the toolset exposes no
// ProcessRequest.
func TestMcpToolsetImplementsRequestProcessor(t *testing.T) {
	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport: &mcp.StreamableClientTransport{Endpoint: "http://127.0.0.1:1/mcp"},
	})
	if err != nil {
		t.Fatalf("mcptoolset.New: %v", err)
	}
	type requestProcessor interface {
		ProcessRequest(ctx any, req any) error
	}
	if _, ok := any(ts).(requestProcessor); ok {
		t.Fatalf("MCP toolset unexpectedly implements ProcessRequest; dedup premise changed")
	}
}
