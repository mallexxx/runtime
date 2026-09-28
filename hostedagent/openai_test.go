package hostedagent

import (
	"encoding/json"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// testFuncName is the function name used across these fixtures.
const testFuncName = "list_vaults"

// JSON Schema type keywords expected on the wire. They are lower case: genai
// spells its own enum values in upper case, and sending that form is rejected.
const (
	schemaTypeObject  = "object"
	schemaTypeString  = "string"
	schemaTypeInteger = "integer"
	schemaTypeNumber  = "number"
	schemaTypeBoolean = "boolean"
	schemaTypeArray   = "array"
)

func TestOpenAIToolsFromConfig_NilConfig(t *testing.T) {
	defs := openAIToolsFromConfig(nil)
	if len(defs) != 0 {
		t.Fatalf("expected empty, got %d", len(defs))
	}
}

func TestOpenAIToolsFromConfig_EmptyTools(t *testing.T) {
	cfg := &genai.GenerateContentConfig{}
	defs := openAIToolsFromConfig(cfg)
	if len(defs) != 0 {
		t.Fatalf("expected empty, got %d", len(defs))
	}
}

func TestOpenAIToolsFromConfig_SingleFunction(t *testing.T) {
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        testFuncName,
						Description: "List all vaults",
					},
				},
			},
		},
	}

	defs := openAIToolsFromConfig(cfg)
	if len(defs) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(defs))
	}
	if defs[0].Type != "function" {
		t.Fatalf("expected type 'function', got %q", defs[0].Type)
	}

	var fn openAIFunction
	if err := json.Unmarshal(defs[0].Function, &fn); err != nil {
		t.Fatalf("unmarshal function: %v", err)
	}
	if fn.Name != testFuncName {
		t.Fatalf("expected name 'list_vaults', got %q", fn.Name)
	}
	if fn.Description != "List all vaults" {
		t.Fatalf("expected description 'List all vaults', got %q", fn.Description)
	}
}

func TestOpenAIToolsFromConfig_WithParameters(t *testing.T) {
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        "search_notes",
						Description: "Search notes",
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"query": {Type: genai.TypeString},
							},
							Required: []string{"query"},
						},
					},
				},
			},
		},
	}

	defs := openAIToolsFromConfig(cfg)
	if len(defs) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(defs))
	}

	var fn openAIFunction
	if err := json.Unmarshal(defs[0].Function, &fn); err != nil {
		t.Fatalf("unmarshal function: %v", err)
	}
	var params map[string]any
	if err := json.Unmarshal(fn.Parameters, &params); err != nil {
		t.Fatalf("unmarshal parameters: %v", err)
	}
	if params["type"] != schemaTypeObject {
		t.Fatalf("expected the JSON Schema keyword %q, got %v", schemaTypeObject, params["type"])
	}
}

func TestOpenAIToolsFromConfig_MultipleFunctions(t *testing.T) {
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{Name: "tool_a"},
					{Name: "tool_b"},
				},
			},
		},
	}

	defs := openAIToolsFromConfig(cfg)
	if len(defs) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(defs))
	}
}

func TestParseChatResponse_TextOnly(t *testing.T) {
	resp := `{"choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"}}]}`
	content, err := parseChatResponse([]byte(resp))
	if err != nil {
		t.Fatalf("parseChatResponse() error = %v", err)
	}
	if len(content.Parts) != 1 || content.Parts[0].Text != "Hello!" {
		t.Fatalf("expected text 'Hello!', got %#v", content.Parts)
	}
}

func TestParseChatResponse_ToolCallsOnly(t *testing.T) {
	resp := `{
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":null,
				"tool_calls":[
					{
						"id":"call_1",
						"type":"function",
						"function":{"name":"list_vaults","arguments":"{}"}
					}
				]
			}
		}]
	}`
	content, err := parseChatResponse([]byte(resp))
	if err != nil {
		t.Fatalf("parseChatResponse() error = %v", err)
	}
	if len(content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(content.Parts))
	}
	if content.Parts[0].FunctionCall == nil {
		t.Fatal("expected FunctionCall part, got nil")
	}
	if content.Parts[0].FunctionCall.Name != testFuncName {
		t.Fatalf("expected function name 'list_vaults', got %q", content.Parts[0].FunctionCall.Name)
	}
}

func TestParseChatResponse_TextAndToolCalls(t *testing.T) {
	resp := `{
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"I will search for that.",
				"tool_calls":[
					{
						"id":"call_1",
						"type":"function",
						"function":{"name":"search_notes","arguments":"{\"query\":\"test\"}"}
					}
				]
			}
		}]
	}`
	content, err := parseChatResponse([]byte(resp))
	if err != nil {
		t.Fatalf("parseChatResponse() error = %v", err)
	}
	if len(content.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(content.Parts))
	}
	if content.Parts[0].Text != "I will search for that." {
		t.Fatalf("expected text, got %q", content.Parts[0].Text)
	}
	if content.Parts[1].FunctionCall == nil || content.Parts[1].FunctionCall.Name != "search_notes" {
		t.Fatal("expected FunctionCall in second part")
	}
}

func TestParseChatResponse_MissingChoices(t *testing.T) {
	resp := `{"choices":[]}`
	_, err := parseChatResponse([]byte(resp))
	if err == nil {
		t.Fatal("expected error for empty choices")
	}
}

func TestParseChatResponse_EmptyContentAndToolCalls(t *testing.T) {
	resp := `{"choices":[{"index":0,"message":{"role":"assistant","content":""}}]}`
	_, err := parseChatResponse([]byte(resp))
	if err == nil {
		t.Fatal("expected error for empty content and no tool_calls")
	}
}

func TestContentToOpenAI_NilContent(t *testing.T) {
	text, calls, responses := contentToOpenAI(nil)
	if text != "" || len(calls) != 0 || len(responses) != 0 {
		t.Fatalf("expected empty, got text=%q calls=%d responses=%d", text, len(calls), len(responses))
	}
}

func TestContentToOpenAI_TextOnly(t *testing.T) {
	c := genai.NewContentFromText("hello", genai.RoleModel)
	text, calls, _ := contentToOpenAI(c)
	if text != "hello" || len(calls) != 0 {
		t.Fatalf("expected text 'hello', got text=%q calls=%d", text, len(calls))
	}
}

func TestContentToOpenAI_FunctionCall(t *testing.T) {
	c := &genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{
			genai.NewPartFromFunctionCall(testFuncName, map[string]any{}),
		},
	}
	text, calls, _ := contentToOpenAI(c)
	if text != "" || len(calls) != 1 {
		t.Fatalf("expected 1 call, got text=%q calls=%d", text, len(calls))
	}
	if calls[0].Function.Name != testFuncName {
		t.Fatalf("expected 'list_vaults', got %q", calls[0].Function.Name)
	}
}

func TestContentToOpenAI_FunctionResponse(t *testing.T) {
	c := &genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{
			genai.NewPartFromFunctionResponse(testFuncName, map[string]any{"result": "ok"}),
		},
	}
	text, calls, responses := contentToOpenAI(c)
	if text != "" || len(calls) != 0 {
		t.Fatalf("expected no calls, got text=%q calls=%d", text, len(calls))
	}
	if len(responses) != 1 {
		t.Fatalf("expected 1 tool response, got %d", len(responses))
	}
	if responses[0].Role != "tool" {
		t.Fatalf("expected role 'tool', got %q", responses[0].Role)
	}
	if responses[0].ToolCallID != testFuncName {
		t.Fatalf("expected ToolCallID %q, got %q", testFuncName, responses[0].ToolCallID)
	}
}

func TestBuildChatRequest_WithTools(t *testing.T) {
	req := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{Name: testFuncName},
				},
			},
		},
	}

	llmReq := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
		Config: req,
	}

	payload, err := buildChatRequest(llmReq, "test-model")
	if err != nil {
		t.Fatalf("buildChatRequest() error = %v", err)
	}
	if len(payload.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(payload.Tools))
	}
	if payload.Tools[0].Type != "function" {
		t.Fatalf("expected type 'function', got %q", payload.Tools[0].Type)
	}
}

func TestBuildChatRequest_WithoutTools(t *testing.T) {
	llmReq := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
	}

	payload, err := buildChatRequest(llmReq, "test-model")
	if err != nil {
		t.Fatalf("buildChatRequest() error = %v", err)
	}
	if len(payload.Tools) != 0 {
		t.Fatalf("expected 0 tools, got %d", len(payload.Tools))
	}
}

func TestBuildChatRequest_MissingContent(t *testing.T) {
	_, err := buildChatRequest(&model.LLMRequest{}, "test-model")
	if err == nil {
		t.Fatal("expected error for missing content")
	}
}
