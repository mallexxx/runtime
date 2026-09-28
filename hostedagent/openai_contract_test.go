package hostedagent

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

func TestOpenAIToolAliasesRoundTripMCPAndLongNames(t *testing.T) {
	const mcpName = "balda.control.shutdown"
	longName := strings.Repeat("a", 65)
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{Name: mcpName},
				{Name: longName},
			},
		}},
	}

	defs, openAIToRuntime, runtimeToOpenAI := openAIToolsWithAliases(cfg)
	if len(defs) != 2 {
		t.Fatalf("tool definitions = %d, want 2", len(defs))
	}
	for _, original := range []string{mcpName, longName} {
		alias := runtimeToOpenAI[original]
		if !isValidOpenAIFuncName(alias) {
			t.Fatalf("alias %q for %q is not OpenAI-safe", alias, original)
		}
		if got := openAIToRuntime[alias]; got != original {
			t.Fatalf("alias %q resolves to %q, want %q", alias, got, original)
		}
	}

	alias := runtimeToOpenAI[mcpName]
	response := fmt.Sprintf(`{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":%q,"arguments":"{}"}}]}}]}`, alias)
	content, err := parseChatResponseWithAliases([]byte(response), openAIToRuntime)
	if err != nil {
		t.Fatalf("parseChatResponseWithAliases: %v", err)
	}
	if got := content.Parts[0].FunctionCall.Name; got != mcpName {
		t.Fatalf("returned function name = %q, want original MCP name %q", got, mcpName)
	}

	messages := openAIMessagesFromRequestWithAliases(&model.LLMRequest{
		Contents: []*genai.Content{{
			Role:  genai.RoleModel,
			Parts: []*genai.Part{genai.NewPartFromFunctionCall(mcpName, map[string]any{})},
		}},
	}, runtimeToOpenAI)
	if got := messages[0].ToolCalls[0].Function.Name; got != alias {
		t.Fatalf("history function name = %q, want OpenAI alias %q", got, alias)
	}
}

func TestMarshalJSONSchemaPreservesEveryGenaiSchemaField(t *testing.T) {
	maxItems, maxLength, maxProperties := int64(5), int64(6), int64(7)
	maximum := 8.5
	minItems, minLength, minProperties := int64(1), int64(2), int64(3)
	minimum := 4.5
	nullable := true
	schema := &genai.Schema{
		AnyOf:         []*genai.Schema{{Type: genai.TypeString, Pattern: "^[a-z]+$"}},
		Default:       map[string]any{"type": "EXAMPLE_VALUE", "enabled": true},
		Description:   "complete schema",
		Enum:          []string{"a", "b"},
		Example:       map[string]any{"type": "EXAMPLE_VALUE", "name": "sample"},
		Format:        "uuid",
		Items:         &genai.Schema{Type: genai.TypeInteger, Minimum: &minimum},
		MaxItems:      &maxItems,
		MaxLength:     &maxLength,
		MaxProperties: &maxProperties,
		Maximum:       &maximum,
		MinItems:      &minItems,
		MinLength:     &minLength,
		MinProperties: &minProperties,
		Minimum:       &minimum,
		Nullable:      &nullable,
		Pattern:       "^[A-Z]+$",
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString, Title: "Name", Pattern: ".+"},
		},
		PropertyOrdering: []string{"name"},
		Required:         []string{"name"},
		Title:            "Complete",
		Type:             genai.TypeObject,
	}

	original, err := json.Marshal(schema)
	if err != nil {
		t.Fatalf("marshal original schema: %v", err)
	}
	var want map[string]any
	if err := json.Unmarshal(original, &want); err != nil {
		t.Fatalf("unmarshal original schema: %v", err)
	}
	want["type"] = schemaTypeObject
	want["items"].(map[string]any)["type"] = schemaTypeInteger
	want["properties"].(map[string]any)["name"].(map[string]any)["type"] = schemaTypeString
	want["anyOf"].([]any)[0].(map[string]any)["type"] = schemaTypeString

	encoded, err := marshalJSONSchema(schema)
	if err != nil {
		t.Fatalf("marshalJSONSchema: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatalf("unmarshal normalized schema: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("normalized schema changed fields\n got: %#v\nwant: %#v", got, want)
	}
}

// TestToolsSerializeLowercaseJSONSchema pins the JSON Schema type keywords sent
// to OpenAI. genai spells its schema types in upper case ("OBJECT"), and sending
// that form is rejected; the wire payload must carry the lower-case keywords.
func TestToolsSerializeLowercaseJSONSchema(t *testing.T) {
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name: testFuncName,
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"query": {Type: genai.TypeString},
								"limit": {Type: genai.TypeInteger},
								"ratio": {Type: genai.TypeNumber},
								"deep":  {Type: genai.TypeBoolean},
								"tags": {
									Type:  genai.TypeArray,
									Items: &genai.Schema{Type: genai.TypeString},
								},
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

	wire, err := json.Marshal(defs[0])
	if err != nil {
		t.Fatalf("marshal tool definition: %v", err)
	}
	payload := string(wire)

	// Upper-case genai enum spellings must never reach the wire.
	for _, forbidden := range []string{`"OBJECT"`, `"STRING"`, `"INTEGER"`, `"NUMBER"`, `"BOOLEAN"`, `"ARRAY"`} {
		if strings.Contains(payload, forbidden) {
			t.Fatalf("tool definition carries genai enum spelling %s: %s", forbidden, payload)
		}
	}

	var fn openAIFunction
	if err := json.Unmarshal(defs[0].Function, &fn); err != nil {
		t.Fatalf("unmarshal function: %v", err)
	}
	var params map[string]any
	if err := json.Unmarshal(fn.Parameters, &params); err != nil {
		t.Fatalf("unmarshal parameters: %v", err)
	}
	if got := params["type"]; got != schemaTypeObject {
		t.Fatalf("root type = %v, want %q", got, schemaTypeObject)
	}
	properties, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties missing or not an object: %v", params["properties"])
	}
	for name, want := range map[string]string{
		"query": schemaTypeString,
		"limit": schemaTypeInteger,
		"ratio": schemaTypeNumber,
		"deep":  schemaTypeBoolean,
		"tags":  schemaTypeArray,
	} {
		property, ok := properties[name].(map[string]any)
		if !ok {
			t.Fatalf("property %q missing: %v", name, properties[name])
		}
		if got := property["type"]; got != want {
			t.Fatalf("property %q type = %v, want %q", name, got, want)
		}
	}
	// Nested items must be converted too, not left as a raw genai value.
	tags := properties["tags"].(map[string]any)
	items, ok := tags["items"].(map[string]any)
	if !ok {
		t.Fatalf("array items missing: %v", tags["items"])
	}
	if got := items["type"]; got != schemaTypeString {
		t.Fatalf("items type = %v, want %q", got, schemaTypeString)
	}
	if required, ok := params["required"].([]any); !ok || len(required) != 1 || required[0] != "query" {
		t.Fatalf("required = %v, want [\"query\"]", params["required"])
	}
}

// TestEmptyToolResultKeepsContentField pins the tool-message wire shape. The
// OpenAI schema marks content as required for role=tool, so an empty result must
// still serialize the key instead of dropping it.
//
// Both empty shapes are covered: a nil response renders as an empty string, and
// an empty object renders as "{}" . Either way the key must be present.
func TestEmptyToolResultKeepsContentField(t *testing.T) {
	cases := []struct {
		name     string
		response map[string]any
	}{
		{name: "nil response", response: nil},
		{name: "empty response object", response: map[string]any{}},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			content := &genai.Content{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call-1",
							Name:     testFuncName,
							Response: testCase.response,
						},
					},
				},
			}

			_, _, toolResponses := contentToOpenAI(content)
			if len(toolResponses) != 1 {
				t.Fatalf("expected 1 tool response, got %d", len(toolResponses))
			}

			wire, err := json.Marshal(toolResponses[0])
			if err != nil {
				t.Fatalf("marshal tool message: %v", err)
			}
			payload := string(wire)
			if !strings.Contains(payload, `"content":`) {
				t.Fatalf("tool message dropped the required content key: %s", payload)
			}
			if !strings.Contains(payload, `"role":"tool"`) {
				t.Fatalf("tool message role missing: %s", payload)
			}
			if !strings.Contains(payload, `"tool_call_id":"call-1"`) {
				t.Fatalf("tool message call id missing: %s", payload)
			}
		})
	}
}

// TestMixedToolAndTextContentKeepsText pins that a Content holding both a tool
// result and text keeps the text. ADK permits the mixed shape; the tool branch
// used to drop the text entirely.
func TestMixedToolAndTextContentKeepsText(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "here is what I found"},
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call-7",
							Name:     testFuncName,
							Response: map[string]any{"result": "ok"},
						},
					},
				},
			},
		},
	}

	messages := openAIMessagesFromRequest(req)

	var sawTool, sawUserText bool
	for _, message := range messages {
		if message.Role == openAIRoleTool && message.Content == "ok" {
			sawTool = true
		}
		if message.Role == openAIRoleUser && message.Content == "here is what I found" {
			sawUserText = true
		}
	}
	if !sawTool {
		t.Fatalf("tool result missing from messages: %+v", messages)
	}
	if !sawUserText {
		t.Fatalf("text of the mixed content was dropped: %+v", messages)
	}
}

// TestArgumentsUnwrapRepeatedEncoding covers the provider shapes for tool-call
// arguments. Every form must reach the executor as usable arguments; the old
// decoder wrapped anything that was not a JSON object into {"raw": ...}.
func TestArgumentsUnwrapRepeatedEncoding(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want map[string]any
	}{
		{
			name: "plain object",
			raw:  `{"query":"x"}`,
			want: map[string]any{"query": "x"},
		},
		{
			name: "single-encoded string holding an object",
			raw:  `"{\"query\":\"x\"}"`,
			want: map[string]any{"query": "x"},
		},
		{
			name: "double-encoded string holding an object",
			raw:  `"\"{\\\"query\\\":\\\"x\\\"}\""`,
			want: map[string]any{"query": "x"},
		},
		{
			name: "string holding an array",
			raw:  `"[{\"query\":\"x\"}]"`,
			want: map[string]any{openAIArgumentsValueKey: []any{map[string]any{"query": "x"}}},
		},
		{
			name: "string holding a scalar",
			raw:  `"just-a-string"`,
			want: map[string]any{openAIArgumentsValueKey: "just-a-string"},
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			got := decodeFunctionArguments(json.RawMessage(testCase.raw))
			gotJSON, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal args: %v", err)
			}
			wantJSON, err := json.Marshal(testCase.want)
			if err != nil {
				t.Fatalf("marshal want: %v", err)
			}
			if string(gotJSON) != string(wantJSON) {
				t.Fatalf("decodeFunctionArguments(%s) = %s, want %s", testCase.raw, gotJSON, wantJSON)
			}
			if _, wrapped := got["raw"]; wrapped {
				t.Fatalf("arguments fell back to the opaque raw wrapper: %s", gotJSON)
			}
		})
	}
}

// TestArgumentsUnwrapIsBounded pins that a pathological nesting cannot drive an
// unbounded peel and that the result is still a map.
func TestArgumentsUnwrapIsBounded(t *testing.T) {
	// A JSON string of a JSON string of ... repeated many times.
	nested := `"value"`
	for i := 0; i < 12; i++ {
		encoded, err := json.Marshal(nested)
		if err != nil {
			t.Fatalf("marshal nested: %v", err)
		}
		nested = string(encoded)
	}

	got := decodeFunctionArguments(json.RawMessage(nested))
	if got == nil {
		t.Fatal("expected a non-nil map for a deeply nested payload")
	}
	if _, wrapped := got["raw"]; wrapped {
		t.Fatalf("unexpected raw wrapper: %v", got)
	}
}

// TestArgumentsUnwrapEmpty covers the absent-arguments case.
func TestArgumentsUnwrapEmpty(t *testing.T) {
	if got := decodeFunctionArguments(nil); got != nil {
		t.Fatalf("expected nil for empty arguments, got %v", got)
	}
}
