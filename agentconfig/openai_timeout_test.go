package agentconfig

import (
	"testing"
	"time"
)

func TestOpenAITimeoutNormalization(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  time.Duration
		bad   bool
	}{
		{value: "", want: 0},
		{value: "180s", want: 180 * time.Second},
		{value: "0s", bad: true},
		{value: "-1s", bad: true},
		{value: "abc", bad: true},
	} {
		cfg := Config{Type: AgentTypeOpenAI, OpenAI: &LocalAPIConfig{APIKey: "key", Model: "model", Timeout: tc.value}}
		got, err := NormalizeConfig(cfg, "")
		if tc.bad {
			if err == nil || cfg.Validate() == nil {
				t.Fatalf("timeout %q accepted", tc.value)
			}
			continue
		}
		if err != nil || got.Timeout != tc.want {
			t.Fatalf("timeout %q: got %v, err %v; want %v", tc.value, got.Timeout, err, tc.want)
		}
	}
}

func TestOpenAIThinkingNormalization(t *testing.T) {
	for _, thinking := range []string{"", "enabled", "disabled", "automatic"} {
		cfg := Config{Type: AgentTypeOpenAI, OpenAI: &LocalAPIConfig{APIKey: "key", Model: "model", Thinking: thinking}}
		got, normalizeErr := NormalizeConfig(cfg, "")
		validateErr := cfg.Validate()
		if thinking == "automatic" || thinking == "enabled" {
			if normalizeErr == nil || validateErr == nil {
				t.Fatalf("invalid thinking %q accepted", thinking)
			}
			continue
		}
		if normalizeErr != nil || validateErr != nil || got.Thinking != thinking {
			t.Fatalf("thinking %q: got %q, normalize=%v, validate=%v", thinking, got.Thinking, normalizeErr, validateErr)
		}
	}
}
