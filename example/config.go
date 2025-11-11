package main

import "time"

const (
	OUTPUT_FORMAT_TEXT = "text"
	OUTPUT_FORMAT_JSON = "json"
)

type Config struct {
	// ModelProvider is the AI provider to use. Can be one of "anthropic", "openai", "gemini".
	ModelProvider string `env:"MODEL_PROVIDER"`
	// Model is the model to use for the AI agent.
	Model string `env:"MODEL"`
	// MaxOutputTokens is the maximum number of output tokens for the AI agent.
	MaxOutputTokens int `env:"MAX_OUTPUT_TOKENS"`
	// MaxToolLogLength is the maximum length of tool use logs to keep.
	MaxToolLogLength int `env:"MAX_TOOL_LOG_LENGTH" default:"500"`
	// MaxTokenUsage is the maximum number of total tokens that can be used in the run.
	// If exceeded, the agent will fail with an error.
	// If set to 0, there is no limit.
	MaxTokenUsage int `env:"MAX_TOKEN_USAGE"`
	// Timebox is the maximum duration for the run using all tools available.
	// The agent cannot use tools after the timebox is exceeded, it must return
	// its final response as the next step based on its current knowledge.
	// Its normal to exceed the timebox, as we are waiting for the agent to
	// finish.
	// If set to 0, there is no limit.
	Timebox time.Duration `env:"TIMEBOX"`
	// CacheBust enables cache busting, which generates a unique message at the
	// start of each conversation, enforcing a fresh conversation.
	// Useful for testing the same exact input multiple times.
	// Prompt caching will still be enabled for the conversation.
	CacheBust bool `env:"CACHE_BUST"`
	// SessionFilePath is path to the file to read existing conversation history from and write the conversation to.
	SessionFilePath string `env:"SESSION_FILE_PATH"`
}
