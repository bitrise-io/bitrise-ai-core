package llm

import (
	"context"
	"log/slog"

	"github.com/invopop/jsonschema"
)

type NewMessageParams struct {
	SystemPrompt    string
	ToolDefinitions []ToolDefinition
	History         []Message
	EnableCaching   bool
	Logger          *slog.Logger
}

type ToolDefinition struct {
	Name        string
	Description string
	Schema      *jsonschema.Schema
}

type Provider interface {
	NewMessage(ctx context.Context, params NewMessageParams) (Message, error)
}
