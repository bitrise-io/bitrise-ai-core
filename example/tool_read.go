package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/bitrise-io/bitrise-ai-core/pkg/llm"
	"github.com/bitrise-io/bitrise-ai-core/pkg/tool"
)

type readInput struct {
	FilePath string `json:"file_path" jsonschema_description:"The absolute path to the file to read"`
}

var readTool = tool.Definition{
	ToolDefinition: llm.ToolDefinition{
		Name: "Read",
		Description: "Reads files from the local filesystem. This tool provides direct access to any file on the machine. " +
			"If a file path is provided, assume it's valid and attempt to read it - errors will be returned for non-existent files.",
		Schema: tool.GenerateSchema[readInput](),
	},
	UseFunc: func(ctx context.Context, llmInput json.RawMessage) (string, error) {
		var input readInput
		if err := json.Unmarshal(llmInput, &input); err != nil {
			return "", fmt.Errorf("unmarshal input: %w", err)
		}
		if input.FilePath == "" {
			return "", fmt.Errorf("file_path is required")
		}

		if _, err := os.Stat(input.FilePath); os.IsNotExist(err) {
			return "", fmt.Errorf("file does not exist: %s", input.FilePath)
		}
		file, err := os.Open(input.FilePath)
		if err != nil {
			return "", fmt.Errorf("open file: %w", err)
		}
		defer file.Close()

		content, err := io.ReadAll(file)
		if err != nil {
			return "", fmt.Errorf("read file: %w", err)
		}
		return string(content), nil
	},
}
