package core

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/bitrise-io/bitrise-ai-core/pkg/llm"
)

func (agent *Agent[ResultT]) restoreSession() error {
	if agent.sessionFilePath == "" {
		return nil
	}

	agent.logger.Debug("restoring session", "file_path", agent.sessionFilePath)
	file, err := os.Open(agent.sessionFilePath)
	switch {
	case os.IsNotExist(err):
		agent.logger.Debug("session file does not exist, starting new session")
		return nil
	case err != nil:
		return fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	registerTypesForSession()
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&agent.llmMessages); err != nil {
		return fmt.Errorf("gob decode: %w", err)
	}
	return nil
}

func (agent *Agent[ResultT]) saveSession() error {
	if agent.sessionFilePath == "" {
		return nil
	}
	agent.logger.Debug("saving session", "file_path", agent.sessionFilePath)
	file, err := os.Create(agent.sessionFilePath)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer file.Sync()
	defer file.Close()

	registerTypesForSession()
	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(agent.llmMessages); err != nil {
		return fmt.Errorf("gob encode: %w", err)
	}
	return nil
}

func registerTypesForSession() {
	gob.Register(llm.Message{})
	gob.Register(llm.TextContent{})
	gob.Register(llm.ToolCall{})
	gob.Register(llm.ToolResult{})
	gob.Register(llm.TokenUsage{})
}
