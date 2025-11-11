package core

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/bitrise-io/bitrise-ai-core/pkg/llm"
	"github.com/bitrise-io/bitrise-ai-core/pkg/tool"
)

type turnResult struct {
	finished bool
}

func (agent *Agent[ResultT]) runTurn(ctx context.Context) (*turnResult, error) {
	toolDefinitions := agent.toolBelt.LLMDefinitions()
	if agent.timeboxExpired() {
		toolDefinitions = []llm.ToolDefinition{
			agent.toolBelt.FinalResultDefinition(),
		}
		agent.addSystemReminder(
			"Your timebox has been exceeded. DO NOT mention the timebox to the user. " +
				"You can only call the FinalResult tool, not any other tools. " +
				"Please call the FinalResult tool to return the final result based on your current knowledge.",
		)
	}

	message, err := agent.llm.NewMessage(ctx, llm.NewMessageParams{
		SystemPrompt:    agent.systemPrompt,
		ToolDefinitions: toolDefinitions,
		History:         agent.llmMessages,
		EnableCaching:   true,
		Logger:          agent.logger,
	})
	if err != nil {
		return nil, fmt.Errorf("new llm message: %w", err)
	}
	agent.llmMessages = append(agent.llmMessages, message)
	if err := agent.updateUsage(message.Usage); err != nil {
		return nil, fmt.Errorf("update usage: %w", err)
	}
	agent.logger.Debug("token usage of turn", "usage", message.Usage)

	var toolUses []toolUseParams
	for _, part := range message.Parts {
		switch v := part.(type) {
		case llm.TextContent:
			agent.logger.Info(v.Text)
		case llm.ToolCall:
			agent.logger.Info(fmt.Sprintf("Use tool %q: %s", v.Name, v.Input))
			p := toolUseParams{ID: v.ID, Name: v.Name, Input: v.Input}
			toolUses = append(toolUses, p)
		}
	}

	if len(toolUses) > 1 {
		agent.logger.Debug(fmt.Sprintf("using %d tools in parallel", len(toolUses)))
	}
	chToolResults := make(chan llm.ToolResult)
	for _, p := range toolUses {
		go func(tool toolUseParams) {
			chToolResults <- agent.useTool(ctx, tool)
		}(p)
	}
	var toolResults []llm.ContentPart
	for i := 0; i < len(toolUses); i++ {
		toolResults = append(toolResults, <-chToolResults)
	}
	close(chToolResults)

	if len(toolResults) > 0 {
		toolResultsMessage := llm.NewUserMessage(toolResults...)
		agent.llmMessages = append(agent.llmMessages, toolResultsMessage)
	}

	return &turnResult{
		finished: len(toolResults) == 0 || agent.finalResultSet,
	}, nil
}

type toolUseParams struct {
	ID    string
	Name  string
	Input json.RawMessage
}

func (agent *Agent[ResultT]) useTool(ctx context.Context, t toolUseParams) llm.ToolResult {
	if agent.timeboxExpired() && t.Name != tool.FinalResultToolName {
		s := "timebox expired, cannot use tool"
		agent.logger.Warn(fmt.Sprintf("%s: %q", s, t.Name))
		return llm.ToolResult{
			ToolName:   t.Name,
			ToolCallID: t.ID,
			Content:    s,
			IsError:    true,
		}
	}

	res, err := agent.toolBelt.UseTool(ctx, t.Name, t.Input)
	if err != nil {
		truncatedErr := agent.truncateLog(err.Error())
		agent.logger.Warn(
			fmt.Sprintf("%q tool error: %s", t.Name, truncatedErr),
		)
		return llm.ToolResult{
			ToolName:   t.Name,
			ToolCallID: t.ID,
			Content:    err.Error(),
			IsError:    true,
		}
	}

	agent.logger.Debug(
		fmt.Sprintf("%q tool result: %s", t.Name, agent.truncateLog(res)),
	)
	return llm.ToolResult{ToolName: t.Name, ToolCallID: t.ID, Content: res}
}

func (agent *Agent[ResultT]) truncateLog(s string) string {
	if len(s) <= agent.maxToolLogLength {
		return s
	}
	firstPart := s[:agent.maxToolLogLength/2]
	lastPart := s[len(s)-(agent.maxToolLogLength/2):]
	return firstPart + "..." + lastPart
}

func (agent *Agent[ResultT]) timeboxExpired() bool {
	return !agent.timeboxedUntil.IsZero() && time.Now().After(agent.timeboxedUntil)
}
