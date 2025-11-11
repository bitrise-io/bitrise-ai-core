package llm

import (
	"encoding/json"
	"fmt"
)

type Message struct {
	Role  MessageRole
	Parts []ContentPart
	Usage TokenUsage
}

func NewUserMessage(parts ...ContentPart) Message {
	return Message{Role: RoleUser, Parts: parts}
}

type MessageRole string

const (
	RoleAssistant MessageRole = "assistant"
	RoleUser      MessageRole = "user"
)

type ContentPart interface {
	isPart()
}

type TextContent struct {
	Text string
}

func (tc TextContent) String() string {
	return tc.Text
}

func (TextContent) isPart() {}

type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

func (ToolCall) isPart() {}

type ToolResult struct {
	ToolCallID string
	ToolName   string
	Content    string
	IsError    bool
}

func (ToolResult) isPart() {}

type TokenUsage struct {
	InputTokens         int64
	OutputTokens        int64
	CacheCreationTokens int64
	CacheReadTokens     int64
}

func (ts TokenUsage) String() string {
	return fmt.Sprintf(
		"input: %d, output: %d, cache creation: %d, cache read: %d",
		ts.InputTokens, ts.OutputTokens, ts.CacheCreationTokens, ts.CacheReadTokens,
	)
}

func (ts TokenUsage) Total() int64 {
	return ts.InputTokens + ts.OutputTokens + ts.CacheCreationTokens + ts.CacheReadTokens
}
