package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	backoff "github.com/cenkalti/backoff/v4"
	"github.com/google/uuid"
	"google.golang.org/genai"
)

type GeminiProvider struct {
	Client          *genai.Client
	Model           string
	MaxOutputTokens int
}

func (gp *GeminiProvider) NewMessage(ctx context.Context, params NewMessageParams) (Message, error) {
	// For simplicity, we'll retry everything for now.
	fn := func() (Message, error) {
		return gp.tryNewMessage(ctx, params)
	}
	opts := backoff.WithContext(backoff.NewExponentialBackOff(
		backoff.WithMaxElapsedTime(30*time.Second), // Retry for up to 30 seconds
	), ctx)
	notify := func(err error, d time.Duration) {
		params.Logger.Warn("retrying tryNewMessage", "delay", d, "error", err)
	}
	message, err := backoff.RetryNotifyWithData(fn, opts, notify)
	if err != nil {
		return Message{}, fmt.Errorf("new message with retries: %w", err)
	}
	return message, nil
}

func (gp *GeminiProvider) tryNewMessage(ctx context.Context, params NewMessageParams) (Message, error) {
	config := &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{
			Parts: []*genai.Part{{
				Text: params.SystemPrompt,
			}},
		},
		MaxOutputTokens: int32(gp.MaxOutputTokens),
		Tools:           gp.convertTools(params.ToolDefinitions),
	}

	allMessages, err := gp.convertMessages(params.History)
	if err != nil {
		return Message{}, fmt.Errorf("convert messages: %w", err)
	}
	history := allMessages[:len(allMessages)-1] // All but last message

	chat, err := gp.Client.Chats.Create(ctx, gp.Model, config, history)
	if err != nil {
		return Message{}, fmt.Errorf("chat create: %w", err)
	}

	lastMessage := allMessages[len(allMessages)-1]
	var lastMessageParts []genai.Part
	for _, part := range lastMessage.Parts {
		lastMessageParts = append(lastMessageParts, *part)
	}
	result, err := chat.SendMessage(ctx, lastMessageParts...)
	switch {
	case err != nil:
		return Message{}, fmt.Errorf("send message: %w", err)
	case len(result.Candidates) == 0:
		return Message{}, fmt.Errorf("no candidates in response")
	case result.Candidates[0].Content == nil:
		v := result.Candidates[0]
		return Message{}, fmt.Errorf(
			"no content in response (finish reason: %q, finish message: %q)",
			v.FinishReason, v.FinishMessage,
		)
	}

	var tokenUsage TokenUsage
	if u := result.UsageMetadata; u != nil {
		tokenUsage = TokenUsage{
			InputTokens:         int64(u.PromptTokenCount),
			OutputTokens:        int64(u.CandidatesTokenCount),
			CacheCreationTokens: 0, // Not directly provided by Gemini
			CacheReadTokens:     int64(u.CachedContentTokenCount),
		}
	}
	resultMessage := Message{
		Role:  RoleAssistant,
		Usage: tokenUsage,
	}

	for _, part := range result.Candidates[0].Content.Parts {
		switch {
		case part.Text != "":
			v := TextContent{Text: part.Text}
			resultMessage.Parts = append(resultMessage.Parts, v)
		case part.FunctionCall != nil:
			args, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return Message{}, fmt.Errorf("marshal function args: %w", err)
			}
			v := ToolCall{
				// For some reason the ID field is not set in the response.
				ID:    "call_" + uuid.New().String(),
				Name:  part.FunctionCall.Name,
				Input: args,
			}
			resultMessage.Parts = append(resultMessage.Parts, v)
		}
	}

	return resultMessage, nil
}

func (gp *GeminiProvider) convertMessages(messages []Message) ([]*genai.Content, error) {
	var gMessages []*genai.Content

	for _, msg := range messages {
		switch msg.Role {
		case RoleUser:
			var gParts []*genai.Part
			for _, part := range msg.Parts {
				switch v := part.(type) {
				case TextContent:
					gParts = append(gParts, &genai.Part{Text: v.Text})
				case ToolResult:
					response := map[string]any{}
					if v.IsError {
						errorMessage := v.Content
						if errorMessage == "" {
							errorMessage = "tool call failed"
						}
						response["error"] = errorMessage
					} else {
						output := v.Content
						if output == "" {
							output = "tool call succeeded with no output"
						}
						response["output"] = output
					}
					gParts = append(gParts, &genai.Part{
						FunctionResponse: &genai.FunctionResponse{
							Name:     v.ToolName,
							ID:       v.ToolCallID,
							Response: response,
						},
					})
				default:
					return nil, fmt.Errorf("unknown user message part type %T", v)
				}
			}
			if len(gParts) > 0 {
				gMessages = append(gMessages, &genai.Content{
					Parts: gParts,
					Role:  "user",
				})
			}

		case RoleAssistant:
			var gParts []*genai.Part
			for _, part := range msg.Parts {
				switch v := part.(type) {
				case TextContent:
					gParts = append(gParts, &genai.Part{Text: v.Text})
				case ToolCall:
					args := map[string]any{}
					if err := json.Unmarshal(v.Input, &args); err != nil {
						return nil, fmt.Errorf("unmarshal tool call args: %w", err)
					}
					gParts = append(gParts, &genai.Part{
						FunctionCall: &genai.FunctionCall{
							ID:   v.ID,
							Name: v.Name,
							Args: args,
						},
					})
				default:
					return nil, fmt.Errorf("unknown assistant message part type %T", v)
				}
			}
			if len(gParts) > 0 {
				gMessages = append(gMessages, &genai.Content{
					Parts: gParts,
					Role:  "model",
				})
			}
		}
	}

	return gMessages, nil
}

func (gp *GeminiProvider) convertTools(tools []ToolDefinition) []*genai.Tool {
	gTool := &genai.Tool{}

	for _, tool := range tools {
		v := &genai.FunctionDeclaration{
			Description:          tool.Description,
			Name:                 tool.Name,
			ParametersJsonSchema: tool.Schema,
		}
		gTool.FunctionDeclarations = append(gTool.FunctionDeclarations, v)
	}
	return []*genai.Tool{gTool}
}
