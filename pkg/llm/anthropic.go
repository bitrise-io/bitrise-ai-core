package llm

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	backoff "github.com/cenkalti/backoff/v4"
)

type AnthropicProvider struct {
	Client          anthropic.Client
	Model           string
	MaxOutputTokens int
}

func (ap *AnthropicProvider) NewMessage(ctx context.Context, params NewMessageParams) (Message, error) {
	// The SDK has built-in retry logic for transient errors, but it doesn't
	// handle all possible cases like "connection reset by peer".
	// For simplicity, we'll retry everything for now.
	fn := func() (Message, error) {
		return ap.tryNewMessage(ctx, params)
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

func (ap *AnthropicProvider) tryNewMessage(ctx context.Context, params NewMessageParams) (Message, error) {
	systemPrompt := anthropic.TextBlockParam{Text: params.SystemPrompt}
	tools := ap.convertTools(params.ToolDefinitions)
	messages, err := ap.convertMessages(params.History, params.Logger)
	if err != nil {
		return Message{}, fmt.Errorf("convert messages: %w", err)
	}

	if params.EnableCaching {
		cacheFlag := anthropic.CacheControlEphemeralParam{Type: "ephemeral"}
		systemPrompt.CacheControl = cacheFlag
		if len(tools) > 0 {
			tools[len(tools)-1].OfTool.CacheControl = cacheFlag
		}
		if err := ap.setCachedParams(messages); err != nil {
			return Message{}, fmt.Errorf("set cached params: %w", err)
		}
	}

	messageParams := anthropic.MessageNewParams{
		Model:       anthropic.Model(ap.Model),
		System:      []anthropic.TextBlockParam{systemPrompt},
		Tools:       tools,
		Messages:    messages,
		MaxTokens:   int64(ap.MaxOutputTokens),
		Temperature: anthropic.Float(0.0),
	}

	// By default the client retries all transient errors 2 times.
	// Can be overridden using option.WithMaxRetries.
	message, err := ap.Client.Messages.New(ctx, messageParams)
	if err != nil {
		return Message{}, fmt.Errorf("new message: %w", err)
	}

	resultMessage := Message{
		Role: RoleAssistant,
		Usage: TokenUsage{
			InputTokens:         message.Usage.InputTokens,
			OutputTokens:        message.Usage.OutputTokens,
			CacheCreationTokens: message.Usage.CacheCreationInputTokens,
			CacheReadTokens:     message.Usage.CacheReadInputTokens,
		},
	}
	for _, block := range message.Content {
		switch variant := block.AsAny().(type) {
		case anthropic.TextBlock:
			resultMessage.Parts = append(resultMessage.Parts, TextContent{
				Text: variant.Text,
			})
		case anthropic.ToolUseBlock:
			resultMessage.Parts = append(resultMessage.Parts, ToolCall{
				ID:    variant.ID,
				Name:  variant.Name,
				Input: block.Input,
			})
		}
	}
	return resultMessage, nil
}

func (ap *AnthropicProvider) convertMessages(messages []Message, logger *slog.Logger) ([]anthropic.MessageParam, error) {
	var anthropicMessages []anthropic.MessageParam

	for _, msg := range messages {
		switch msg.Role {
		case RoleUser:
			var blocks []anthropic.ContentBlockParamUnion
			for _, part := range msg.Parts {
				switch v := part.(type) {
				case TextContent:
					block := anthropic.NewTextBlock(v.Text)
					blocks = append(blocks, block)
				case ToolResult:
					block := anthropic.NewToolResultBlock(v.ToolCallID, v.Content, v.IsError)
					blocks = append(blocks, block)
				default:
					return nil, fmt.Errorf("unknown user message part type %T", v)
				}
			}
			message := anthropic.NewUserMessage(blocks...)
			anthropicMessages = append(anthropicMessages, message)

		case RoleAssistant:
			var blocks []anthropic.ContentBlockParamUnion
			for _, part := range msg.Parts {
				switch v := part.(type) {
				case TextContent:
					block := anthropic.NewTextBlock(v.Text)
					blocks = append(blocks, block)
				case ToolCall:
					block := anthropic.NewToolUseBlock(v.ID, v.Input, v.Name)
					blocks = append(blocks, block)
				default:
					return nil, fmt.Errorf("unknown assistant message part type %T", v)
				}
			}
			if len(blocks) > 0 {
				message := anthropic.NewAssistantMessage(blocks...)
				anthropicMessages = append(anthropicMessages, message)
			} else {
				logger.Warn("skipping assistant message with no content")
			}
		}
	}

	return anthropicMessages, nil
}

func (ap *AnthropicProvider) convertTools(tools []ToolDefinition) []anthropic.ToolUnionParam {
	var anthropicTools []anthropic.ToolUnionParam

	for _, tool := range tools {
		toolParam := anthropic.ToolParam{
			Name:        tool.Name,
			Description: anthropic.String(tool.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: tool.Schema.Properties,
				Required:   tool.Schema.Required,
			},
		}
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &toolParam,
		})
	}

	return anthropicTools
}

func (ap *AnthropicProvider) setCachedParams(messages []anthropic.MessageParam) error {
	// Setting numCached on the last two user messages. See for more:
	// https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/oh95z/prompt-caching
	var numCached int
	for n := len(messages) - 1; n >= 0; n-- {
		msg := messages[n]
		if msg.Role != anthropic.MessageParamRoleUser {
			continue
		}
		if len(msg.Content) == 0 {
			// We already checked this when converting messages, so this should never happen.
			return fmt.Errorf("empty user message")
		}

		content := msg.Content[len(msg.Content)-1]
		cacheFlag := anthropic.CacheControlEphemeralParam{Type: "ephemeral"}
		switch {
		case content.OfText != nil:
			content.OfText.CacheControl = cacheFlag
		case content.OfToolResult != nil:
			content.OfToolResult.CacheControl = cacheFlag
		default:
			return fmt.Errorf("unknown user message content type %T", content)
		}

		numCached++
		if numCached >= 2 {
			return nil
		}
	}
	return nil
}
