package llm

import (
	"context"
	"errors"
	"fmt"
	"time"

	backoff "github.com/cenkalti/backoff/v4"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
)

var reasoningEffortDefaults = map[string]openai.ReasoningEffort{
	"gpt-5":      openai.ReasoningEffortMinimal,
	"gpt-5-mini": openai.ReasoningEffortMinimal,
}

type OpenAIProvider struct {
	Client          openai.Client
	Model           string
	MaxOutputTokens int
}

func (oaip *OpenAIProvider) NewMessage(ctx context.Context, params NewMessageParams) (Message, error) {
	// The SDK has built-in retry logic for transient errors, but it doesn't
	// handle all possible cases like "connection reset by peer".
	// For simplicity, we'll retry everything for now.
	fn := func() (Message, error) {
		return oaip.tryNewMessage(ctx, params)
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

func (oaip *OpenAIProvider) tryNewMessage(ctx context.Context, params NewMessageParams) (Message, error) {
	tools := oaip.convertTools(params.ToolDefinitions)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(params.SystemPrompt),
	}
	history, err := oaip.convertMessages(params.History)
	if err != nil {
		return Message{}, fmt.Errorf("convert messages: %w", err)
	}
	messages = append(messages, history...)

	var maxTokens param.Opt[int64]
	if oaip.MaxOutputTokens > 0 {
		maxTokens = openai.Int(int64(oaip.MaxOutputTokens))
	}
	completionParams := openai.ChatCompletionNewParams{
		Model:               oaip.Model,
		Messages:            messages,
		Tools:               tools,
		MaxCompletionTokens: maxTokens,
	}

	if reasoningEffort, ok := reasoningEffortDefaults[oaip.Model]; ok {
		completionParams.ReasoningEffort = reasoningEffort
	}

	// By default the client retries all transient errors 2 times.
	// Can be overridden using option.WithMaxRetries.
	completion, err := oaip.Client.Chat.Completions.New(ctx, completionParams)
	if err != nil {
		return Message{}, fmt.Errorf("new chat completion: %w", err)
	}
	if len(completion.Choices) == 0 {
		return Message{}, errors.New("no choices in chat completion")
	}
	oaiMessage := completion.Choices[0].Message

	cachedTokens := completion.Usage.PromptTokensDetails.CachedTokens
	inputTokens := completion.Usage.PromptTokens - cachedTokens
	resultMessage := Message{
		Role: RoleAssistant,
		Usage: TokenUsage{
			InputTokens:         inputTokens,
			OutputTokens:        completion.Usage.CompletionTokens,
			CacheCreationTokens: 0, // OpenAI doesn't provide this directly
			CacheReadTokens:     cachedTokens,
		},
	}
	if oaiMessage.Content != "" {
		resultMessage.Parts = append(resultMessage.Parts, TextContent{
			Text: oaiMessage.Content,
		})
	}
	for _, v := range oaiMessage.ToolCalls {
		resultMessage.Parts = append(resultMessage.Parts, ToolCall{
			ID:    v.ID,
			Name:  v.Function.Name,
			Input: []byte(v.Function.Arguments),
		})
	}
	return resultMessage, nil
}

func (oaip *OpenAIProvider) convertMessages(messages []Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	var oaiMessages []openai.ChatCompletionMessageParamUnion

	for _, msg := range messages {
		switch msg.Role {
		case RoleUser:
			for _, part := range msg.Parts {
				switch v := part.(type) {
				case TextContent:
					message := openai.UserMessage(v.Text)
					oaiMessages = append(oaiMessages, message)
				case ToolResult:
					message := openai.ToolMessage(v.Content, v.ToolCallID)
					oaiMessages = append(oaiMessages, message)
				default:
					return nil, fmt.Errorf("unknown user message part type %T", v)
				}
			}

		case RoleAssistant:
			assistantMsg := openai.ChatCompletionAssistantMessageParam{
				Role: "assistant",
			}
			for _, part := range msg.Parts {
				switch v := part.(type) {
				case TextContent:
					assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: openai.String(v.Text),
					}
				case ToolCall:
					fn := openai.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
							ID: v.ID,
							Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
								Arguments: string(v.Input),
								Name:      v.Name,
							},
						},
					}
					assistantMsg.ToolCalls = append(assistantMsg.ToolCalls, fn)
				default:
					return nil, fmt.Errorf("unknown assistant message part type %T", v)
				}
			}
			message := openai.ChatCompletionMessageParamUnion{
				OfAssistant: &assistantMsg,
			}
			oaiMessages = append(oaiMessages, message)
		}
	}

	return oaiMessages, nil
}

func (oaip *OpenAIProvider) convertTools(tools []ToolDefinition) []openai.ChatCompletionToolUnionParam {
	var oaiTools []openai.ChatCompletionToolUnionParam

	for _, tool := range tools {
		oaiTool := openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: openai.FunctionDefinitionParam{
					Name:        tool.Name,
					Description: openai.String(tool.Description),
					Parameters: openai.FunctionParameters{
						"type":       "object",
						"properties": tool.Schema.Properties,
					},
				},
			},
		}
		if len(tool.Schema.Required) > 0 {
			oaiTool.OfFunction.Function.Parameters["required"] = tool.Schema.Required
		}
		oaiTools = append(oaiTools, oaiTool)
	}

	return oaiTools
}
