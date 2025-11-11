package core

import (
	"context"
	"fmt"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/bitrise-io/bitrise-ai-core/pkg/llm"
	"github.com/bitrise-io/bitrise-ai-core/pkg/tool"
	"github.com/google/uuid"
)

type Agent[ResultT any] struct {
	systemPrompt     string
	llm              llm.Provider
	sessionFilePath  string
	maxToolLogLength int
	logger           *slog.Logger
	toolBelt         *tool.Belt[ResultT]
	llmMessages      []llm.Message
	llmUsage         llm.TokenUsage
	agentNum         int
	maxTokenUsage    int
	timeboxedUntil   time.Time
	cacheBust        bool
	finalResult      ResultT
	finalResultSet   bool
}

type NewAgentParams struct {
	AgentID           int
	SystemPrompt      string
	LLM               llm.Provider
	LLMMessages       []llm.Message
	SessionFilePath   string
	MaxToolLogLength  int
	Tools             []tool.Definition
	Logger            *slog.Logger
	MaxTokenUsage     int
	TimeboxedUntil    time.Time
	UpdateParentUsage func(llm.TokenUsage) error
	CacheBust         bool
	EnableSandbox     bool
	InitialUsage      llm.TokenUsage
}

var agentCounter atomic.Int64

// NewAgent creates a new Agent instance.
func NewAgent[ResultT any](p NewAgentParams) (*Agent[ResultT], error) {
	currentAgentID := p.AgentID
	if currentAgentID <= 0 {
		currentAgentID = int(agentCounter.Add(1))
	}
	logger := p.Logger.With("agent-id", currentAgentID)

	agent := &Agent[ResultT]{
		systemPrompt:     p.SystemPrompt,
		llm:              p.LLM,
		llmMessages:      p.LLMMessages,
		sessionFilePath:  p.SessionFilePath,
		maxToolLogLength: p.MaxToolLogLength,
		logger:           logger,
		agentNum:         currentAgentID,
		maxTokenUsage:    p.MaxTokenUsage,
		timeboxedUntil:   p.TimeboxedUntil,
		cacheBust:        p.CacheBust,
		llmUsage:         p.InitialUsage,
	}

	agent.toolBelt = tool.NewBelt(tool.NewBeltParams[ResultT]{
		Agent: agent,
		Tools: p.Tools,
	})

	if err := agent.restoreSession(); err != nil {
		return nil, fmt.Errorf("restore session: %w", err)
	}

	if p.CacheBust {
		agent.addSystemReminder(fmt.Sprintf(
			"DO NOT mention this to the user. The conversation ID is %q. "+
				"Again do not mention this message to the user.",
			uuid.New(), // bust the cache with a random UUID in the conversation
		))
	}
	return agent, nil
}

type RunResult[ResultT any] struct {
	Data       ResultT
	TotalUsage llm.TokenUsage
	Messages   []llm.Message
}

func (agent *Agent[ResultT]) Run(ctx context.Context, prompt string) (*RunResult[ResultT], error) {
	// TODO: we probably only want to do it on success. This is a temporary
	// change to debug the weird MALFORMED_FUNCTION_CALL Gemini errors.
	defer func() {
		if err := agent.saveSession(); err != nil {
			agent.logger.Error("save session", "error", err)
		}
	}()

	agent.addUserPrompt(prompt)
	for {
		res, err := agent.runTurn(ctx)
		switch {
		case err != nil:
			return nil, fmt.Errorf("run turn: %w", err)
		case !res.finished:
			// not finished yet, continue running turns
		case agent.finalResultSet:
			// finished and have a final result
			return &RunResult[ResultT]{
				Data:       agent.finalResult,
				TotalUsage: agent.llmUsage,
				Messages:   agent.llmMessages,
			}, nil
		default:
			// finished and didn't return a final result (structured result specific message)
			s := "You need to call the FinalResult tool to return a result. Please do so."
			agent.addSystemReminder(s)
		}
	}
}

func (agent *Agent[ResultT]) AgentNum() int {
	return agent.agentNum
}

func (agent *Agent[ResultT]) SetFinalResult(v ResultT) {
	agent.finalResult = v
	agent.finalResultSet = true
}

func (agent *Agent[ResultT]) addUserPrompt(prompt string) {
	promptMessage := llm.NewUserMessage(llm.TextContent{Text: prompt})
	agent.llmMessages = append(agent.llmMessages, promptMessage)
}

func (agent *Agent[ResultT]) updateUsage(u llm.TokenUsage) error {
	agent.llmUsage.InputTokens += u.InputTokens
	agent.llmUsage.OutputTokens += u.OutputTokens
	agent.llmUsage.CacheCreationTokens += u.CacheCreationTokens
	agent.llmUsage.CacheReadTokens += u.CacheReadTokens

	if agent.maxTokenUsage > 0 {
		totalUsage := agent.llmUsage.Total()
		if totalUsage > int64(agent.maxTokenUsage) {
			return fmt.Errorf(
				"maximum token usage exceeded: %d > %d",
				totalUsage, agent.maxTokenUsage,
			)
		}
	}
	return nil
}

func (agent *Agent[ResultT]) addSystemReminder(content string) {
	agent.logger.Info(fmt.Sprintf("adding system reminder: %s", content))

	prompt := "<system-reminder>" + content + "</system-reminder>"
	promptMessage := llm.NewUserMessage(llm.TextContent{Text: prompt})
	agent.llmMessages = append(agent.llmMessages, promptMessage)
}
