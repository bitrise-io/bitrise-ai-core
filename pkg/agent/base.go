package agent

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/bitrise-io/bitrise-ai-core/pkg/core"
	"github.com/bitrise-io/bitrise-ai-core/pkg/llm"
	"github.com/bitrise-io/bitrise-ai-core/pkg/tool"
)

type Base struct {
	// Mandatory fields:
	Model            llm.Model
	MaxToolLogLength int
	Logger           *slog.Logger
	// Optional fields:
	CacheBust       bool
	SessionFilePath string
	MaxTokenUsage   int
	Timebox         time.Duration
	// Internal fields:
	llmUsage llm.TokenUsage
	mu       sync.Mutex
}

type RunMeta struct {
	AgentID  int
	Usage    llm.TokenUsage
	Messages []llm.Message
}

type RunParams struct {
	Prompt       string            // mandatory
	System       string            // optional override
	Tools        []tool.Definition // optional
	PreviousMeta RunMeta           // optional to continue a conversation
}

// Run runs the base agent with the given parameters.
// Its a method for the Base struct, but Go does not support generic methods.
func Run[ResultT any](ctx context.Context, b *Base, p RunParams) (ResultT, RunMeta, error) {
	provider, err := b.Model.NewProvider(ctx)
	if err != nil {
		return *new(ResultT), RunMeta{}, fmt.Errorf("new provider: %w", err)
	}

	var timeboxedUntil time.Time
	if b.Timebox > 0 {
		timeboxedUntil = time.Now().Add(b.Timebox)
	}
	agentInstance, err := core.NewAgent[ResultT](core.NewAgentParams{
		AgentID:          p.PreviousMeta.AgentID,
		SystemPrompt:     p.System,
		LLM:              provider,
		SessionFilePath:  b.SessionFilePath,
		MaxToolLogLength: b.MaxToolLogLength,
		Tools:            p.Tools,
		Logger:           b.Logger,
		TimeboxedUntil:   timeboxedUntil,
		MaxTokenUsage:    b.MaxTokenUsage - int(b.LLMUsage().Total()),
		CacheBust:        b.CacheBust,
		LLMMessages:      p.PreviousMeta.Messages,
		InitialUsage:     p.PreviousMeta.Usage,
	})
	if err != nil {
		return *new(ResultT), RunMeta{}, fmt.Errorf("new agent: %w", err)
	}
	res, err := agentInstance.Run(ctx, p.Prompt)
	// Result could be nil in case of an error (e.g.: budget exceeded), or if the agent is canceled.
	if res != nil {
		additionalUsage := res.TotalUsage
		additionalUsage.InputTokens -= p.PreviousMeta.Usage.InputTokens
		additionalUsage.OutputTokens -= p.PreviousMeta.Usage.OutputTokens
		additionalUsage.CacheCreationTokens -= p.PreviousMeta.Usage.CacheCreationTokens
		additionalUsage.CacheReadTokens -= p.PreviousMeta.Usage.CacheReadTokens
		b.addUsage(additionalUsage)
	}
	if err != nil {
		return *new(ResultT), RunMeta{}, fmt.Errorf("run agent: %w", err)
	}
	if res == nil {
		return *new(ResultT), RunMeta{}, fmt.Errorf("agent returned nil result")
	}
	return res.Data, RunMeta{
		AgentID:  agentInstance.AgentNum(),
		Usage:    res.TotalUsage,
		Messages: res.Messages,
	}, nil
}

func (b *Base) addUsage(u llm.TokenUsage) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.llmUsage.InputTokens += u.InputTokens
	b.llmUsage.OutputTokens += u.OutputTokens
	b.llmUsage.CacheCreationTokens += u.CacheCreationTokens
	b.llmUsage.CacheReadTokens += u.CacheReadTokens
}

func (b *Base) LLMUsage() llm.TokenUsage {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.llmUsage
}
