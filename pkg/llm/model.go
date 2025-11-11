package llm

import (
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	anthropic_option "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/openai/openai-go/v2"
	"google.golang.org/genai"
)

type ProviderName string

const (
	ProviderAnthropic ProviderName = "anthropic"
	ProviderOpenAI    ProviderName = "openai"
	ProviderGemini    ProviderName = "gemini"
	ProviderBedrock   ProviderName = "bedrock"
)

type Model struct {
	Provider        ProviderName
	Name            string
	MaxOutputTokens int
}

var defaultModels = map[ProviderName]Model{
	ProviderAnthropic: {
		Provider:        ProviderAnthropic,
		Name:            "claude-haiku-4-5-20251001",
		MaxOutputTokens: 15000,
	},
	ProviderBedrock: {
		Provider:        ProviderBedrock,
		Name:            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
		MaxOutputTokens: 15000,
	},
	ProviderOpenAI: {
		Provider: ProviderOpenAI,
		Name:     "gpt-5",
	},
	ProviderGemini: {
		Provider: ProviderGemini,
		Name:     "gemini-2.5-pro",
	},
}

func (m *Model) NewProvider(ctx context.Context) (Provider, error) {
	switch m.Provider {
	case ProviderAnthropic:
		return &AnthropicProvider{
			Client:          anthropic.NewClient(),
			Model:           m.Name,
			MaxOutputTokens: m.MaxOutputTokens,
		}, nil
	case ProviderBedrock:
		return &BedrockProvider{
			AnthropicProvider: &AnthropicProvider{
				Client:          anthropic.NewClient(bedrock.WithLoadDefaultConfig(ctx), anthropic_option.WithAPIKey(os.Getenv("BEDROCK_API_KEY"))),
				Model:           m.Name,
				MaxOutputTokens: m.MaxOutputTokens,
			},
		}, nil
	case ProviderOpenAI:
		return &OpenAIProvider{
			Client:          openai.NewClient(),
			Model:           m.Name,
			MaxOutputTokens: m.MaxOutputTokens,
		}, nil
	case ProviderGemini:
		client, err := genai.NewClient(ctx, &genai.ClientConfig{})
		if err != nil {
			return nil, fmt.Errorf("new genai client: %w", err)
		}
		return &GeminiProvider{
			Client:          client,
			Model:           m.Name,
			MaxOutputTokens: m.MaxOutputTokens,
		}, nil
	}
	return nil, fmt.Errorf("unknown provider %q", m.Provider)
}

func (m *Model) SetDefaults() error {
	if err := m.setDefaultProvider(); err != nil {
		return fmt.Errorf("set default provider: %w", err)
	}

	if defaultModel := defaultModels[m.Provider]; m.Name == "" {
		m.Name = defaultModel.Name
		m.MaxOutputTokens = defaultModel.MaxOutputTokens
	}
	return nil
}

func (m *Model) setDefaultProvider() error {
	switch m.Provider {
	case ProviderOpenAI, ProviderAnthropic, ProviderGemini, ProviderBedrock:
		// already set
	case "":
		v, err := findProvider()
		if err != nil {
			return fmt.Errorf("find provider: %w", err)
		}
		m.Provider = v
	default:
		return fmt.Errorf("unknown provider %q", m.Provider)
	}
	return nil
}

func findProvider() (ProviderName, error) {
	switch {
	case os.Getenv("ANTHROPIC_API_KEY") != "":
		return ProviderAnthropic, nil
	case os.Getenv("OPENAI_API_KEY") != "":
		return ProviderOpenAI, nil
	case os.Getenv("GEMINI_API_KEY") != "":
		return ProviderGemini, nil
	case os.Getenv("BEDROCK_API_KEY") != "":
		return ProviderBedrock, nil
	}
	return "", fmt.Errorf("no valid provider found")
}
