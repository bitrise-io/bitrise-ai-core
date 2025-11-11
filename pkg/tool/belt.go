package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"

	"github.com/bitrise-io/bitrise-ai-core/pkg/llm"
	"github.com/invopop/jsonschema"
)

// Belt is a collection of tools that can be used by the agent.
// Follow best practices for defining tools:
// https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#best-practices-for-tool-definitions
//
// IMPORTANT: optional input fields should be marked with JSON `omitempty` tag
// to indicated that they are not required in the JSON schema (behaviour of the
// github.com/invopop/jsonschema lib).
type Belt[ResultT any] struct {
	agent           agenter[ResultT]
	toolDefinitions map[string]Definition
}

type agenter[ResultT any] interface {
	SetFinalResult(ResultT)
}

type Definition struct {
	llm.ToolDefinition
	UseFunc func(context.Context, json.RawMessage) (string, error)
}

type NewBeltParams[ResultT any] struct {
	Agent agenter[ResultT]
	Tools []Definition
}

func NewBelt[ResultT any](p NewBeltParams[ResultT]) *Belt[ResultT] {
	tb := &Belt[ResultT]{agent: p.Agent}

	finalResultSchema := GenerateSchema[finalResultPrimitiveInput[ResultT]]()
	if structResultType[ResultT]() {
		finalResultSchema = GenerateSchema[ResultT]()
	}
	tb.toolDefinitions = map[string]Definition{
		FinalResultToolName: {
			ToolDefinition: llm.ToolDefinition{
				Name:        FinalResultToolName,
				Description: finalResultDescription,
				Schema:      finalResultSchema,
			},
			UseFunc: tb.finalResult,
		},
	}
	for _, def := range p.Tools {
		tb.toolDefinitions[def.Name] = Definition{
			ToolDefinition: def.ToolDefinition,
			UseFunc:        def.UseFunc,
		}
	}

	return tb
}

func (tb *Belt[ResultT]) UseTool(ctx context.Context, name string, input json.RawMessage) (string, error) {
	toolFunc, ok := tb.toolDefinitions[name]
	if !ok {
		return "", fmt.Errorf("unknown tool: %s", name)
	}
	return toolFunc.UseFunc(ctx, input)
}

func (tb *Belt[ResultT]) LLMDefinitions() []llm.ToolDefinition {
	var keys []string
	for name := range tb.toolDefinitions {
		keys = append(keys, name)
	}
	sort.Strings(keys)

	var params []llm.ToolDefinition
	for _, name := range keys {
		tool := tb.toolDefinitions[name]
		params = append(params, tool.ToolDefinition)
	}
	return params
}

func (tb *Belt[ResultT]) FinalResultDefinition() llm.ToolDefinition {
	return tb.toolDefinitions[FinalResultToolName].ToolDefinition
}

func GenerateSchema[T any]() *jsonschema.Schema {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	return reflector.Reflect(v)
}

func structResultType[ResultT any]() bool {
	var val ResultT
	return reflect.TypeOf(val).Kind() == reflect.Struct
}
