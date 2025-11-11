package tool

import (
	"context"
	"encoding/json"
	"fmt"
)

const FinalResultToolName = "FinalResult"
const finalResultDescription = `The final response which ends this conversation`

type finalResultPrimitiveInput[ResultT any] struct {
	Response ResultT `json:"response" jsonschema_description:"The final response to the user"`
}

func (tb *Belt[ResultT]) finalResult(_ context.Context, llmInput json.RawMessage) (string, error) {
	// Primitive types must be wrapped in an object to be valid JSON.
	// We could also wrap complex types to make the code simpler, eliminating
	// all checks doing `...structResultType[ResultT]...`, but that would be an
	// unnecessary extra layer for the LLM.
	if !structResultType[ResultT]() {
		var input finalResultPrimitiveInput[ResultT]
		if err := json.Unmarshal(llmInput, &input); err != nil {
			return "", fmt.Errorf("unmarshal input: %w", err)
		}
		tb.agent.SetFinalResult(input.Response)
	} else {
		var input ResultT
		if err := json.Unmarshal(llmInput, &input); err != nil {
			return "", fmt.Errorf("unmarshal input: %w", err)
		}
		tb.agent.SetFinalResult(input)
	}
	return "Final result processed.", nil
}
