package main

import (
	"context"
	"fmt"

	"github.com/bitrise-io/bitrise-ai-core/pkg/agent"
	"github.com/bitrise-io/bitrise-ai-core/pkg/tool"
)

type FileReviewerResult struct {
	StrongLanguage bool   `json:"strong_language" jsonschema_description:"Indicates if strong language was found in the file."`
	HarmfulContent bool   `json:"harmful_content" jsonschema_description:"Indicates if harmful content was found in the file."`
	WritingStyle   string `json:"writing_style" jsonschema_description:"Description of the writing style used in the file."`
	Tone           string `json:"tone" jsonschema_description:"Description of the tone of the file."`
	Comments       string `json:"comments" jsonschema_description:"Detailed comments about the file content."`
}

func NewFileReviewer(b *agent.Base) fileReviewer {
	return fileReviewer{Base: b}
}

type fileReviewer struct{ *agent.Base }

func (r fileReviewer) Run(ctx context.Context, path string) (FileReviewerResult, agent.RunMeta, error) {
	return agent.Run[FileReviewerResult](ctx, r.Base, agent.RunParams{
		System: systemFileReviewer,
		Prompt: promptFileReviewer(path),
		// In reality, we would read the file content and inject it into the
		// prompt or in case of large files, enable reading parts of it via
		// the read tool.
		// This is simplified for the example.
		Tools: []tool.Definition{readTool},
	})
}

const systemFileReviewer = "Your task is to read and review a file."

func promptFileReviewer(path string) string {
	return fmt.Sprintf(
		"Review %q file and return the results by calling the %q tool.",
		path, tool.FinalResultToolName,
	)
}
