package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/bitrise-io/bitrise-ai-core/pkg/agent"
)

func NewSummarizer(b *agent.Base) summarizer {
	return summarizer{Base: b}
}

type summarizer struct{ *agent.Base }

func (r summarizer) Run(ctx context.Context, files []string, reviews []FileReviewerResult) (string, agent.RunMeta, error) {
	return agent.Run[string](ctx, r.Base, agent.RunParams{
		System: systemSummarizer,
		Prompt: promptSummarizer(files, reviews),
	})
}

const systemSummarizer = "Your task is to summarize reviews of files in a directory."

func promptSummarizer(files []string, reviews []FileReviewerResult) string {
	var a []string
	for i, file := range files {
		fileReview := fmt.Sprintf("<file>%s</file>\n", file)
		reviewJSON, _ := json.Marshal(reviews[i])
		fileReview += fmt.Sprintf("<review>%s</review>\n", reviewJSON)
		a = append(a, fmt.Sprintf("<review>%s</review>", fileReview))
	}
	return fmt.Sprintf(
		"Summarize the reviews of the following files:\n\n%s",
		strings.Join(a, "\n\n"),
	)
}
