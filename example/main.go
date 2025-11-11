package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"

	"github.com/bitrise-io/bitrise-ai-core/pkg/agent"
	"github.com/bitrise-io/bitrise-ai-core/pkg/llm"
	"github.com/jinzhu/configor"
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("error: %s\n", err)
	}
}

func run() error {
	ctx := context.Background()

	if len(os.Args) < 2 {
		return fmt.Errorf("no directory provided")
	}
	dir := os.Args[1]
	info, err := os.Stat(dir)
	if err != nil {
		return fmt.Errorf("stat dir: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("provided path is not a directory")
	}
	dirEntries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("read dir: %w", err)
	}

	var cfg Config
	if err := configor.Load(&cfg); err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}

	logger := slog.New(
		slog.NewTextHandler(
			log.Writer(),
			&slog.HandlerOptions{Level: slog.LevelInfo},
		),
	)

	model := llm.Model{
		Provider:        llm.ProviderName(cfg.ModelProvider),
		Name:            cfg.Model,
		MaxOutputTokens: cfg.MaxOutputTokens,
	}
	if err := model.SetDefaults(); err != nil {
		return fmt.Errorf("set defaults on model: %w", err)
	}
	logger.Info(fmt.Sprintf("using model %+v", model))

	agentBase := &agent.Base{
		Model:            model,
		MaxToolLogLength: cfg.MaxToolLogLength,
		Logger:           logger,
		CacheBust:        cfg.CacheBust,
		SessionFilePath:  cfg.SessionFilePath,
		MaxTokenUsage:    cfg.MaxTokenUsage,
		Timebox:          cfg.Timebox,
	}

	reviewer := NewFileReviewer(agentBase)
	type workerResult struct {
		file         string
		reviewResult FileReviewerResult
		err          error
	}
	cWorkerResults := make(chan workerResult)
	var numFiles int
	for _, entry := range dirEntries {
		if entry.IsDir() {
			continue
		}
		numFiles++

		go func() {
			filePath := fmt.Sprintf("%s/%s", dir, entry.Name())
			reviewResult, _, err := reviewer.Run(ctx, filePath)
			if err != nil {
				cWorkerResults <- workerResult{
					file: entry.Name(),
					err:  fmt.Errorf("review file %s: %w", filePath, err),
				}
				return
			}
			cWorkerResults <- workerResult{
				file:         entry.Name(),
				reviewResult: reviewResult,
			}
		}()
	}
	var files []string
	var reviewResults []FileReviewerResult
	for i := 0; i < numFiles; i++ {
		result := <-cWorkerResults
		if result.err != nil {
			logger.Error(result.err.Error())
			continue
		}
		files = append(files, result.file)
		reviewResults = append(reviewResults, result.reviewResult)
	}

	summarizer := NewSummarizer(agentBase)
	summary, _, err := summarizer.Run(ctx, files, reviewResults)
	if err != nil {
		return fmt.Errorf("summarize reviews: %w", err)
	}

	fmt.Printf("Summary of reviews:\n\n%s\n\n", summary)
	logger.Info(fmt.Sprintf("total usage: %s", agentBase.LLMUsage()))

	return nil
}
