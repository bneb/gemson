package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"

	"github.com/go-playground/validator/v10"
	"github.com/joho/godotenv"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	_ = godotenv.Load()

	backend := flag.String("backend", "ollama", "Backend to use: 'gemini' or 'ollama'")
	model := flag.String("model", "gemma4:26b", "Ollama model name (if backend=ollama)")
	workers := flag.Int("workers", 2, "Number of concurrent workers")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	config := Config{
		Backend:        *backend,
		OllamaModel:    *model,
		APIKey:         os.Getenv("GEMINI_API_KEY"),
		TargetRecords:  1000,
		Workers:        *workers,
		OutputFile:     "training_data.jsonl",
		MaxConsecutive: 5,
	}

	if config.Backend == "gemini" && config.APIKey == "" {
		slog.Error("GEMINI_API_KEY is not set but backend is gemini")
		os.Exit(1)
	}

	existingLines := CountExistingRecords(config.OutputFile)
	if existingLines >= config.TargetRecords {
		slog.Info("target records already achieved", "existing", existingLines)
		return
	}

	slog.Info("starting generation pipeline", "existing", existingLines, "target", config.TargetRecords)

	jobs := make(chan Job, 100)
	results := make(chan Result, 500)

	var workerWg sync.WaitGroup
	var writerWg sync.WaitGroup
	var successCount atomic.Uint64
	successCount.Store(existingLines)
	var consecutiveErrors atomic.Uint32

	// 1. Thread-safe File Writer
	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		StartWriter(ctx, config.OutputFile, results, &successCount, config.TargetRecords, cancel)
	}()

	// 2. Worker Pool
	validate := validator.New()
	for i := 0; i < config.Workers; i++ {
		workerWg.Add(1)
		go func(workerID int) {
			defer workerWg.Done()
			StartWorker(ctx, workerID, config, jobs, results, validate, &consecutiveErrors, cancel)
		}(i)
	}

	// 3. Dynamic Producer
	go func() {
		jobID := uint64(0)
		for {
			if successCount.Load() >= config.TargetRecords {
				close(jobs)
				return
			}
			select {
			case <-ctx.Done():
				close(jobs)
				return
			case jobs <- Job{ID: jobID}:
				jobID++
			}
		}
	}()

	// 4. Graceful Shutdown Cascade
	go func() {
		workerWg.Wait()
		close(results) // Close results only after all workers are done sending
	}()

	<-ctx.Done()
	slog.Info("shutting down, waiting for I/O to flush...")
	writerWg.Wait()
	slog.Info("shutdown complete. exit 0.")
}
