package main

import (
	"bufio"
	"context"
	"log/slog"
	"os"
	"sync/atomic"
)

func CountExistingRecords(filename string) uint64 {
	file, err := os.Open(filename)
	if err != nil {
		return 0
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := uint64(0)
	for scanner.Scan() {
		lines++
	}
	return lines
}

func StartWriter(ctx context.Context, filename string, results <-chan Result, successCount *atomic.Uint64, target uint64, cancel context.CancelFunc) {
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		slog.Error("failed to open output file", "err", err)
		cancel()
		return
	}
	defer file.Close()
	defer file.Sync()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	shuttingDown := false
	for res := range results {
		if _, err := writer.WriteString(res.Data + "\n"); err != nil {
			slog.Error("failed to write result", "err", err)
			cancel()
			return
		}
		
		sc := successCount.Add(1)
		
		// Flush periodically to disk
		if sc%10 == 0 {
			if err := writer.Flush(); err != nil {
				slog.Error("periodic flush failed", "err", err)
				cancel()
				return
			}
		}

		if sc%50 == 0 {
			slog.Info("progress update", "records", sc, "target", target)
		}

		if sc >= target && !shuttingDown {
			shuttingDown = true
			slog.Info("target reached! initiating graceful shutdown...")
			cancel() // Graceful shutdown trigger
		}
	}
}
