package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync/atomic"

	"github.com/go-playground/validator/v10"
)

func StartWorker(ctx context.Context, id int, cfg Config, jobs <-chan Job, results chan<- Result, val *validator.Validate, errors *atomic.Uint32, cancel context.CancelFunc) {
	for {
		select {
		case <-ctx.Done():
			return
		case job, ok := <-jobs:
			if !ok {
				return
			}

			// Perform Resilient API Call
			report, err := GenerateBugReport(ctx, cfg)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				slog.Warn("api error", "worker", id, "job", job.ID, "err", err)
				ce := errors.Add(1)
				if ce >= cfg.MaxConsecutive {
					slog.Error("circuit breaker tripped!", "consecutive_errors", ce)
					cancel()
					return
				}
				continue
			}

			// Validate Structure
			if err := val.Struct(report); err != nil {
				slog.Warn("validation failed", "worker", id, "job", job.ID, "err", err)
				ce := errors.Add(1)
				if ce >= cfg.MaxConsecutive {
					slog.Error("circuit breaker tripped due to persistent validation failures", "consecutive_errors", ce)
					cancel()
					return
				}
				continue
			}

			// Success
			errors.Store(0)

			// Format ChatML
			// We strip the RawTranscript out of the target JSON so the LLM learns to generate ONLY the extracted fields
			extractedData := map[string]interface{}{
				"user_name":          report.UserName,
				"os_version":         report.OSVersion,
				"device_model":       report.DeviceModel,
				"issue_type":         report.IssueType,
				"reproduction_steps": report.ReproductionSteps,
			}

			example := TrainingExample{
				Messages: []Message{
					{Role: "user", Content: "Extract bug data from this transcript:\n\n" + report.RawTranscript},
					{Role: "assistant", Content: toJSONString(extractedData)},
				},
			}
			
			resBytes, _ := json.Marshal(example)

			select {
			case <-ctx.Done():
				return
			case results <- Result{Data: string(resBytes)}:
			}
		}
	}
}

func toJSONString(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}
