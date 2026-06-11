package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"
)

var httpClient = &http.Client{
	Timeout: 120 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 10 * time.Second,
	},
}

// Ollama Structs
type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Format string `json:"format"`
	Stream bool   `json:"stream"`
}

type OllamaResponse struct {
	Response string `json:"response"`
}

// Gemini Structs
type GeminiRequest struct {
	Contents         []Content        `json:"contents"`
	GenerationConfig GenerationConfig `json:"generationConfig"`
}
type Content struct {
	Parts []Part `json:"parts"`
}
type Part struct {
	Text string `json:"text"`
}
type GenerationConfig struct {
	ResponseMimeType string  `json:"responseMimeType"`
	Temperature      float32 `json:"temperature"`
}
type GeminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

func GenerateBugReport(ctx context.Context, config Config) (*BugReport, error) {
	prompt := "You are a data generator. Generate a realistic, highly conversational, and messy customer support transcript where a user is complaining about a software bug. Then, extract the data. Your response MUST be valid JSON matching this exact structure: {\"raw_transcript\": \"the messy conversation string\", \"user_name\": \"string\", \"os_version\": \"string\", \"device_model\": \"string\", \"issue_type\": \"crash|UI_glitch|latency|feature_request\", \"reproduction_steps\": [\"step 1\", \"step 2\"]}"

	maxRetries := 5
	baseDelay := 1000 * time.Millisecond

	for attempt := 0; attempt < maxRetries; attempt++ {
		var req *http.Request
		var err error

		if config.Backend == "gemini" {
			req, err = buildGeminiRequest(ctx, config.APIKey, prompt)
		} else {
			req, err = buildOllamaRequest(ctx, config.OllamaModel, prompt)
		}

		if err != nil {
			return nil, err
		}

		resp, err := httpClient.Do(req)
		if err != nil {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(applyJitter(baseDelay, attempt)):
			}
			continue
		}

		if config.Backend == "gemini" && (resp.StatusCode == 400 || resp.StatusCode == 401 || resp.StatusCode == 403 || resp.StatusCode == 404) {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			return nil, fmt.Errorf("fatal gemini status code: %d", resp.StatusCode)
		}

		if resp.StatusCode != 200 {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(applyJitter(baseDelay, attempt)):
			}
			continue
		}

		var jsonText string
		if config.Backend == "gemini" {
			var geminiResp GeminiResponse
			if err := json.NewDecoder(resp.Body).Decode(&geminiResp); err != nil {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				return nil, err
			}
			if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				return nil, fmt.Errorf("no choices returned from gemini")
			}
			jsonText = geminiResp.Candidates[0].Content.Parts[0].Text
		} else {
			var ollamaResp OllamaResponse
			if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				return nil, err
			}
			if ollamaResp.Response == "" {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				return nil, fmt.Errorf("empty response from ollama")
			}
			jsonText = ollamaResp.Response
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()

		var report BugReport
		if err := json.Unmarshal([]byte(jsonText), &report); err != nil {
			return nil, err
		}

		return &report, nil
	}

	return nil, fmt.Errorf("max retries exceeded")
}

func buildGeminiRequest(ctx context.Context, apiKey, prompt string) (*http.Request, error) {
	reqBody := GeminiRequest{
		Contents: []Content{{Parts: []Part{{Text: prompt}}}},
		GenerationConfig: GenerationConfig{
			ResponseMimeType: "application/json",
			Temperature:      0.9,
		},
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key=%s", apiKey)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return req, nil
}

func buildOllamaRequest(ctx context.Context, model, prompt string) (*http.Request, error) {
	reqBody := OllamaRequest{
		Model:  model,
		Prompt: prompt,
		Format: "json",
		Stream: false,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	url := "http://localhost:11434/api/generate"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return req, nil
}

func applyJitter(base time.Duration, attempt int) time.Duration {
	factor := 1 << attempt
	delay := time.Duration(factor) * base
	jitter := time.Duration(rand.Int63n(int64(delay / 2)))
	return delay + jitter
}
