package main

// Config holds application configuration
type Config struct {
	Backend        string // "gemini" or "ollama"
	OllamaModel    string // e.g., "gemma4:26b"
	APIKey         string
	TargetRecords  uint64
	Workers        int
	OutputFile     string
	MaxConsecutive uint32
}

// BugReport is our validated data schema
type BugReport struct {
	RawTranscript     string   `json:"raw_transcript" validate:"required,min=50"`
	UserName          string   `json:"user_name" validate:"required,min=2"`
	OSVersion         string   `json:"os_version" validate:"required,min=3"`
	DeviceModel       string   `json:"device_model" validate:"required"`
	IssueType         string   `json:"issue_type" validate:"required,oneof=crash UI_glitch latency feature_request"`
	ReproductionSteps []string `json:"reproduction_steps" validate:"required,min=1,dive,min=5"`
}

type Job struct {
	ID uint64
}

type Result struct {
	Data string
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type TrainingExample struct {
	Messages []Message `json:"messages"`
}
