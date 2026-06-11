package main

import (
	"testing"

	"github.com/go-playground/validator/v10"
)

func TestBugReportValidation(t *testing.T) {
	validate := validator.New()

	tests := []struct {
		name    string
		report  BugReport
		wantErr bool
	}{
		{
			name: "Valid Report",
			report: BugReport{
				UserName:          "Alice",
				OSVersion:         "macOS 14.0",
				DeviceModel:       "MacBook Pro",
				IssueType:         "crash",
				ReproductionSteps: []string{"Step 1", "Step 2", "Step 3", "Step 4", "Step 5"},
			},
			wantErr: false,
		},
		{
			name: "Invalid IssueType",
			report: BugReport{
				UserName:          "Bob",
				OSVersion:         "iOS",
				DeviceModel:       "iPhone",
				IssueType:         "AI generated filler", // Invalid
				ReproductionSteps: []string{"Step 1", "Step 2", "Step 3", "Step 4", "Step 5"},
			},
			wantErr: true,
		},
		{
			name: "Empty Steps",
			report: BugReport{
				UserName:          "Charlie",
				OSVersion:         "Windows",
				DeviceModel:       "PC",
				IssueType:         "latency",
				ReproductionSteps: []string{}, // Must have at least 1 step
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validate.Struct(tt.report)
			if (err != nil) != tt.wantErr {
				t.Errorf("validate.Struct() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
