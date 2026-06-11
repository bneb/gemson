package main

import (
	"testing"
	"time"
)

func TestApplyJitter(t *testing.T) {
	base := 100 * time.Millisecond
	for attempt := 0; attempt < 5; attempt++ {
		delay := applyJitter(base, attempt)
		factor := time.Duration(1 << attempt)
		expectedMin := factor * base
		expectedMax := expectedMin + (expectedMin / 2)
		
		if delay < expectedMin || delay >= expectedMax+time.Millisecond {
			t.Errorf("applyJitter(%v, %d) = %v; want between %v and %v", base, attempt, delay, expectedMin, expectedMax)
		}
	}
}
