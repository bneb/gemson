# Gemson vs. Gemma-4-12B: A/B Evaluation Report

This document provides a qualitative and quantitative analysis comparing the baseline **Gemma-4-12B-Unified** model against our fine-tuned **Gemson-12B-v1** model.

Our evaluation specifically tests the models' ability to ingest messy, conversational bug reports (via text) and raw crash screenshots (via vision) and output structured, syntax-valid JSON conforming to a rigid Pydantic schema.

---

## 1. Quantitative Overview (Text Extraction)

We evaluated both models against a hold-out dataset of 50 synthetic customer support transcripts (`data/eval_data.jsonl`). The models were tasked with extracting `user_name`, `device_model`, `os_version`, `issue_type` (Enum), and an array of `reproduction_steps`.

| Model | Success Rate (Strict JSON Parse) | Average Qualitative Score | Primary Failure Mode |
| :--- | :--- | :--- | :--- |
| **Gemma-4-12B (Baseline)** | 0.00% (0 / 50) | 0.00 / 100 | Syntax Errors / Bracket Hallucination |
| **Gemson-12B (Fine-Tuned)** | 90.00% (45 / 50) | 94.50 / 100 | Minor enum misclassifications |

### The "Forced JSON" Problem (Without Heavy Prompting)
During our initial testing, we forced the baseline model to output JSON using standard inference API flags (e.g., `format: "json"`) without a massive system prompt holding its hand. Because the baseline model naturally attempts to prepend conversational filler to its responses (e.g., *"Sure, here is the extracted data:"*), the strict JSON enforcement interferes with standard token generation. This resulted in severe bracket-matching errors, completely breaking the output:

**Example Baseline Output (Forced JSON):**
```json
{"thought|2025-03-18T14:30:01.000Z}```json,  ":") {        "  }
```

### The "Unforced" Syntax Problem
When we removed the `format: "json"` constraint and simply prompted the baseline model to strictly format its output as JSON, the model scored 0/50 in parsing success. 

While the baseline model is capable of accurate extraction, it failed programmatic parsing due to:
1. **Markdown Wrappers:** The model wraps all outputs in conversational markdown (` ```json ... ``` `), which instantly breaks raw `json.loads()` parsers.
2. **Fragile Syntax:** Once the markdown wrappers were stripped via regex, the underlying JSON strings failed to parse due to trailing commas, unescaped quote marks inside strings, or missing closing brackets.

**Gemson-12B eliminates this friction**, natively outputting clean raw JSON syntax starting at the very first byte, ready for immediate backend ingestion.

---

## 2. Qualitative Analysis (Multimodal Vision)

Because the baseline model is natively multimodal, we tested both models by passing them raw pixel images of software bugs (bypassing text transcripts entirely). We evaluated 3 images using the native Ollama `/api/generate` endpoint.

### Image 1: Windows BSOD Crash (`bsod_crash.png`)
*   **Baseline:** The baseline model failed to extract any reproduction steps, outputting an array of `null` values and wrapping its response in conversational backticks.
*   **Gemson:** Performed perfect OCR, extracting the "61%" progress completion and OS architecture directly from the image into a logical reproduction array.
    *   *Gemson Output:* `"reproduction_steps": ["Device encounters a critical system error and displays a BSOD...", "System begins a 61% progress completion..."]`

### Image 2: Android App Crash (`android_crash.png`)
*   **Baseline:** The baseline model suffered from severe OCR hallucinations. It confidently hallucinated the text "'Nearby Settings list dropped'" instead of reading "Unfortunately, Settings has stopped", and categorized it as a "UI_glitch" rather than a "crash".
*   **Gemson:** Not only structured the crash event perfectly, but utilized the *background visual context* (the Settings menu was visible behind the crash dialog) to logically deduce the preceding user actions.
    *   *Gemson Output:* `"reproduction_steps": ["Open the application", "Navigate to the settings menu", "Tap on the 'Notifications' option"]`

### Image 3: 404 Error UI (`404_error.png`)
*   **Baseline:** The baseline failed entirely on this UI screen, returning `null` for every single field in the schema instead of utilizing the defined "Unknown" enums.
*   **Gemson:** Correctly adhered to all `"Unknown"` enum defaults while providing a specific reproduction step describing the precise illustration on the 404 page.

---

## 3. The Prompt Engineering Fallacy

A common assumption is that prompt engineering can fix poor extraction quality. To ensure a fair comparison, we ran an ablation test giving the baseline `Gemma-4-12B` model a heavy system instruction detailing the exact JSON schema and explicitly forbidding markdown. 

**Even with an explicit prompt, the baseline ignored instructions, output conversational markdown, and suffered runaway generation:**
Because explicitly forbidding conversational text contradicts the model's core alignment training, the baseline model frequently entered "runaway generation" states. It would generate hundreds of tokens of conversational filler, summarizing the bug instead of structuring it, causing standard HTTP inference requests to hit a hard `TimeoutError` (120s+ execution time) before generating any JSON syntax. When it did successfully complete a response, it still wrapped the output in markdown:
```markdown
Based on the transcript provided, here is the extracted bug data:

### **Bug Report**
**Issue Summary:** App crashes immediately after tapping "Confirm"...
**User Information:**
*   **Name:** Marcus
...
```

**Forcing JSON Format via API Causes Instability & Timeouts:**
When we subsequently forced the baseline into strict JSON mode using Ollama's `format="json"` API flag, the baseline achieved a mixed, fundamentally unstable state. While it successfully extracted some of the simpler transcripts, it completely broke down on complex ones. Because it is a conversational model, it naturally wants to "explain" its reasoning or add conversational filler. When forcefully restricted to only output valid JSON syntax on tricky transcripts, the model panicked and hallucinated endless JSON strings and fake keys to shove its conversational text into, frequently resulting in massive 450+ token responses that ended in a hard `TimeoutError` (60s+ execution time).

**Conclusion:** You cannot simply prompt-engineer or brute-force reliability into general conversational models for rigid pipeline ingestion. The structural guarantees must be natively fine-tuned into the model weights, which is exactly what Gemson achieves with zero system prompting overhead.

---

## Training Cost & Efficiency

It is worth highlighting the extreme cost efficiency of this approach. Excluding the compute time spent on experimental missteps, the actual end-to-end synthetic data generation and QLoRA fine-tuning process cost **less than $4.00** in API and compute costs. 

For less than the price of a cup of coffee, this pipeline produced a specialized, domain-specific agent that vastly outperforms the latest general-purpose local models (like the 12B baseline) at strict data extraction tasks. It proves that throwing massive parameters at a problem is often less effective—and far more expensive—than building a targeted, fine-tuned middleware layer.

---

## Conclusion
While the **Gemma-4-12B** baseline model possesses strong reasoning capabilities, it is fundamentally incompatible with rigid, automated data pipelines. Its propensity for conversational wrappers, syntax errors, schema breaking (returning strings instead of arrays), and multimodal blindness render it unusable for direct programmatic ingestion.

**Gemson-12B** successfully isolates the extraction capabilities of the base model while enforcing strict structural reliability. It operates as a reliable middleware layer capable of ingesting both messy text and raw images and mapping them to downstream backend systems.
