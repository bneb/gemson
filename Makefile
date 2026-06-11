.PHONY: serve-gateway serve-model eval-text eval-vision setup data-pipeline clean

# Install Python dependencies and setup environment
setup:
	@echo "Setting up Python environment..."
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@echo "Setup complete. Please ensure you have the GGUF models available."

# Start the Rust API Gateway (Concurrency Manager)
serve-gateway:
	@echo "Starting Rust Concurrency Gateway on port 8000..."
	cd serve && cargo run --release

# Start the local llama.cpp model server
serve-model:
	@echo "Starting local llama-server on port 8080..."
	cd serve && ./start_llama_cpp.sh

# Run qualitative text evaluation comparing Gemma-4 to Gemson
eval-text:
	@echo "Running Text Evaluation Pipeline..."
	.venv/bin/python evals/qualitative_analysis.py

# Run multimodal vision evaluation comparing Gemma-4 to Gemson
eval-vision:
	@echo "Running Vision Evaluation Pipeline..."
	.venv/bin/python evals/sequential_ab_test.py

# Run the Go synthetic data pipeline
data-pipeline:
	@echo "Starting Synthetic Data Generation Pipeline..."
	cd data_pipeline && go run main.go

# Clean compiled artifacts and scratch files
clean:
	@echo "Cleaning up..."
	cd serve && cargo clean
	rm -rf scratch/*
	find . -type d -name "__pycache__" -exec rm -r {} +
	@echo "Clean complete."
