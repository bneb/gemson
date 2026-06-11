# Gemson Quickstart Guide

Gemson is designed to run locally using `llama.cpp` as the blazing-fast execution engine, wrapped by a custom Rust gateway to manage concurrency and prevent memory exhaustion (OOM errors) under high load.

## 1. Install Dependencies
First, ensure you have Python 3 installed. Run the setup command to initialize a virtual environment and install the required dependencies:
```bash
make setup
```
*(Note: Ensure you have downloaded the `.gguf` model weights and placed them in the appropriate directory before starting the server).*

## 2. Start the Inference Engine & Gateway
To handle inference efficiently, Gemson uses a split architecture. You will need to start both the `llama-server` and the Rust rate-limiting gateway. 

Run these commands in **separate terminal windows**:

**Terminal 1 (The Inference Engine):**
```bash
make serve-model    
```
*This boots the GGUF file into memory and exposes a raw API on port `8080`.*

**Terminal 2 (The Rust Gateway):**
```bash
make serve-gateway  
```
*This starts the Rust concurrency manager on port `8000`. It acts as a smart queue, ensuring parallel requests don't crash your system's unified memory.*

## 3. Run the Demo
We provide a simple, out-of-the-box demo script to showcase the model's structured extraction capabilities. In a new terminal, run:
```bash
python demo.py
```
This will send a messy, conversational text transcript about an iOS app crash to the gateway, and print the perfectly structured JSON output.

## 4. Query the API Directly
You can integrate Gemson directly into your own applications by sending conversational transcripts or base64 images to the Rust gateway. 

Send your payloads to:
`http://localhost:8000/v1/extract`

The gateway will automatically route the request to the inference engine and return the strict JSON payload, cleanly ready for your backend Pydantic models.
