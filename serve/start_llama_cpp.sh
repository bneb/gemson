#!/bin/bash
# Boots the local inference server

echo "Starting llama.cpp server for gemson-12b on port 8080..."
# Requires llama.cpp built locally
llama-server -m ../outputs/gemson-12b-lora.gguf --mmproj ../outputs/gemson-12b-lora-mmproj.gguf --port 8080
