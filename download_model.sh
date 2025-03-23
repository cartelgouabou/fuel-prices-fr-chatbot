#!/bin/bash

# Bash script to download TinyLlama model from Hugging Face locally.

# Define variables
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_DIR="./models/tinyllama"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Download the model using Hugging Face CLI
huggingface-cli download "$MODEL_NAME" --local-dir "$LOCAL_DIR"

# Instruction to execute the script (make sure it has execute permission):
# chmod +x download_model.sh
# ./download_model.sh
