import os
import json
import random
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner.trainer import train, TrainingArgs
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.datasets import load_dataset
import mlx_lm.fuse

def prepare_data(source_file, output_dir):
    with open(source_file, "r") as f:
        lines = f.readlines()
    
    random.seed(42)
    random.shuffle(lines)
    split_idx = int(len(lines) * 0.9)
    train_lines = lines[:split_idx]
    valid_lines = lines[split_idx:]
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
        f.writelines(train_lines)
    with open(os.path.join(output_dir, "valid.jsonl"), "w") as f:
        f.writelines(valid_lines)
    
    # We also write eval_data.jsonl for run_evals.py
    with open("../data/eval_data.jsonl", "w") as f:
        f.writelines(valid_lines)
        
    print(f"Data split: {len(train_lines)} train, {len(valid_lines)} valid")

def fuse_adapters(base_model, adapter_path, save_path):
    # Programmatic execution of MLX fusion
    print("Fusing LoRA adapters into base model...")
    args = SimpleNamespace(
        model=base_model,
        adapter_path=adapter_path,
        save_path=save_path,
        de_quantize=True, # MUST be True for llama.cpp GGUF conversion later
        upload_name=None,
        hf_path=None,
    )
    try:
        mlx_lm.fuse.main(args)
        print("Fusion complete. Note: GGUF export requires llama.cpp scripts on the fused output.")
    except Exception as e:
        print(f"Programmatic fusion failed, please run: python -m mlx_lm.fuse --model {base_model} --adapter-path {adapter_path} --save-path {save_path}")
        print(e)

def main():
    base_model = "mlx-community/gemma-4-12b-it-4bit"
    data_dir = "../data/mlx_data"
    adapter_dir = "adapters"
    adapter_file = f"{adapter_dir}/adapters.safetensors"
    save_path = "Gemson-12B-Fused"
    
    # 1. Prepare MLX formatted data directories (using raw ChatML messages for native masking)
    prepare_data("../data/training_data.jsonl", data_dir)
    
    # 2. Load model and tokenizer natively optimized for Apple Silicon
    print(f"Loading {base_model}...")
    model, tokenizer = load(base_model)
    
    # 3. Load Datasets
    args = SimpleNamespace(data=data_dir)
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    
    # 4. Inject LoRA Layers
    print("Injecting LoRA adapters...")
    model.freeze()
    linear_to_lora_layers(model, num_lora_layers=42)
    
    # 5. Training Configuration
    training_args = TrainingArgs(
        batch_size=2,
        iters=100,
        val_batches=25,
        steps_per_report=10,
        steps_per_eval=50,
        steps_per_save=50,
        max_seq_length=2048,
        adapter_file=adapter_file,
        grad_checkpoint=True,
    )
    
    os.makedirs(adapter_dir, exist_ok=True)
    optimizer = optim.Adam(learning_rate=2e-4)
    
    # 6. Run Training
    print("Starting MLX training loop...")
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=training_args,
    )
    print("Training complete!")
    
    # 7. Explicit LoRA Fusion & GGUF Export
    fuse_adapters(base_model, adapter_file, save_path)

if __name__ == "__main__":
    main()
