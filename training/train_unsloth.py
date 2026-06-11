import os
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig

from datasets import load_dataset

def main():
    max_seq_length = 2048
    model_name = "unsloth/gemma-4-12b-it"
    hf_token = os.environ.get("HF_TOKEN")

    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=hf_token
    )

    print("Injecting LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth"
    )

    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma",
    )

    print("Loading dataset...")
    dataset = load_dataset("json", data_files="/workspace/training_data.jsonl", split="train")

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=2)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            output_dir="/workspace/outputs",
        ),
    )

    # Force Unsloth mapping to 2 processes to avoid OOM
    import psutil; psutil.cpu_count = lambda *args, **kwargs: 2
    try:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
            **{"num_proc": 2}
        )
    except TypeError:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

    print("Starting Training...")
    trainer_stats = trainer.train()

    print("Exporting to GGUF...")
    import json
    os.makedirs("/workspace/gemson_model", exist_ok=True)
    with open("/workspace/gemson_model/preprocessor_config.json", "w") as f:
        json.dump({"image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711]}, f)
    model.save_pretrained_gguf("/workspace/gemson_model", tokenizer, quantization_method="q4_k_m")
    print("Training and Export Complete.")

if __name__ == "__main__":
    main()
