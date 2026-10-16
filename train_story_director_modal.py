"""
Serverless Modal Fine-Tuning Pipeline for Dedicated Animation Story Director.
Trains a QLoRA adapter on an NVIDIA A100 GPU using Unsloth + HuggingFace TRL.

Usage:
  1. Export training data: python scripts/export_story_director_dataset.py
  2. Launch Modal training: modal run train_story_director_modal.py
"""

import os
import pathlib
import modal

APP_NAME = "story-director-trainer"
app = modal.App(APP_NAME)

# High-performance fine-tuning container image with Unsloth & PyTorch
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.42.0",
        "datasets>=2.19.0",
        "peft>=0.11.0",
        "trl>=0.8.6",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.30.0",
        "scipy",
    )
)

# Persistent volume to store trained LoRA checkpoints and exported GGUF models
model_volume = modal.Volume.from_name("story-director-models", create_if_missing=True)


@app.function(
    image=training_image,
    gpu="A100",
    timeout=3600,
    volumes={"/models": model_volume},
)
def train_story_director(
    dataset_records: list,
    base_model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length: int = 4096,
    learning_rate: float = 2e-4,
    epochs: int = 3,
):
    import json
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    print(f"[StoryDirector Trainer] Loaded {len(dataset_records)} training records.")
    print(f"[StoryDirector Trainer] Base Model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format texts for SFT
    formatted_texts = []
    for item in dataset_records:
        convs = item.get("conversations", [])
        if convs:
            full_prompt = ""
            for msg in convs:
                role = "User" if msg["from"] == "human" else "Assistant"
                full_prompt += f"### {role}:\n{msg['value']}\n\n"
            formatted_texts.append(full_prompt)

    train_dataset = Dataset.from_dict({"text": formatted_texts})
    print(f"[StoryDirector Trainer] Formatted dataset with {len(train_dataset)} examples.")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    output_dir = "/models/story_director_lora"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("[StoryDirector Trainer] Commencing QLoRA training...")
    trainer.train()

    print(f"[StoryDirector Trainer] Saving fine-tuned LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model_volume.commit()
    print("[StoryDirector Trainer] Training complete and weights persisted to Modal Volume!")
    return output_dir


@app.local_entrypoint()
def main(dataset_path: str = "dataset/story_director_sft.jsonl"):
    import json
    p = pathlib.Path(dataset_path)
    if not p.exists():
        print(f"Dataset not found at {dataset_path}! Running export script first...")
        os.system(f"python scripts/export_story_director_dataset.py --output-dir {p.parent}")

    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Launching Modal training job on A100 GPU with {len(records)} training samples...")
    out_dir = train_story_director.remote(dataset_records=records)
    print(f"Modal job succeeded! LoRA adapter saved to volume: {out_dir}")
