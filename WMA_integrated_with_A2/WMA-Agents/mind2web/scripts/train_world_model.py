import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH = "../data/abstracted/wm_transition_abstraction.jsonl"
OUTPUT_DIR = "../models/world_model"
# TEST_DATA_PATH = "../data/abstracted/test_split.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

# dataset_split = dataset.train_test_split(test_size=0.2, seed=42)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

tokenizer.pad_token = tokenizer.eos_token

def format_example(example):
    text = (
        "<|begin_of_text|>\n"
        "User:\n"
        f"{example['input']}\n\n"
        "Assistant:\n"
        f"{example['output']}"
    )
    return {"text": text}

dataset = dataset.map(format_example)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding=True,
        max_length=512,
    )

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset.column_names
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Smaller batch, larger accumulation = stable memory

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    report_to="none",
    optim="adamw_torch",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("World model fine-tuned and saved.")
