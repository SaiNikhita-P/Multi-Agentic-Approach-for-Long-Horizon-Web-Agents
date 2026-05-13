# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# ADAPTER_PATH = "../models/world_model"

# print("Loading base tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# print("Loading base model...")
# model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# print("Loading LoRA adapter...")
# model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# model.eval()

# prompt = """User:
# Objective: Book a flight
# Current URL: https://example.com
# Previous Action: None
# Observation: Homepage with search bar

# Assistant:
# """

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=200,
#         temperature=0.7
#     )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# Config
# =========================

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "../models/world_model"
DATA_PATH = "../data/abstracted/wm_transition_abstraction.jsonl"

# =========================
# Load Dataset Example
# =========================

print("Loading dataset sample...")

with open(DATA_PATH, "r") as f:
    sample = json.loads(f.readline())  # first example

input_text = sample["input"]
ground_truth = sample["output"]

print("\n===== DATASET INPUT =====\n")
print(input_text)

print("\n===== GROUND TRUTH OUTPUT =====\n")
print(ground_truth)

# =========================
# Load Base Tokenizer
# =========================

print("\nLoading base tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# =========================
# Load Base Model
# =========================

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# =========================
# Load LoRA Adapter
# =========================

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()

# =========================
# Format Prompt EXACTLY like training
# =========================

prompt = (
    "<|begin_of_text|>\n"
    "User:\n"
    f"{input_text}\n\n"
    "Assistant:\n"
)

print("\n===== MODEL INPUT PROMPT =====\n")
print(prompt)

# =========================
# Tokenize
# =========================

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# =========================
# Generate (Greedy Decoding)
# =========================

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,                      # greedy decoding
        eos_token_id=tokenizer.eos_token_id   # stop properly
    )

# =========================
# Decode ONLY generated part
# =========================

generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n===== MODEL OUTPUT =====\n")
print(result)
