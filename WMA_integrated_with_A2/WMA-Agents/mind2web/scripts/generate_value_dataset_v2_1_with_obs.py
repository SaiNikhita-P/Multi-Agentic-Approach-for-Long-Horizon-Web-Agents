import os
import json
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
WORLD_MODEL_PATH = "../models/world_model"

INPUT_DATASET = "../data/value/value_dataset_v2_1.jsonl"
OUTPUT_DATASET = "../data/value/value_dataset_v2_1_with_pred_obs_1.jsonl"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 200

os.makedirs(os.path.dirname(OUTPUT_DATASET), exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading world model LoRA...")

model = PeftModel.from_pretrained(base_model, WORLD_MODEL_PATH)
model.config.use_cache = True
model.eval()

def parse_input(text):

    obs = text.split("Current Observation:")[1].split("Current Action:")[0].strip()

    action = text.split("Current Action:")[1].split("Predict")[0].strip()

    return obs, action

def build_prompt(intent, obs, action):

    return (
        "<|begin_of_text|>\n"
        "User:\n"
        f"Objective: {intent}\n\n"
        f"Current Observation:\n{obs}\n\n"
        f"Action:\n{action}\n\n"
        "Predict the next observation.\n\n"
        "Assistant:\n"
    )

def predict_batch(prompts):

    inputs = tokenizer(
    prompts,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(next(model.parameters()).device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            # temperature=0.2,
            do_sample=False,
            use_cache=True
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = []

    for text in decoded:

        if "Assistant:" in text:
            pred = text.split("Assistant:")[-1].strip()
        else:
            pred = text.strip()

        predictions.append(pred)

    return predictions

print("Loading dataset...")

with open(INPUT_DATASET) as f:
    dataset = [json.loads(line) for line in f]

print("Running world model inference...")

total = 0

with open(OUTPUT_DATASET, "w") as f_out:

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):

        batch = dataset[i:i+BATCH_SIZE]

        prompts = []
        metadata = []

        for sample in batch:

            intent = sample["intent"]
            text = sample["input"]
            reward = sample["score"]

            obs, action = parse_input(text)

            prompt = build_prompt(intent, obs, action)

            prompts.append(prompt)

            metadata.append({
                "intent": intent,
                "obs": obs,
                "action": action,
                "reward": reward
            })

        preds = predict_batch(prompts)

        for meta, pred_obs in zip(metadata, preds):

            new_entry = {
                "intent": meta["intent"],
                "observation": meta["obs"],
                "action": meta["action"],
                "predicted_observation": pred_obs,
                "reward": meta["reward"]
            }

            f_out.write(json.dumps(new_entry) + "\n")

            total += 1


print("\nDataset generation complete.")
print("Total samples:", total)
print("Saved to:", OUTPUT_DATASET)