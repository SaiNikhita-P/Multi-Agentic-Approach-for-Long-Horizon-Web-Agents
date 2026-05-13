import torch
import torch.nn as nn
import json
from transformers import AutoTokenizer, AutoModel

# =========================
# Config
# =========================

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

MODEL_PATH = "../models/value_model_v3/value_head_v3_1.pt"
DATA_PATH = "../data/value/value_dataset_clean.jsonl"

NUM_SAMPLES = 20
MAX_LENGTH = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Value Model Definition
# =========================

class LlamaValueModel(nn.Module):

    def __init__(self, base_model_name):

        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):

        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        last_hidden = outputs.last_hidden_state

        lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)

        last_token = last_hidden[batch_indices, lengths].float()

        value = self.value_head(last_token)

        return value.squeeze(-1)


# =========================
# Load tokenizer
# =========================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token


# =========================
# Load model
# =========================

print("Loading value model...")

model = LlamaValueModel(BASE_MODEL).to(device)

model.value_head.load_state_dict(torch.load(MODEL_PATH))

model.eval()


# =========================
# Load dataset
# =========================

samples = []

with open(DATA_PATH) as f:
    for line in f:
        samples.append(json.loads(line))


# =========================
# Test samples
# =========================

print("\nTesting value predictions\n")

for i in range(NUM_SAMPLES):

    ex = samples[i]

    prompt = (
        "<|begin_of_text|>\n"
        "User:\n"
        f"Objective: {ex['intent']}\n\n"
        f"Current Observation:\n{ex['observation']}\n\n"
        f"Action Taken:\n{ex['action']}\n\n"
        f"Predicted Next Observation:\n{ex['predicted_observation']}\n\n"
        "Predict the reward.\n\n"
        "Assistant:\n"
    )

    encoding = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        pred = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        ).item()

    true = ex["reward"]

    print("------------")
    print(f"True Reward: {true:.3f}")
    print(f"Predicted  : {pred:.3f}")