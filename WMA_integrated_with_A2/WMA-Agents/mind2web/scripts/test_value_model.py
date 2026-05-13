import torch
import torch.nn as nn
import json
from transformers import AutoTokenizer, AutoModel
import os

# ======================
# CONFIG
# ======================

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
VALUE_HEAD_PATH = "../models/value_model/value_head.pt"
DATA_PATH = "../data/value/value_dataset.jsonl"

device = torch.device("cpu")   # 🔥 force CPU for safe testing

# ======================
# MODEL DEFINITION
# ======================

class LlamaValueModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()

        # 🔥 Load backbone fully on CPU (no meta, no auto)
        self.backbone = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map={"": "cpu"}
        )

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):

        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )

        last_hidden = outputs.last_hidden_state  # fp16

        lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0))

        last_token = last_hidden[batch_indices, lengths]
        last_token = last_token.float()

        value = self.value_head(last_token)
        value = torch.sigmoid(value)

        return value.squeeze(-1)


# ======================
# LOAD TOKENIZER
# ======================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ======================
# LOAD MODEL
# ======================

print("Loading model...")
model = LlamaValueModel(BASE_MODEL)

# 🔥 Load ONLY value_head weights
value_head_state = torch.load(VALUE_HEAD_PATH, map_location="cpu")
model.value_head.load_state_dict(value_head_state)

model.eval()

# ======================
# LOAD ONE SAMPLE
# ======================

with open(DATA_PATH, "r") as f:
    sample = json.loads(f.readline())

print("\n===== INPUT =====\n")
print(sample["input"])
print("\nTarget score:", sample["score"])

prompt = (
    "<|begin_of_text|>\n"
    "User:\n"
    f"{sample['input']}\n\n"
    "Assistant:\n"
)

# 🔥 NO max_length padding during testing
encoding = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    padding=True
)

input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

print("\nRunning forward pass...")

with torch.no_grad():
    prediction = model(input_ids, attention_mask)

print("Done.\n")

print("Predicted score:", prediction.item())
