import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

DATA_PATH = "../data/value/value_dataset_clean.jsonl"
SAVE_PATH = "../models/value_model_v3"
# TEST_SPLIT_PATH = "../data/value/value_test_split.jsonl"

BATCH_SIZE = 2
EPOCHS = 15
LR = 5e-6
MAX_LENGTH = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_PATH, exist_ok=True)


# =========================
# Dataset
# =========================

class ValueDataset(Dataset):

    def __init__(self, path, tokenizer):

        self.samples = []
        self.tokenizer = tokenizer

        with open(path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        ex = self.samples[idx]

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

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "reward": torch.tensor(ex["reward"], dtype=torch.float32)
        }


# =========================
# Value Model
# =========================

class LlamaValueModel(nn.Module):

    def __init__(self, base_model_name):

        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16
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
                attention_mask=attention_mask
            )

        last_hidden = outputs.last_hidden_state

        lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)

        last_token = last_hidden[batch_indices, lengths].float()

        value = self.value_head(last_token)

        return value.squeeze(-1)


# =========================
# Tokenizer
# =========================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token


# =========================
# Dataset
# =========================

dataset = ValueDataset(DATA_PATH, tokenizer)

# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size

# train_dataset, test_dataset = random_split(
#     dataset,
#     [train_size, test_size],
#     generator=torch.Generator().manual_seed(42)
# )

# print(f"Train size: {train_size}")
# print(f"Test size: {test_size}")

# with open(TEST_SPLIT_PATH, "w") as f:
#     for idx in test_dataset.indices:
#         json.dump(dataset.samples[idx], f)
#         f.write("\n")

# print("Test split saved to:", TEST_SPLIT_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =========================
# Model
# =========================

print("Loading model...")

model = LlamaValueModel(BASE_MODEL).to(device)

optimizer = torch.optim.AdamW(
    model.value_head.parameters(),
    lr=LR
)

loss_fn = nn.MSELoss()


# =========================
# Training
# =========================

print("Starting training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in tqdm(loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["reward"].to(device)

        preds = model(input_ids, attention_mask)

        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.value_head.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.6f}")


# =========================
# Save model
# =========================

torch.save(
    model.value_head.state_dict(),
    os.path.join(SAVE_PATH, "value_head_v3_1.pt")
)

print("\n Value model v3 training complete.")
print("Saved to:", os.path.join(SAVE_PATH, "value_head_v3_1.pt"))