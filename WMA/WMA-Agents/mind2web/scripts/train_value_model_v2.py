import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH = "../data/value/value_dataset_v2.jsonl"
SAVE_PATH = "../models/value_model_v2"

BATCH_SIZE = 2
EPOCHS = 3
LR = 5e-6
MAX_LENGTH = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_PATH, exist_ok=True)


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
            f"{ex['input']}\n\n"
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
            "score": torch.tensor(ex["score"], dtype=torch.float32)
        }


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

        last_token = last_hidden[batch_indices, lengths]
        last_token = last_token.float()

        value = self.value_head(last_token)
        return value.squeeze(-1)


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

dataset = ValueDataset(DATA_PATH, tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Loading model...")
model = LlamaValueModel(BASE_MODEL).to(device)

optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["score"].to(device)

        preds = model(input_ids, attention_mask)

        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.6f}")

torch.save(
    model.value_head.state_dict(),
    os.path.join(SAVE_PATH, "value_head_v2.pt")
)

print("\n✅ Value model v2 training complete.")
