"""
OPTIONAL UPGRADE: Fine-tuned multilingual BERT intent classifier.

The TF-IDF + Logistic Regression model in train_classifier.py is fast and
requires no downloads, making it a good MVP baseline. This script shows
the upgrade path to a BERT-based classifier for higher accuracy on more
complex or ambiguous phrasing, following the same pattern used in your
action-item-classifier project (Bert model variant).

Requirements (not needed for the base project):
    pip install transformers torch --break-system-packages

Usage:
    python bert_classifier.py

Note: this downloads a pretrained multilingual BERT model from HuggingFace,
so it requires internet access to huggingface.co on the machine you run it.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    AdamW,
)

DATA_PATH = "data/intents.json"
MODEL_NAME = "bert-base-multilingual-cased"  # supports Urdu script + Roman transliteration reasonably well
SAVE_DIR = "models/bert_intent_classifier"


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_len=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.label2id[self.labels[idx]])
        return item


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, labels = [], []
    for intent, phrases in data.items():
        for phrase in phrases:
            texts.append(phrase)
            labels.append(intent)
    return texts, labels


def train():
    texts, labels = load_dataset(DATA_PATH)
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(unique_labels)
    )

    dataset = IntentDataset(texts, labels, tokenizer, label2id)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} — loss: {total_loss / len(loader):.4f}")

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    with open(f"{SAVE_DIR}/label_map.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    print(f"Model saved to {SAVE_DIR}")


if __name__ == "__main__":
    train()
