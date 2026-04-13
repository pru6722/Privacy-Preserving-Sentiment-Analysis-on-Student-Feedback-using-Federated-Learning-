import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# ✅ DEVICE
# =========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU 🚀")
else:
    device = torch.device("cpu")
    print("Using CPU")

# =========================
# ✅ LOAD DATA
# =========================
df = pd.read_csv("../data/feedback.csv")

# 🔥 USE EXTRA FEATURES (IMPORTANT)
df = df[['feedback_text', 'emotion_tag', 'sarcasm_flag', 'sentiment_label']].dropna()

# 🔥 COMBINE FEATURES INTO TEXT
df['combined_text'] = (
    df['feedback_text']
    + " [EMOTION] " + df['emotion_tag'].astype(str)
    + " [SARCASM] " + df['sarcasm_flag'].astype(str)
)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment_label'])

# =========================
# ✅ SPLIT
# =========================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['combined_text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# =========================
# ✅ TOKENIZER (DistilBERT)
# =========================
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# =========================
# ✅ DATASET
# =========================
class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FeedbackDataset(train_encodings, train_labels)
val_dataset = FeedbackDataset(val_encodings, val_labels)

# =========================
# ✅ MODEL (DistilBERT)
# =========================
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)

model.to(device)

# =========================
# ✅ TRAINING CONFIG
# =========================
os.makedirs("../outputs", exist_ok=True)

training_args = TrainingArguments(
    output_dir="../outputs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,   # slightly higher for DistilBERT
    weight_decay=0.01,
    logging_steps=50,
    report_to="none"
)

# =========================
# ✅ METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# =========================
# ✅ TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# =========================
# 🚀 TRAIN
# =========================
trainer.train()

# =========================
# 📊 EVALUATE
# =========================
results = trainer.evaluate()
print("\nFinal Results:", results)