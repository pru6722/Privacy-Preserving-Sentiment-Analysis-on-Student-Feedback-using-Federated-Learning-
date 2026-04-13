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
# DEVICE
# =========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU 🚀")
else:
    device = torch.device("cpu")

# =========================
# LOAD CLEANED DATA
# =========================
df = pd.read_csv("../data/cleaned_reviews.csv")

# Reduce size for faster training (IMPORTANT)
df = df.sample(20000, random_state=42)  # start small

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])

# =========================
# SPLIT
# =========================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# =========================
# TOKENIZER
# =========================
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# =========================
# DATASET CLASS
# =========================
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)

# =========================
# MODEL
# =========================
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)

model.to(device)

# =========================
# TRAINING CONFIG
# =========================
training_args = TrainingArguments(
    output_dir="../outputs",
    num_train_epochs=2,   # keep small initially
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    report_to="none"
)

# =========================
# METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1
    }

# =========================
# TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# EVALUATE
# =========================
results = trainer.evaluate()
print("\nFinal Results:", results)