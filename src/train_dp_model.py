import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/cleaned_reviews.csv")
df = df.sample(5000, random_state=42)

le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])

# =========================
# TF-IDF FEATURES
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# =========================
# DATASET
# =========================
train_data = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# =========================
# SIMPLE MODEL
# =========================
model = nn.Sequential(
    nn.Linear(5000, 128),
    nn.ReLU(),
    nn.Linear(128, 3)
)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================
# DIFFERENTIAL PRIVACY
# =========================
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0
)

# =========================
# TRAIN
# =========================
for epoch in range(3):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = nn.CrossEntropyLoss()(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================
# PRIVACY BUDGET
# =========================
epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"\nPrivacy Budget (ε): {epsilon:.2f}")