import copy
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import DataLoader, TensorDataset

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/cleaned_reviews.csv")
df = df.sample(6000, random_state=42)

le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].values

# =========================
# SPLIT INTO CLIENTS
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Split into 3 clients
client_data = []
num_clients = 3
split_size = len(X_train) // num_clients

for i in range(num_clients):
    start = i * split_size
    end = (i + 1) * split_size

    X_c = X_train[start:end]
    y_c = y_train[start:end]

    dataset = TensorDataset(
        torch.tensor(X_c, dtype=torch.float32),
        torch.tensor(y_c, dtype=torch.long)
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    client_data.append(loader)

# =========================
# MODEL
# =========================
def create_model():
    return nn.Sequential(
        nn.Linear(3000, 128),
        nn.ReLU(),
        nn.Linear(128, 3)
    )

global_model = create_model()

# =========================
# FEDERATED TRAINING
# =========================
def train_local(model, loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1):  # local epochs
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model.state_dict()

# =========================
# FEDAVG
# =========================
def average_weights(weights):
    avg_weights = copy.deepcopy(weights[0])

    for key in avg_weights.keys():
        for i in range(1, len(weights)):
            avg_weights[key] += weights[i][key]
        avg_weights[key] = avg_weights[key] / len(weights)

    return avg_weights
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32))
        preds = torch.argmax(outputs, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')

    return acc, f1

# =========================
# TRAINING LOOP
# =========================
rounds = 3

for r in range(rounds):
    print(f"\n--- Round {r+1} ---")

    local_weights = []

    for i, loader in enumerate(client_data):
        local_model = create_model()
        local_model.load_state_dict(global_model.state_dict())

        weights = train_local(local_model, loader)
        local_weights.append(weights)

        print(f"Client {i+1} trained")

    # Aggregate
    global_weights = average_weights(local_weights)
    global_model.load_state_dict(global_weights)

print("\nFederated Training Complete ✅")
acc, f1 = evaluate(global_model, X_test, y_test)

print("\nFinal Federated Model Results:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
