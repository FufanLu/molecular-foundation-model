"""Training utilities for odor classifier (baseline + LoRA)."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def split_data(embeddings_dict, df, test_size=0.2, seed=42):
    compounds = list(embeddings_dict.keys())
    X = np.stack([embeddings_dict[c] for c in compounds])
    Y = np.stack([df.loc[df["compound"] == c, "y"].values[0] for c in compounds])

    indices = np.arange(len(compounds))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)

    return compounds, X, Y, train_idx, test_idx


class BaselineMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_baseline(X_train, Y_train, X_test, Y_test, epochs=100, lr=1e-3, batch_size=64, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        total_loss = 0.0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx].to(device)
            yb = Y_train_t[idx].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, (len(X_train_t) // batch_size))
        scheduler.step(avg_loss)

        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(X_test_t.to(device))).cpu().numpy()
            preds_bin = (preds > 0.5).astype(int)
            f1_macro = f1_score(Y_test, preds_bin, average="macro", zero_division=0)

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  loss {avg_loss:.4f}  f1_macro {f1_macro:.4f}  best {best_f1:.4f}")

        if patience_counter >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_f1


def evaluate_baseline(model, X_test, Y_test, class_names):
    device = next(model.parameters()).device
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_t)).cpu().numpy()

    preds_bin = (preds > 0.5).astype(int)

    print(f"\n{'Class':<12} {'F1':>7}")
    print("-" * 21)
    for i, cls in enumerate(class_names):
        f1 = f1_score(Y_test[:, i], preds_bin[:, i], zero_division=0)
        print(f"{cls:<12} {f1:>7.4f}")

    f1_macro = f1_score(Y_test, preds_bin, average="macro", zero_division=0)
    print("-" * 21)
    print(f"{'macro avg':<12} {f1_macro:>7.4f}")
    return f1_macro
