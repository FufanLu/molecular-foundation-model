"""Predict odor categories from SMILES using trained classifier."""

import torch
import torch.nn as nn
from .label_encoder import ALL_5_CLASSES
from .train import BaselineMLP


def load_baseline(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_baseline(model, embedding, threshold=0.5):
    device = next(model.parameters()).device
    x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()

    results = {}
    for i, cls in enumerate(ALL_5_CLASSES):
        results[cls] = {
            "prob": float(probs[i]),
            "pred": bool(probs[i] >= threshold),
        }
    return results
