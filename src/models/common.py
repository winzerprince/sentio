"""
Shared utilities and helper functions for models.
"""

import torch

def save_model(model, path):
    """Saves a PyTorch model."""
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Loads a PyTorch model."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
