
from src.dataset import BaseSegmentedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from timm import create_model
from torch.utils.data import DataLoader, random_split
import torch

def select_inputs(config, base_img, seg_img):
    if config == "A":      # non segmentée uniquement
        return base_img, base_img
    elif config == "B":    # segmentée uniquement
        return seg_img, seg_img
    elif config == "C1":   # non-seg → Small | seg → Large
        return base_img, seg_img
    elif config == "C2":   # seg → Small | non-seg → Large
        return seg_img, base_img
    else:
        raise ValueError("Configuration inconnue")

def train_one_epoch(model, loader, optimizer, criterion, device, config):
    model.train()
    total_loss = 0

    for base_img, seg_img, labels in loader:
        base_img = base_img.to(device)
        seg_img = seg_img.to(device)
        labels = labels.to(device)

        x_small, x_large = select_inputs(config, base_img, seg_img)

        optimizer.zero_grad()
        logits = model.forward_dual(x_small, x_large)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device, config):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for base_img, seg_img, labels in loader:
            base_img = base_img.to(device)
            seg_img = seg_img.to(device)

            x_small, x_large = select_inputs(config, base_img, seg_img)
            logits = model.forward_dual(x_small, x_large)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return acc, f1

def create_train_val_loaders(dataset,batch_size=16,val_ratio=0.2,num_workers=4,seed=42):
    # Taille des splits
    n_total = len(dataset)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    # Split reproductible
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=generator
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

