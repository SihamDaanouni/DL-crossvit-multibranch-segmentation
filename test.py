import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Tuple, Dict, Optional
from torchvision.transforms import InterpolationMode


from src.dataset import SplitDataset  
from src.models.crossvit_general import CrossViTClassifier, CrossViTBackbone  


# ====================== CONFIGURATION ======================
def load_config(config_path: str) -> Dict:
    """Charger la configuration depuis un fichier YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

# ====================== METRIQUES ======================
def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Calculer Accuracy et F1-score."""
    preds = preds.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

def compute_iou(heatmap: torch.Tensor, seg_mask: torch.Tensor, threshold: float = 0.5):
    """
    Calculer l'IoU entre une heatmap et un masque de segmentation.
    Args:
        heatmap: [B, H, W] (valeurs entre 0 et 1)
        seg_mask: [B, 1, H, W] (masque binaire)
        threshold: Seuil pour binariser la heatmap
    Returns:
        IoU moyen sur le batch
    """
    heatmap = (heatmap > threshold).float()
    seg_mask = seg_mask.squeeze(1).float()

    intersection = (heatmap * seg_mask).sum(dim=(1, 2))
    union = heatmap.sum(dim=(1, 2)) + seg_mask.sum(dim=(1, 2)) - intersection
    iou = intersection / (union + 1e-6)
    return iou.mean()

# ====================== ATTENTION ROLLOUT ======================
def attention_rollout(model: CrossViTBackbone, inputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Calculer la heatmap d'attention via rollout.
    Args:
        model: Modèle CrossViT
        inputs: Liste des entrées [nonseg, seg]
    Returns:
        Heatmap [B, H, W]
    """

    # Forward pour récupérer les poids d'attention
    xs = model.forward_features(inputs)
    attn_weights = []

    # Extraire les poids d'attention des blocs
    for blk in model.blocks:
        if hasattr(blk, "attn"):
            attn_weights.append(blk.attn.attn_weights)  # À adapter selon votre implémentation

    # Rollout (simplifié)
    attn_rollout = torch.eye(attn_weights[0].size(-1)).to(inputs[0].device)
    for attn in attn_weights:
        attn_rollout = attn @ attn_rollout

    # Extraire la heatmap pour le token CLS
    cls_heatmap = attn_rollout[:, 0, 1:]  # Ignorer le token CLS
    cls_heatmap = cls_heatmap.reshape(-1, 14, 14)  # Adapter selon la taille de votre grille
    cls_heatmap = torch.nn.functional.interpolate(
        cls_heatmap.unsqueeze(1),
        size=(224, 224),
        mode="bilinear"
    ).squeeze(1)

    return cls_heatmap

# ====================== ENTRAÎNEMENT ======================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    iou_weight: float = 0.1
) -> Tuple[float, float]:
    """Entraîner le modèle pour une époque."""
    model.train()
    total_loss, total_iou_loss = 0.0, 0.0

    for nonseg, seg, labels in tqdm(loader, desc="Training"):
        nonseg, seg, labels = nonseg.to(device), seg.to(device), labels.to(device)

        # Forward
        outputs = model(nonseg, seg)
        cls_loss = criterion(outputs, labels)

        # Calcul de l'IoU (si segmenté)
        if seg.size(1) == 1:  # Masque binaire
            heatmap = attention_rollout(model.backbone, [nonseg, seg])
            iou_loss = 1 - compute_iou(heatmap, seg)
        else:
            iou_loss = 0.0

        # Loss totale
        loss = cls_loss + iou_weight * iou_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += cls_loss.item()
        total_iou_loss += iou_loss

    return total_loss / len(loader), total_iou_loss / len(loader)

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Évaluer le modèle sur un ensemble de données."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for nonseg, seg, labels in tqdm(loader, desc="Evaluating"):
            nonseg, seg, labels = nonseg.to(device), seg.to(device), labels.to(device)

            outputs = model(nonseg, seg)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.append(outputs)
            all_labels.append(labels)

    # Calcul des métriques
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    pred_classes = all_preds.argmax(dim=1).cpu().numpy()
    true_classes = all_labels.cpu().numpy()

    print("Pred distribution:", np.unique(pred_classes, return_counts=True))
    print("True distribution:", np.unique(true_classes, return_counts=True))

    metrics = compute_metrics(all_preds, all_labels)

    return total_loss / len(loader), metrics

# ====================== MAIN ======================
def main():
    # Arguments en ligne de commande
    parser = argparse.ArgumentParser(description="CrossViT pour la détection d'épines")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de configuration YAML")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "interpret"], required=True, help="Mode d'exécution")
    parser.add_argument("--config_name", type=str, required=True, help="Nom de la configuration (A/B/C1/C2/O2/O3/O5)")
    args = parser.parse_args()

    # Charger la configuration
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ====================== CONFIGURATIONS ======================
    # Définir les configurations selon les objectifs
    configs = {
        # O1: Comparaison des 4 configurations
        "A": {"route": [0, 0], "same_resolution": False,"pw": False, "desc": "Non-segmented only"},
        "B": {"route": [1, 1], "same_resolution": False, "pw": False, "desc": "Segmented only"},
        "C1": {"route": [0, 1], "same_resolution": False, "pw": False,"desc": "Non-seg → Small, Seg → Large"},
        "C2": {"route": [1, 0], "same_resolution": False, "pw": False, "desc": "Seg → Small, Non-seg → Large"},

        # O2: Iso-résolution
        "O2": {"route": [0, 1], "same_resolution": True, "pw": False, "desc": "Iso-resolution (Small=Large)"},

        # O3: Pondération par patch (à activer dans le modèle)
        "O3": {"route": [0, 1], "same_resolution": True, "pw": True, "desc": "Iso-resolution + patch weighting"},

        # O5: Loss IoU (à activer dans l'entraînement)
        "O5": {"route": [0, 1], "same_resolution": True, "iou_weight": 0.1, "desc": "Iso-resolution + loss IoU"}
    }

    # Sélectionner la configuration
    config = configs[args.config_name]
    print(f"\nConfiguration: {args.config_name} - {config['desc']}")

    # ====================== DATASET & DATALOADERS ======================
    # Transformations pour les images non segmentées
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    dataset = SplitDataset(
        path_to_base=cfg["data"]["base_image_dir"],
        path_to_segmented=cfg["data"]["segmented_image_dir"],
        path_to_csv=cfg["data"]["path_to_csv"],
        img_transform=img_transform,
        mask_transform=mask_transform
    )

    train_set, val_set = dataset.split(cfg["split"]["train_test"])
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0
    )

    # ====================== MODÈLE ======================
    model = CrossViTClassifier(
        same_resolution=config["same_resolution"],
        route=config["route"],
        num_classes=2,
        img_size=224,
        patch_weighting=config["pw"]
    ).to(device)

    # Optimiseur et loss
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    # ====================== ENTRAÎNEMENT (MODE TRAIN) ======================
    if args.mode == "train":
        print("\n=== Début de l'entraînement ===")
        best_f1 = 0.0
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(cfg["training"]["epochs"]):
            # Entraînement
            train_loss, iou_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                iou_weight=config.get("iou_weight", 0.0)  # O5
            )

            # Évaluation
            val_loss, metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            # Historique
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(metrics["f1"])

            print(f"Epoch {epoch + 1}/{cfg['training']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {metrics['f1']:.4f} | "
                  f"IoU Loss: {iou_loss:.4f}")

            # Sauvegarder le meilleur modèle
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(model.state_dict(), f"best_model_{args.config_name}.pth")
                print(f"Nouveau meilleur modèle sauvegardé (F1: {best_f1:.4f})")

        # Sauvegarder l'historique
        torch.save(history, f"history_{args.config_name}.pth")

        # Tracer les courbes
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(history["val_f1"], label="Val F1")
        plt.legend()
        plt.title("F1-score")
        plt.savefig(f"curves_{args.config_name}.png")
        plt.close()

    # ====================== ÉVALUATION (MODE EVAL) ======================
    elif args.mode == "eval":
        print("\n=== Évaluation ===")
        model.load_state_dict(torch.load(f"best_model_{args.config_name}.pth"))
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    # ====================== INTERPRÉTABILITÉ (MODE INTERPRET) ======================
    elif args.mode == "interpret":
        print("\n=== Interprétabilité ===")
        model.load_state_dict(torch.load(f"best_model_{args.config_name}.pth"))

        # Extraire un batch
        nonseg, seg, labels = next(iter(val_loader))
        nonseg, seg = nonseg.to(device), seg.to(device)

        # Générer les heatmaps
        heatmaps = attention_rollout(model.backbone, [nonseg, seg])

        # Afficher les résultats
        for i in range(min(3, len(nonseg))):  # Afficher 3 exemples
            plt.figure(figsize=(12, 4))

            # Image non segmentée
            plt.subplot(1, 3, 1)
            plt.imshow(nonseg[i].cpu().permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406]))
            plt.title("Non-segmented")

            # Masque segmenté
            plt.subplot(1, 3, 2)
            plt.imshow(seg[i].cpu().squeeze(), cmap="gray")
            plt.title("Segmented Mask")

            # Heatmap
            plt.subplot(1, 3, 3)
            plt.imshow(heatmaps[i].cpu(), cmap="hot")
            plt.title("Attention Heatmap")
            plt.colorbar()

            plt.savefig(f"interpret_{args.config_name}_{i}.png")
            plt.close()

            # Calculer l'IoU
            iou = compute_iou(heatmaps[i].unsqueeze(0), seg[i].unsqueeze(0))
            print(f"Exemple {i + 1} | IoU: {iou:.4f}")
import pandas as pd
if __name__ == "__main__":
    main()
