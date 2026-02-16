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
from src.models.crossvit_general import CrossViTClassifier, RolloutCrossVitClassifier, RolloutCrossVitBackbone
from src.utils.metrics import compute_metrics, compute_rollout, compute_heatmap, compute_iou, compute_iou_loss


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
from src.models.crossvit_general import CrossViTClassifier, RolloutCrossVitClassifier, RolloutCrossVitBackbone
from src.utils.metrics import compute_metrics, compute_rollout, compute_heatmap, compute_iou, compute_iou_loss


# ====================== CONFIGURATION ======================
def load_config(config_path: str) -> Dict:
    """Charger la configuration depuis un fichier YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


# ====================== ENTRAÎNEMENT ======================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    iou_weight: float = 0.3,
    use_iou_loss: bool = False
) -> Tuple[float, float]:
    """Entraîner le modèle pour une époque."""
    model.train()
    total_loss, total_iou_loss = 0.0, 0.0
    
    for nonseg, seg, labels in tqdm(loader, desc="Training"):
        nonseg, seg, labels = nonseg.to(device), seg.to(device), labels.to(device)

        # Forward - handle both model types
        if isinstance(model, RolloutCrossVitClassifier):
            outputs, all_attn = model(nonseg, seg)
        else:
            outputs = model(nonseg, seg)
            all_attn = None
        
        # Classification loss
        cls_loss = criterion(outputs, labels)

        # IoU loss (only if enabled and model supports it)
        iou_loss_value = 0.0
        if use_iou_loss and all_attn is not None:
            try:
                # Compute rollout
                rollout = compute_rollout(all_attn, model.route[1])
                
                # Ensure rollout is on the correct device
                if not rollout.is_cuda and device.type == 'cuda':
                    rollout = rollout.to(device)
                
                # Compute heatmap
                heatmaps = compute_heatmap(rollout, seg, alpha=0.6)
                
                # Ensure heatmap is on the correct device
                if not heatmaps.is_cuda and device.type == 'cuda':
                    heatmaps = heatmaps.to(device)
                
                # Compute IoU loss
                iou_loss_tensor = compute_iou_loss(heatmaps, seg)
                iou_loss_value = iou_loss_tensor.item()
                
                # Total loss
                loss = cls_loss + iou_weight * iou_loss_tensor
            except Exception as e:
                print(f"Warning: IoU loss computation failed: {e}")
                import traceback
                traceback.print_exc()
                loss = cls_loss
        else:
            loss = cls_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += cls_loss.item()
        total_iou_loss += iou_loss_value

    return total_loss / len(loader), total_iou_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    iou_weight: float = 0.3,
    use_iou_loss: bool = False
) -> Tuple[float, float, Dict[str, float]]:
    """Évaluer le modèle sur un ensemble de données."""
    model.eval()
    total_loss, total_iou_loss = 0.0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for nonseg, seg, labels in tqdm(loader, desc="Evaluating"):
            nonseg, seg, labels = nonseg.to(device), seg.to(device), labels.to(device)

            # Forward - handle both model types
            if isinstance(model, RolloutCrossVitClassifier):
                outputs, all_attn = model(nonseg, seg)
            else:
                outputs = model(nonseg, seg)
                all_attn = None
            
            # Classification loss
            cls_loss = criterion(outputs, labels)

            # IoU loss (only if enabled and model supports it)
            iou_loss_value = 0.0
            if use_iou_loss and all_attn is not None:
                try:
                    rollout = compute_rollout(all_attn, model.route[1])
                    
                    # Ensure rollout is on the correct device
                    if not rollout.is_cuda and device.type == 'cuda':
                        rollout = rollout.to(device)
                    
                    heatmaps = compute_heatmap(rollout, seg, alpha=0.6)
                    
                    # Ensure heatmap is on the correct device
                    if not heatmaps.is_cuda and device.type == 'cuda':
                        heatmaps = heatmaps.to(device)
                    
                    iou_loss_tensor = compute_iou_loss(heatmaps, seg)
                    iou_loss_value = iou_loss_tensor.item()
                    
                    # Total loss
                    loss = cls_loss + iou_weight * iou_loss_tensor
                except Exception as e:
                    print(f"Warning: IoU loss computation failed: {e}")
                    loss = cls_loss
            else:
                loss = cls_loss

            total_loss += cls_loss.item()
            total_iou_loss += iou_loss_value

            # Prédictions
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcul des métriques
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }

    return total_loss / len(loader), total_iou_loss / len(loader), metrics


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

    cfg_name = cfg_name
    print(f"\nConfiguration: {cfg_name} - {cfg[cfg_name]['desc']}")

    # Dataset & DataLoaders
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

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda')  # Speed up data transfer to GPU
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )

    # Modèle
    if cfg_name == "O5":
        model = RolloutCrossVitClassifier(
            same_resolution=cfg[cfg_name]["same_resolution"],
            route=cfg[cfg_name]["route"],
            num_classes=2,
            img_size=224,
            patch_weighting=cfg[cfg_name]["pw"]
        ).to(device)
    else:
        model = CrossViTClassifier(
            same_resolution=cfg[cfg_name]["same_resolution"],
            route=cfg[cfg_name]["route"],
            num_classes=2,
            img_size=224,
            patch_weighting=cfg[cfg_name]["pw"]
        ).to(device)

    # Verify model is on correct device
    print(f"Model device: {next(model.parameters()).device}")

    # Optimiseur et loss
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    # Entraînement
    if args.mode == "train":
        print("\n=== Début de l'entraînement ===")
        best_f1 = 0.0
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(1, cfg["model"]["epochs"] + 1):
            print(f"\nEpoch {epoch}/{cfg['model']['epochs']}")
            
            train_loss, train_iou_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                iou_weight=cfg[cfg_name].get("iou_weight", 0.0),
                use_iou_loss=(cfg_name == "O5")
            )
            
            val_loss, val_iou_loss, metrics = evaluate(
                model, val_loader, criterion, device,
                iou_weight=cfg[cfg_name].get("iou_weight", 0.0),
                use_iou_loss=(cfg_name == "O5")
            )
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if cfg_name == "O5":
                print(f"Train IoU Loss: {train_iou_loss:.4f} | Val IoU Loss: {val_iou_loss:.4f}")
            print(f"Val Accuracy: {metrics['accuracy']:.4f} | Val F1: {metrics['f1']:.4f}")
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(metrics["f1"])
            
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(model.state_dict(), f"best_model_{cfg_name}.pth")
                print(f"✓ Best model saved (F1: {best_f1:.4f})")

        torch.save(history, f"history_{cfg_name}.pth")

        # Plots
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
        plt.savefig(f"curves_{cfg_name}.png")
        plt.close()

    elif args.mode == "eval":
        print("\n=== Évaluation ===")
        model.load_state_dict(torch.load(f"best_model_{cfg_name}.pth", map_location=device))
        val_loss, iou_loss, metrics = evaluate(
            model, val_loader, criterion, device,
            iou_weight=cfg[cfg_name].get("iou_weight", 0.0),
            use_iou_loss=(cfg_name == "O5")
        )
        print(f"Val Loss: {val_loss:.4f} | IoU Loss: {iou_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    elif args.mode == "interpret":
        print("\n=== Interprétabilité ===")
        model.load_state_dict(torch.load(f"best_model_{cfg_name}.pth", map_location=device))

        nonseg, seg, _ = next(iter(val_loader))
        nonseg, seg = nonseg.to(device), seg.to(device)

        if isinstance(model, RolloutCrossVitClassifier):
            _, all_attn = model(nonseg, seg)
            rollout = compute_rollout(all_attn, model.route[1])
            
            # Ensure on correct device
            if not rollout.is_cuda and device.type == 'cuda':
                rollout = rollout.to(device)
            
            heatmaps = compute_heatmap(rollout, seg, alpha=0.6)

            for i in range(min(3, len(nonseg))):
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                img_nonseg = nonseg[i].cpu().permute(1, 2, 0).numpy()
                plt.imshow(np.clip(img_nonseg, 0, 1))
                plt.title("Non-segmented")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                img_seg = seg[i].cpu().permute(1, 2, 0).numpy()
                plt.imshow(np.clip(img_seg, 0, 1))
                plt.title("Segmented")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                heatmap_np = heatmaps[i].cpu().numpy()
                if heatmap_np.ndim == 3:
                    heatmap_np = heatmap_np.mean(axis=0)
                plt.imshow(heatmap_np, cmap="hot")
                plt.title("Attention Heatmap")
                plt.colorbar()
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(f"interpret_{cfg_name}_{i}.png")
                plt.close()

                iou = compute_iou(
                    heatmaps[i:i+1] if heatmaps[i].dim() == 2 else heatmaps[i].unsqueeze(0),
                    seg[i].unsqueeze(0)
                )
                print(f"Exemple {i + 1} | IoU: {iou.mean():.4f}")

if __name__ == "__main__":
    main()
