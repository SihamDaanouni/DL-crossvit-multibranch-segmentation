import argparse
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.utils.metrics import compute_classif_metrics
from src.utils.plotting import save_curve
from src.data.herb_dataset_o1a import HerbO1ADataset
from src.models.crossvit_o1a import CrossViTO1A


def train_one_epoch(model, loader, optimizer, criterion, device, amp: bool):
    model.train()
    losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))

    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, amp: bool):
    model.eval()
    losses = []
    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_classif_metrics(y_true, y_pred)
    return (sum(losses) / max(len(losses), 1)), metrics


def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed_everything(int(cfg["seed"]))

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg["train"].get("amp", True))

    img_size = int(cfg["train"]["img_size"])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    ds_train = HerbO1ADataset(
        split_csv=cfg["data"]["train_csv"],
        images_dir=cfg["data"]["images_dir"],
        image_exts=cfg["data"]["image_exts"],
        transform=train_tf,
    )
    ds_val = HerbO1ADataset(
        split_csv=cfg["data"]["val_csv"],
        images_dir=cfg["data"]["images_dir"],
        image_exts=cfg["data"]["image_exts"],
        transform=val_tf,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
    )

    model = CrossViTO1A(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    save_best_on = cfg["output"].get("save_best_on", "f1")
    best_score = -1.0
    best_path = out_dir / "best.pth"

    hist = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, optimizer, criterion, device, amp)
        va_loss, metrics = eval_one_epoch(model, dl_val, criterion, device, amp)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_accuracy"].append(metrics["accuracy"])
        hist["val_f1"].append(metrics["f1"])

        score = metrics["f1"] if save_best_on == "f1" else metrics["accuracy"]
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "config": cfg, "best_score": best_score}, best_path)

        print(
            f"[EPOCH {epoch}/{epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} best={best_score:.4f}"
        )

    (out_dir / "metrics.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    save_curve(hist["train_loss"], hist["val_loss"], "Loss", "loss", out_dir / "loss_curve.png")
    save_curve(hist["val_f1"], hist["val_f1"], "Val F1", "f1", out_dir / "val_f1_curve.png")
    save_curve(hist["val_accuracy"], hist["val_accuracy"], "Val Accuracy", "accuracy", out_dir / "val_accuracy_curve.png")
    print(f"[DONE] Saved -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
