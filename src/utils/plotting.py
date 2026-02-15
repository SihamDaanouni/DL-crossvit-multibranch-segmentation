from pathlib import Path
import matplotlib.pyplot as plt

def save_curve(values_train, values_val, title, ylabel, save_path:Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(values_train, label="train")
    plt.plot(values_val, label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_loss(train_loss, val_loss, epochs, save_path:Path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evolution of Luss during training')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, format='png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_metric(metric, epochs:int, metric_label:str, title:str, save_path:Path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), metric, label=f"{metric_label}")

    plt.xlabel('Epochs')
    plt.ylabel(f"{metric_label}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, format='png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_heatmap(heatmap, base_image=None, save_path:Path=Path("heatmaps/heatmap.png")):
    if base_image is None:
        heatmap_np = heatmap.permute(1,2,0).cpu().numpy() # C,H,W -> H,W,C
        plt.imshow(heatmap_np)
        plt.axis('off')
        plt.savefig(save_path, format='png', dpi=200, bbox_inches='tight')
        plt.close()
    else:
        # Créer une figure avec 1 ligne et 2 colonnes de subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Afficher les images dans chaque subplot
        axes[0].imshow(base_image, cmap='viridis')
        axes[0].set_title("Base Image")

        heatmap_np = heatmap.permute(1,2,0).cpu().numpy() # C,H,W -> H,W,C
        axes[1].imshow(heatmap_np, cmap='viridis')
        axes[1].set_title("Heatmap")

        # Ajuster l'espacement entre les subplots
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(save_path, format='png', dpi=200, bbox_inches='tight')
        plt.close()