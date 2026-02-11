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
