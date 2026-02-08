from pathlib import Path
import matplotlib.pyplot as plt

def save_curve(values_train, values_val, title, ylabel, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(values_train, label="train")
    plt.plot(values_val, label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
