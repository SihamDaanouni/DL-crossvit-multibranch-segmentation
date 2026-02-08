import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


class BaseSegmentedDataset(Dataset):
    def __init__(
        self,
        path_to_base: str,
        path_to_segmented: str,
        path_to_csv: str,
        transform=transforms.ToTensor(),
        is_test: bool = False,
    ):
        super().__init__()

        self.base_path = path_to_base
        self.segmented_path = path_to_segmented
        self.transform = transform
        self.is_test = is_test

        # 🔹 Charger le CSV
        data = pd.read_csv(path_to_csv)

        self.samples = []

        for _, row in data.iterrows():
            filename = row["code"] + ".jpg"
            label = float(row["epines"])

            base_path = os.path.join(self.base_path, filename)
            seg_path = os.path.join(self.segmented_path, filename)

            if os.path.exists(base_path) and os.path.exists(seg_path):
                self.samples.append((filename, label))

        print(f"✅ Dataset prêt : {len(self.samples)} samples valides")

        if self.is_test:
            print("🔍 Exemples:")
            for i in range(min(10, len(self.samples))):
                print(self.samples[i])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]

        base_image_path = os.path.join(self.base_path, filename)
        segmented_image_path = os.path.join(self.segmented_path, filename)

        base_image = Image.open(base_image_path).convert("RGB")
        segmented_image = Image.open(segmented_image_path).convert("RGB")

        if index == 0 and self.is_test:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(np.array(base_image))
            plt.title("Image de base")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(np.array(segmented_image))
            plt.title("Image segmentée")
            plt.axis("off")

            plt.show()

        base_tensor = self.transform(base_image)
        segmented_tensor = self.transform(segmented_image)

        return base_tensor, segmented_tensor, label
