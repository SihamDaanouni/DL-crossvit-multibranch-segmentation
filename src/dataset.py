import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BaseSegmentedDataset(Dataset):
    def __init__(self, path_to_base:str, path_to_segmented:str, transform=transforms.ToTensor()):
        super().__init__()
        self.base_path = path_to_base
        self.segmented_path = path_to_segmented
        self.transform = transform

        # All filenames for base images (segmented shoudl have same names)
        self.image_files = [filename for filename in os.listdir(path_to_base) if filename.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        base_image_path = os.path.join(self.base_path, self.image_files[index])
        segmented_image_path = os.path.join(self.segmented_path, self.image_files[index])
        
        if not os.path.exists(base_image_path):
            raise IndexError(f"Base image not found ({base_image_path})")

        if not os.path.exists(segmented_image_path):
            raise IndexError(f"Segemented image not found ({segmented_image_path})")
        
        base_image = Image.open(base_image_path).convert("RGB")
        segmented_image = Image.open(segmented_image_path).convert("RGB")

        if index == 0:
            plt.figure(figsize=(10, 5))  # Taille de la figure

            # Sous-graphe pour l'image de base
            plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, position 1
            plt.imshow(np.array(base_image))
            plt.title("Image de base")
            plt.axis('off')  # Masquer les axes

            # Sous-graphe pour le masque
            plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, position 2
            plt.imshow(np.array(segmented_image))  # Utiliser une colormap pour le masque
            plt.title("Image segmenté")
            plt.axis('off')

            plt.show()
        
        base_tensor = self.transform(base_image)
        segmented_tensor = self.transform(segmented_image)

        return base_tensor, segmented_tensor

class ThornDataset(Dataset):
    def __init__(self, path_to_csv:str):
        super().__init__()


if __name__ == "__main__":
    try:
        BASE_PATH = "datasets\mission_herbonaute_2000"
        SEGMENTED_PATH = "datasets\mission_herbonaute_2000_seg_black"
        dataset = BaseSegmentedDataset(BASE_PATH, SEGMENTED_PATH)

        tensor_0 = dataset[0]
        pass
    except IndexError as e:
        print(f"IndexError: {str(e)}")