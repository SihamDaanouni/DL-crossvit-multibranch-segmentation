import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class BaseSegmentedDataset(Dataset):
    def __init__(self, path_to_base:str, path_to_segmented:str, path_to_csv:str, transform=transforms.ToTensor(), is_test:bool = False):
        super().__init__()
        self.base_path = path_to_base
        self.segmented_path = path_to_segmented
        self.transform = transform

        # For testing purposes
        self.is_test = is_test

        # All filenames for base images (segmented shoudl have same names)
        self.image_files, self.labels = self.generate_files_labels(path_to_csv)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        base_image_path = os.path.join(self.base_path, self.image_files[index])
        segmented_image_path = os.path.join(self.segmented_path, self.image_files[index])
        
        if not os.path.exists(base_image_path):
            raise IndexError(f"Base image #{index} not found ({base_image_path})")

        if not os.path.exists(segmented_image_path):
            raise IndexError(f"Segmented image #{index} not found ({segmented_image_path})")
        
        base_image = Image.open(base_image_path).convert("RGB")
        segmented_image = Image.open(segmented_image_path).convert("RGB")

        if index == 0 and self.is_test:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1) 
            plt.imshow(np.array(base_image))
            plt.title("Image de base")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(np.array(segmented_image))
            plt.title("Image segmenté")
            plt.axis('off')

            plt.show()
        
        base_tensor = self.transform(base_image)
        segmented_tensor = self.transform(segmented_image)

        return base_tensor, segmented_tensor, self.labels[index]
    
    # Generates all the filenames and associated labels for __getitem__ usage
    def generate_files_labels(self, csv_path:str, sepv:str=','):
        data = pd.read_csv(csv_path, sep=sepv)
        files = [code + ".jpg" for code in data['code'].to_list()]
        labels = data['epines']

        if self.is_test:
            for i in range(10):
                print(f"File: {files[i]}, Label: {labels[i]}")

        return files, labels

if __name__ == "__main__":
    try:
        BASE_PATH = "datasets\mission_herbonaute_2000"
        SEGMENTED_PATH = "datasets\mission_herbonaute_2000_seg_black"
        CSV_PATH = "datasets\Data_v2.csv"
        dataset = BaseSegmentedDataset(BASE_PATH, SEGMENTED_PATH, CSV_PATH, is_test=True)

        tensor_0 = dataset[0]
        sys.exit()
    except IndexError as e:
        print(f"IndexError: {str(e)}")