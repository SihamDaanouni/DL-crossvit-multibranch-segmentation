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
    
    def get_split_data(self, train_percentage:float=0.8):
        assert abs(train_percentage) <= 1.0 , "The parameter train_percentage should be between 0.0 and 1.0"

        n = self.__len__()
        len_train = int(n * abs(train_percentage))
        train_files, val_files, train_labels, val_labels = [], [], [], []

        for k in range(n):
            if k < len_train:
                train_files.append(self.image_files[k])
                train_labels.append(self.labels[k])
            else:
                val_files.append(self.image_files[k])
                val_labels.append(self.labels[k])

        return train_files, val_files, train_labels, val_labels
    
    def split(self, train_percentage:float=0.8, train_tf=transforms.ToTensor(), val_tf=transforms.ToTensor()):
        train_files, val_files, train_labels, val_labels = self.get_split_data(train_percentage)
        train_dataset = SplitDataset(self.base_path, self.segmented_path, train_files, train_labels, train_tf)
        val_dataset = SplitDataset(self.base_path, self.segmented_path, val_files, val_labels, val_tf)
        return train_dataset, val_dataset

class SplitDataset(Dataset):
    def __init__(self, path_to_base:str, path_to_segmented:str, files:list[str], labels:list[int], transform=transforms.ToTensor()):
        super().__init__()
        self.base_path = path_to_base
        self.segmented_path = path_to_segmented
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        base_image_path = os.path.join(self.base_path, self.image_files[index])
        segmented_image_path = os.path.join(self.segmented_path, self.image_files[index])
        
        if not os.path.exists(base_image_path):
            raise IndexError(f"Base image #{index} not found ({base_image_path})")

        if not os.path.exists(segmented_image_path):
            raise IndexError(f"Segmented image #{index} not found ({segmented_image_path})")
        
        base_image = Image.open(base_image_path).convert("RGB")
        segmented_image = Image.open(segmented_image_path).convert("RGB")
        
        base_tensor = self.transform(base_image)
        segmented_tensor = self.transform(segmented_image)

        return base_tensor, segmented_tensor, self.labels[index]

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