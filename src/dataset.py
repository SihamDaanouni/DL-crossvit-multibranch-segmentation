import os
import sys
import argparse
from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class SplitDataset(Dataset):
    def __init__(self, path_to_base:str, path_to_segmented:str, path_to_csv:str|None = None, extension:str=".jpg", transform=transforms.ToTensor(), files:list[str]|None = None, labels:list[str]|None = None, is_test:bool = False):
        super().__init__()
        self.base_path = path_to_base
        self.segmented_path = path_to_segmented
        self.transform = transform

        # For testing purposes
        self.is_test = is_test

        # All filenames for base images (segmented shoudl have same names)
        self.extension = extension
        if (files is None or labels is None) and not (path_to_csv is None):
            self.image_files, self.labels = self.generate_files_labels(path_to_csv)
        elif (files is not None and labels is not None) and (path_to_csv is None):
            self.image_files, self.labels = files, labels
        else:
            raise ReferenceError("No CSV path, no files/labels. This error should never occur except at the initial dataset creation when the path_to_csv argument is missing!")

        # Files validity test
        if is_test:
            files_valid = True
            missing_files = []
            for index in range(len(self)):
                base_image_path = os.path.join(self.base_path, self.image_files[index])
                segmented_image_path = os.path.join(self.segmented_path, self.image_files[index])

                if not os.path.isfile(base_image_path):
                    files_valid = False
                    missing_files.append(base_image_path)

                if not os.path.isfile(segmented_image_path):
                    files_valid = False
                    missing_files.append(segmented_image_path)
            print(f"Missing files: {len(missing_files)}/{len(self)}\n List: {missing_files}")

    
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
        base_image_files = [file for file in os.listdir(self.base_path) if file.endswith(self.extension)]
        segmented_images_files = [file for file in os.listdir(self.segmented_path) if file.endswith(self.extension)]

        # Tout les fichiers qui ont une image et sa version segmentée
        files = [code + self.extension for code in data['code'].to_list() if (code + self.extension) in base_image_files and (code + self.extension) in segmented_images_files]
        labels = data['epines']

        if self.is_test:
            for i in range(5):
                print(f"File: {files[i]}, Label: {labels[i]}")

        return files, labels
    
    def get_split_data(self, split_percentage:float=0.8):
        assert abs(split_percentage) <= 1.0 , "The parameter split_percentage should be between 0.0 and 1.0"

        n = self.__len__()
        fst_split_len = int(n * abs(split_percentage))
        fst_files, snd_files, fst_labels, snd_labels = [], [], [], []

        for k in range(n):
            if k < fst_split_len:
                fst_files.append(self.image_files[k])
                fst_labels.append(self.labels[k])
            else:
                snd_files.append(self.image_files[k])
                snd_labels.append(self.labels[k])

        return fst_files, snd_files, fst_labels, snd_labels
    
    def split(self, split_percentage:float=0.8, fst_tf=transforms.ToTensor(), snd_tf=transforms.ToTensor()):
        fst_files, snd_files, fst_labels, snd_labels = self.get_split_data(split_percentage)
        fst_dataset = SplitDataset(self.base_path, self.segmented_path, files=fst_files, labels=fst_labels, transform=fst_tf, is_test=self.is_test)
        snd_dataset = SplitDataset(self.base_path, self.segmented_path, files=snd_files, labels=snd_labels, transform=snd_tf, is_test=self.is_test)
        return fst_dataset, snd_dataset

if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser()
        p.add_argument("--config", required=True)
        args = p.parse_args()
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        dataset = SplitDataset(Path(cfg["data"]["base_image_dir"]), Path(cfg["data"]["segmented_image_dir"]), Path(cfg["data"]["path_to_csv"]), is_test=True)
        
        # Split test
        fst, snd = dataset.split(cfg["split"]["train_test"])
        fst_0 = fst[0]
        snd_0 = snd[0]
        sys.exit()
    except IndexError as e:
        print(f"IndexError: {str(e)}")
    except ReferenceError as e:
        print(f"ReferenceError: {str(e)}")