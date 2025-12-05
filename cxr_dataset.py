"""
SwinCXR - Chest X-ray Dataset Module
===================================

This module defines the dataset class for loading and processing chest X-ray images
from the NIH ChestX-ray14 dataset. It handles data loading, preprocessing, and
augmentation for training and evaluation using the official NIH split lists.

Author: Your Name
Company: Your Company Name
Version: 1.0.0
Last Updated: September 14, 2025
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from pathlib import Path
from PIL import Image


def _read_split_file(path: Path):
    """
    Read an official NIH split file and return the file names as a set.
    """
    if not path.exists():
        raise FileNotFoundError(f"NIH split file not found: {path}")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _load_official_splits(splits_dir: Path):
    """
    Load the official train/val and test splits supplied by NIH.
    """
    train_val_file = splits_dir / "train_val_list.txt"
    test_file = splits_dir / "test_list.txt"
    train_val = _read_split_file(train_val_file)
    test = _read_split_file(test_file)
    overlap = train_val & test
    if overlap:
        raise ValueError(f"Official split lists overlap for {len(overlap)} images; please verify split files.")
    return train_val, test


def _split_train_val(df, val_fraction=0.1, seed=42):
    """
    Deterministically split the NIH official train_val set into train/val by patient.
    """
    if val_fraction <= 0:
        return df, df.iloc[0:0]
    if val_fraction >= 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    patients = df["Patient ID"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_fraction))
    val_patients = set(patients[:n_val])
    val_df = df[df["Patient ID"].isin(val_patients)]
    train_df = df[~df["Patient ID"].isin(val_patients)]
    return train_df, val_df


class CXRDataset(Dataset):
    """
    Custom dataset class for NIH ChestX-ray14 dataset handling.
    
    This class provides a PyTorch Dataset implementation specifically designed for
    chest X-ray images with multi-label classification. It handles loading images,
    associated labels, and performing transformations as needed.
    
    The dataset is split into train/val/test folds using the official NIH
    `train_val_list.txt` and `test_list.txt` files. By default the entire NIH
    train_val list is used for training; if a validation fraction is requested,
    a deterministic patient-level split carves validation data from train_val.
    """

    def __init__(
            self,
            path_to_images,
            fold,
            transform=None,
            sample=0,
            finding="any",
            starter_images=False,
            splits_dir="/scratch/psheta12/CXR8",
            val_fraction=0.1,
            split_seed=42):
        """
        Initialize the CXRDataset.
        
        Args:
            path_to_images (str): Directory path where the X-ray images are stored
            fold (str): Dataset split to use: 'train', 'val', or 'test'
            transform (callable, optional): Transformations to apply to images (e.g., resize, normalize)
            sample (int, optional): If > 0, use only a sample of this size from the dataset
            finding (str, optional): Filter dataset to only include specific pathology
            starter_images (bool, optional): Use only starter subset of images if True
            splits_dir (str, optional): Directory containing NIH official split lists (train_val_list.txt and test_list.txt). Default points to /scratch/psheta12/CXR8.
            val_fraction (float, optional): Fraction of the official train_val split to reserve for validation
            split_seed (int, optional): Seed for deterministic train/val patient split and sampling
        """
        self.transform = transform
        self.path_to_images = path_to_images
        splits_dir = Path(splits_dir)
        train_val_set, test_set = _load_official_splits(splits_dir)
        self.df = pd.read_csv("nih_labels.csv")

        if fold == "test":
            self.df = self.df[self.df["Image Index"].isin(test_set)]
        elif fold in ("train", "val"):
            filtered_df = self.df[self.df["Image Index"].isin(train_val_set)]
            train_df, val_df = _split_train_val(filtered_df, val_fraction=val_fraction, seed=split_seed)
            self.df = train_df if fold == "train" else val_df
        else:
            raise ValueError("fold must be one of: 'train', 'val', 'test'")

        if(starter_images):
            starter_images = pd.read_csv("starter_images.csv")
            self.df=pd.merge(left=self.df,right=starter_images, how="inner",on="Image Index")
            
        # can limit to sample, useful for testing
        # if fold == "train" or fold =="val": sample=500
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample, random_state=split_seed)

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print("No positive cases exist, returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")

        self.df = self.df.set_index("Image Index")
        self.PRED_LABEL = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']
        RESULT_PATH = "results/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()
                                   ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        return (image, label,self.df.index[idx])
