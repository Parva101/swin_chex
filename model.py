"""
SwinCXR - Model Architecture and Training Module
==============================================

This module defines the model architecture for the SwinCXR classifier,
including the transformer-based backbone, classification head, and
training functions.

Author: Your Name
Company: Your Company Name
Version: 1.0.0
Last Updated: September 14, 2025
"""

from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree
import copy
# data science imports
import pandas as pd
import numpy as np
import csv

import cxr_dataset as CXR
import eval_model as E
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import timm

MODEL_NAME = 'swin_large_patch4_window7_224'
NUM_LABELS = 14

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


def checkpoint(model, best_loss, epoch, LR):
    """
    Saves checkpoint of SwinCXR model during training.
    
    This function creates a checkpoint that stores the model weights, current best loss,
    epoch number, and learning rate for later resumption of training or inference.
    Checkpoint is saved to the results directory.

    Args:
        model (torch.nn.Module): SwinCXR model to be saved
        best_loss (float): Best validation loss achieved so far in training
        epoch (int): Current epoch number of training
        LR (float): Current learning rate used in training
        
    Returns:
        None: The function saves the checkpoint to disk but does not return any value
    """

    print('saving')
    state = {
        'model_state_dict': model.state_dict(),
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR,
        'model_name': MODEL_NAME
    }

    torch.save(state, 'results/checkpoint')


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, patience=3):
    """
    Fine-tunes a model on NIH CXR data with optional validation/early stopping.

    Args:
        model: The model to be fine-tuned.
        criterion: The loss criterion to be used.
        optimizer: The optimizer for training.
        scheduler: The learning rate scheduler.
        num_epochs: The number of epochs to train for.
        dataloaders: Dataloaders for the training and validation datasets.
        dataset_sizes: Sizes of the training and validation datasets.
        patience: Number of epochs to wait after last time validation loss improved before stopping the training.
                  Ignored when no validation split is provided.

    Returns:
        model: The fine-tuned model.
        best_epoch: The epoch number with the best validation loss.
    """
    since = time.time()

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_since_improvement = 0
    best_epoch = 0
    has_val = 'val' in dataloaders and dataset_sizes.get('val', 0) > 0
    phases = ['train'] + (['val'] if has_val else [])
    early_stopping_enabled = has_val and patience is not None

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        epoch_metrics = {}

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda').float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / max(1, dataset_sizes[phase])
            epoch_metrics[phase] = epoch_loss
            print(f'{phase} Loss: {epoch_loss:.4f}')

        monitor_loss = epoch_metrics['val'] if has_val else epoch_metrics['train']
        scheduler.step(monitor_loss)

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            checkpoint(model, best_loss, epoch, optimizer.param_groups[0]['lr'])  # Save the best model
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print()

        if early_stopping_enabled and epochs_since_improvement == patience:
            print(f'Early stopping triggered after {epoch} epochs.')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f}')

    model.load_state_dict(best_model_wts)
    return model, best_epoch


def train_transformer(PATH_TO_IMAGES, LR, WEIGHT_DECAY, NUM_EPOCHS, BATCH_SIZE, val_fraction=0.1, split_seed=42, splits_dir="/scratch/psheta12/CXR8"):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD
        val_fraction: fraction of the official train_val split to reserve for validation (0.0 uses full official train_val for training)
        split_seed: seed for deterministic patient-level split when val_fraction > 0
        splits_dir: directory containing NIH official split files

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Validation transforms mirror training to align with official split usage
        'val': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    # create train/val dataloaders using official NIH split lists
    transformed_datasets = {
        'train': CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold='train',
            transform=data_transforms['train'],
            splits_dir=splits_dir,
            val_fraction=val_fraction,
            split_seed=split_seed),
    }
    has_val = val_fraction > 0
    if has_val:
        transformed_datasets['val'] = CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold='val',
            transform=data_transforms['val'],
            splits_dir=splits_dir,
            val_fraction=val_fraction,
            split_seed=split_seed)

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            transformed_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=16),
    }
    if has_val:
        dataloaders['val'] = torch.utils.data.DataLoader(
            transformed_datasets['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=16)


    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    # Initialize Swin Transformer
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_LABELS)  # 14 head

    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    criterion = nn.BCEWithLogitsLoss()  # Update loss function for multilabel
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Introducing a ReduceLROnPlateau learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in dataloaders.keys()}

    # train model
    patience = 3 if has_val else None
    model, best_epoch = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS, dataloaders, dataset_sizes, patience=patience)

    return model, best_epoch
