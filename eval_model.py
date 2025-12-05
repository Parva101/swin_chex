"""
SwinCXR - Evaluation Module
=========================

This module provides evaluation functionality for the SwinCXR model,
including prediction generation, AUC calculation, and performance metrics.

Author: Your Name
Company: Your Company Name
Version: 1.0.0
Last Updated: September 14, 2025
"""

import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import cxr_dataset as CXR  # Custom module for dataset loading
from torchvision import transforms
import sklearn.metrics as sklm

def make_pred_multilabel(model, PATH_TO_IMAGES):
    """
    Generates predictions for the test dataset and calculates AUC scores for each pathology.
    
    This function evaluates the SwinCXR model against the test set, processing images in batches,
    generating prediction probabilities for each of the 14 pathologies, and calculating
    the Area Under the ROC Curve (AUC) for each class.

    Args:
        model (torch.nn.Module): Trained SwinCXR model to use for predictions
        PATH_TO_IMAGES (str): Directory path where NIH ChestX-ray14 images are stored

    Returns:
        tuple: Two pandas DataFrames:
            - pred_df: Contains individual predictions and ground truth for each test image
            - auc_df: Contains AUC scores for each pathology class and the average AUC
    """
    BATCH_SIZE = 32
    model.eval()  # Set model to evaluation mode

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=16)

    pred_rows = []
    true_rows = []

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        with torch.no_grad():  # Inference without gradient calculation
            outputs = model(inputs)
            # Apply sigmoid to outputs to convert logits to probabilities
            probs = torch.sigmoid(outputs).cpu().data.numpy()

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape[0]

        for j in range(batch_size):
            global_index = i * BATCH_SIZE + j
            image_id = dataset.df.index[global_index]
            thisrow = {"Image Index": image_id}
            truerow = {"Image Index": image_id}

            for k, label in enumerate(dataset.PRED_LABEL):
                thisrow["prob_" + label] = probs[j, k]
                truerow[label] = true_labels[j, k]

            pred_rows.append(thisrow)
            true_rows.append(truerow)

        if i % 10 == 0:
            print(f"{i * BATCH_SIZE} images processed.")

    pred_df = pd.DataFrame(pred_rows)
    true_df = pd.DataFrame(true_rows)

    auc_rows = []
    for column in true_df.columns[1:]:  # Skip 'Image Index'
        actual = true_df[column].values.astype(int)
        pred = pred_df["prob_" + column].values
        try:
            auc = sklm.roc_auc_score(actual, pred)
            auc_rows.append({"label": column, "auc": auc})
        except ValueError as e:
            print(f"Can't calculate AUC for {column}: {e}")
    auc_df = pd.DataFrame(auc_rows)
    if not auc_df.empty:
        average_auc = auc_df['auc'].mean()
        auc_df = pd.concat(
            [auc_df, pd.DataFrame({'label': ['average_auc'], 'auc': [average_auc]})],
            ignore_index=True
        )
    else:
        auc_df = pd.DataFrame({'label': ['average_auc'], 'auc': [float('nan')]})
    pred_df.to_csv("results/preds.csv", index=False)
    auc_df.to_csv("results/aucs.csv", index=False)

    return pred_df, auc_df
