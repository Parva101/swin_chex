"""
SwinCXR - Training and Evaluation Script
======================================

This script serves as the main entry point for training and evaluating
the SwinCXR model. It handles command-line arguments, sets hyperparameters,
and orchestrates the model training and evaluation process.

Author: Your Name
Company: Your Company Name
Version: 1.0.0
Last Updated: September 14, 2025
"""

import os
import cxr_dataset as CXR
import eval_model as E
import model as M
import argparse
import torch

os.chdir("./")

# Parse command line arguments
parser = argparse.ArgumentParser(description='SwinCXR - Train or evaluate chest X-ray classification model')
parser.add_argument('--images_path', default='/scratch/psheta12/CXR8/images/images', help='Path to directory containing chest X-ray images')
parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='Mode: train model or evaluate on test set')
parser.add_argument('--val_fraction', type=float, default=0.1, help='Fraction of official train_val list to use for validation (patient-level split)')
parser.add_argument('--split_seed', type=int, default=42, help='Seed for deterministic train/val split')
args = parser.parse_args()

# Set configuration parameters
mode = args.mode
PATH_TO_IMAGES = args.images_path
WEIGHT_DECAY = 1e-4           # L2 regularization strength
LEARNING_RATE = 0.0001        # Initial learning rate for optimizer
EPOCHS = 30                   # Number of training epochs
BATCH_SIZE = 32               # Batch size for training and evaluation
VAL_FRACTION = args.val_fraction
SPLIT_SEED = args.split_seed

print("=" * 60)
print("SwinCXR - Chest X-ray Classification System")
print("=" * 60)
print(f"Mode: {mode.upper()}")
print(f"Images path: {PATH_TO_IMAGES}")

if mode == "train":
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Weight decay: {WEIGHT_DECAY}, Val fraction: {VAL_FRACTION}, Split seed: {SPLIT_SEED}")
    # Train the SwinCXR model
    model, best_epoch = M.train_transformer(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, BATCH_SIZE, val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED)
    print(f"Training completed. Best epoch: {best_epoch}")
elif mode == "eval":
    print("Loading model checkpoint for evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_best = torch.load('results/checkpoint', map_location=device)
    state_dict = checkpoint_best.get('model_state_dict')
    if state_dict is None:
        legacy_model = checkpoint_best.get('model')
        if legacy_model is None:
            raise KeyError("Checkpoint missing model weights. Expected 'model_state_dict' or legacy 'model'.")
        state_dict = legacy_model.state_dict()
    model = M.timm.create_model(M.MODEL_NAME, pretrained=False, num_classes=M.NUM_LABELS)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("Model loaded successfully")
    
# Generate predictions and calculate AUC scores
print("Running evaluation on test dataset...")
preds, aucs = E.make_pred_multilabel(model, PATH_TO_IMAGES)
print("Evaluation complete. Results saved to results/preds.csv and results/aucs.csv")
