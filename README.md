# SwinCXR

Swin Transformer fine-tuning for NIH ChestX-ray14 (14-label multi-label classification) with official splits, ready-to-run training and inference scripts, and recorded results.

## Setup
- Python 3.8+ and a CUDA-capable GPU are required.
- Install deps: `pip install -r requirements.txt`
- Data: place ChestX-ray14 images under `/scratch/psheta12/CXR8/images/images` (or pass your path). Official split files `train_val_list.txt` and `test_list.txt` are expected under `/scratch/psheta12/CXR8` by default.
- A sample test image is provided at `test.jpg`.

## Training and Evaluation
- Uses Swin Large (`swin_large_patch4_window7_224`) from `timm`, AdamW, `BCEWithLogitsLoss`, batch size 32, LR `1e-4`, weight decay `1e-4`.
- Official NIH split lists are honored. Validation is carved from the official `train_val_list.txt` at the patient level.
- Reproducibility: `val_fraction=0.1`, `split_seed=42`.
- Commands:
  ```bash
  # Train with validation split and early stopping (patience 3)
  python train_eval.py --mode train --images_path /scratch/psheta12/CXR8/images/images --val_fraction 0.1 --split_seed 42

  # Evaluate saved checkpoint on the official test split
  python train_eval.py --mode eval --images_path /scratch/psheta12/CXR8/images/images
  ```
- SLURM example: see `main_job.sh` (A100, 16 CPUs, 120G RAM, runs train then eval).

## Inference
```bash
python run_inference.py --image_path /path/to/chest_xray.jpg
# or rely on the bundled test.jpg
python run_inference.py
```
Outputs per-class probabilities for the 14 findings.

## Results (current run)
- Dataset: NIH ChestX-ray14, official test list.
- Early stopped at epoch 7; best val loss at epoch 4.
- Average AUC: **0.813**.
- Per-class AUCs (from `results/aucs.csv`):

| Class               | AUC   |
| ------------------- | ----- |
| Atelectasis         | 0.780 |
| Cardiomegaly        | 0.873 |
| Effusion            | 0.829 |
| Infiltration        | 0.691 |
| Mass                | 0.818 |
| Nodule              | 0.761 |
| Pneumonia           | 0.733 |
| Pneumothorax        | 0.883 |
| Consolidation       | 0.760 |
| Edema               | 0.851 |
| Emphysema           | 0.920 |
| Fibrosis            | 0.824 |
| Pleural_Thickening  | 0.775 |
| Hernia              | 0.887 |

Artifacts from this run are under `results/` (checkpoint ~780 MB, `preds.csv`, `aucs.csv`, and SLURM logs).

## Notes
- Code assumes GPU training; CPU is not supported for full runs.
- Uses official NIH labels `nih_labels.csv`; keep split files alongside the dataset or pass `--images_path` and `--split_seed` as needed.
