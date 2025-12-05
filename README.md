# SwinCXR

A production-ready deep learning model for classifying chest X-ray images from the ChestX-ray14 dataset using a state-of-the-art Swin Transformer architecture.

## Overview

The **SwinCXR** is an enterprise-grade solution designed to classify 14 different pathologies found in the ChestX-ray14 dataset. This project leverages the powerful [Swin Transformer](https://github.com/microsoft/Swin-Transformer) architecture as the backbone, which has been optimized for multi-class classification tasks specific to chest X-ray images. Developed for clinical deployment, this model provides radiologists and healthcare providers with an advanced AI assistant for chest X-ray interpretation.

### Key Features

- **Swin Transformer Backbone**: Utilizes the powerful Swin Transformer architecture for state-of-the-art feature extraction from radiological images.
- **14-Class Classification**: Optimized for detecting 14 distinct chest pathologies with high accuracy and reliability.
- **Production-Ready Inference**: Streamlined inference pipeline for rapid integration into clinical workflows.
- **Pre-trained Weights**: Access to rigorously validated model weights for immediate deployment.
- **Customizable Training**: Enterprise-grade configuration options for fine-tuning on proprietary datasets.
- **Comprehensive Documentation**: Detailed code comments and architectural explanations for easy maintenance.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Requirements

- **Python**: Ensure that Python is installed (version 3.7 or higher is recommended).
- **Packages**: Required packages are listed in `requirements.txt`. You can install them using:

  ```bash
  pip install -r requirements.txt
  ```

## Dataset

The model is trained and tested on the **ChestX-ray14** dataset, which contains images of various chest diseases. You can download the dataset from the [official NIH website](https://nihcc.app.box.com/v/ChestXray-NIHCC).

### Training

To train the model, run the following command:

```bash
python3 train_eval.py --mode "train" --images_path "/path_to_images/"
```

### Evaluation

To evaluate the model on the test dataset, execute the `train_eval.py` script:

```bash
python train_eval.py --mode "eval" --images_path "/path_to_images/"
```

The script will output the AUC scores for each disease class and save the results in the `results` directory.

### Inference

To run inference on a single chest X-ray image, use the `run_inference.py` script:

```bash
python run_inference.py --image_path "/path_to_image/chest_xray.jpg"
```

By default, the script will use `test.jpg` if no image path is provided:

```bash
python run_inference.py
```

The script will output the probability scores for each of the 14 disease classes, sorted by their likelihood.

## Model Architecture

The SwinCXR leverages the Swin Transformer architecture, known for its hierarchical representation and shifted windowing mechanism. The architecture has been carefully adapted to fit the 14-class problem specific to the ChestX-ray14 dataset, with optimizations for clinical deployment.

- **Hierarchical Design**: Efficiently models long-range dependencies with a pyramid structure for better understanding of global anatomical context.
- **Shifted Windowing**: Reduces computation by limiting self-attention to non-overlapping local windows while maintaining high accuracy.
- **Multi-scale Feature Extraction**: Captures both fine-grained details and broader contextual information critical for radiological interpretation.
- **Attention Mechanisms**: Focuses on relevant regions of the X-ray image that are most indicative of pathologies.

## Results

The following table shows the AUC performance of the SwinCXR on the ChestX-ray14 dataset:

| Disease            | AUC   |
| ------------------ | ----- |
| Atelectasis        | 0.826 |
| Cardiomegaly       | 0.910 |
| Consolidation      | 0.815 |
| Edema              | 0.891 |
| Effusion           | 0.883 |
| Emphysema          | 0.916 |
| Fibrosis           | 0.836 |
| Hernia             | 0.942 |
| Infiltration       | 0.718 |
| Mass               | 0.858 |
| Nodule             | 0.787 |
| Pleural Thickening | 0.780 |
| Pneumonia          | 0.763 |
| Pneumothorax       | 0.877 |

**Average AUC**: 0.843

These results highlight the model's strong performance across various diseases, demonstrating its effectiveness in the classification task.

## Acknowledgments

- The [Swin Transformer](https://github.com/microsoft/Swin-Transformer) team for their innovative architecture.
- The National Institutes of Health (NIH) for providing the ChestX-ray14 dataset.
