# Vehicle Re-Identification using CNNs (PyTorch)

This project implements a vehicle re-identification (ReID) system using deep convolutional neural networks in PyTorch.  

---

## Project Overview

Vehicle re-identification aims to recognize the same vehicle across multiple camera views.  
In this project, a CNN backbone is trained using:

- Cross-Entropy Loss for classification
- Batch-Hard Triplet Loss for metric learning

The learned embeddings are evaluated using mAP and Rank-1 accuracy.

---

## Dataset

- Dataset: VeRi776 Vehicle Re-Identification Dataset  
- Structure:
⁠Each image filename encodes:
- vehicle ID (pid)
- camera ID (camid)

The dataset is loaded using a custom `VeRiDataset` class implemented in the notebook.

 Dataset link:  
https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset

---

##  Model Architecture

- Backbone: ResNet 
- Feature embedding with L2 normalization
- Classification head for identity prediction

The network outputs:
- Feature embeddings (for retrieval)
- Classification logits (for training)

---

## Training Details

- Framework: PyTorch
- Optimizer: Adam
- Losses:
- Cross-Entropy Loss
- Batch-Hard Triplet Loss
- Input resolution: 224 × 224
- Batch size: 32

Training is performed using GPU when available.

---

## Evaluation

Evaluation follows the standard ReID protocol:

1. Extract features for query and gallery sets
2. Compute pairwise distance matrix
3. Compute:
 - mean Average Precision (mAP)
 - Rank-1 Accuracy

### Example Results (after 2 epochs):
- mAP: ~46%
- Rank-1: ~81%

---

## How to Run

1. Clone the repository
2. Download and place the VeRi dataset in the expected folder structure
3. Open `notebook.ipynb`
4. Run all cells sequentially

---


## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- PIL
- tqdm

---

## Author

Raynold Mezini  