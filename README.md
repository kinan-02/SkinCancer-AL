<h1 align="center">A Robust Active Learning Strategy for Skin Cancer Detection: Combining Multiple Sampling Techniques</h1>
<p align="center">
  Samar Samara, Kinan ibraheem, Rawan Badarneh
  <p align="center">
    Technion
  </p>
</p>

# Overview

The project, titled *A Robust Active Learning Strategy for Skin Cancer Detection: Combining Multiple Sampling Techniques*, focuses on improving the classification of skin cancer images using active learning strategies to reduce the need for large labeled datasets. The main challenge in skin cancer classification is that collecting labeled data is expensive and time-consuming, as it often requires expert annotation. The goal of this research is to optimize the labeling process while maintaining high classification accuracy using active learning approaches.

ResNet50, a deep convolutional neural network, was selected as the base model due to its proven effectiveness in image classification tasks, including medical imaging. The project explored various active learning strategies, starting with baseline approaches such as random sampling and uncertainty entropy-based methods. From there, more sophisticated methods like CEAL (Cost-Effective Active Learning), DeepFool, prediction-probability-based, and competence-based strategies were introduced.

Additionally, density-based approaches, such as KMeans++ clustering and similarity-based methods, were implemented to diversify the selected samples. Finally, hybrid methods like BADGE and Uncertainty KMeans++ were applied to combine uncertainty and diversity in sample selection.

The experiments revealed that the feature extraction model significantly influences the performance of the active learning strategies. Overall, uncertainty-based approaches performed the best, while density-based methods underperformed, likely due to the homogeneity in the skin cancer dataset, which lacks sufficient diversity in the latent space

## Table of Contents

- [Installation](#installation)
- [Prerequisites](#Prerequisites)
- [How to Get The Data](#How-to-Get-The-Data)
- [How to deal with the missing data](#How-to-Deal-With-The-Missing-Data)
- [How to Join The Data](#How-to-Join-The-Data)
- [How to Run](#how-to-run)

## Installation
**Clone the Repository**:

   Clone this repository to your local machine using :

   ```sh
   git clone https://github.com/kinan-02/SkinCancer-AL.git
   cd SkinCancer-AL
   ```
## Prerequisites
Before you begin, ensure you have the following software installed on your computer:

- Jupyter Notebook
- pandas
- numpy
- torch
- matplotlib
- transformers
- torchvision
- sklearn
- scipy
- PIL
- argparse

You can install Python libraries using pip. For example:

```bash
pip install jupyter pandas numpy torch matplotlib transformers torchvision sklearn scipy Pillow argparse
 ```
## How to Get The Data
