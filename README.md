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

You can install Python libraries using pip. For example:

```bash
pip install jupyter pandas numpy torch matplotlib transformers torchvision scikit-learn scipy Pillow 
 ```
## How to Get The Data
The dataset used for this project, including training, validation, and test sets, is available for download via the following Google Drive link:
[Download Dataset from Google Drive](https://drive.google.com/drive/folders/18qQkydqVpx-HI3q6IalgT4bV9hwU_ivr)

### Instructions:
1. Click on the link above to access the dataset.
2. Download the entire dataset folder.
3. After downloading, unzip the folder (if applicable) and place the contents in the appropriate directories.

### Dataset Structure:
- **train_dataset/**: Contains the images and csv file for the training set.
- **test_dataset/**: Contains the images and csv file for the test set.
- **validation_dataset/**: Contains the images and csv file for the validation set.

Before running any scripts, make sure the dataset is set up correctly in the project directory and that you unzip the files you downloaded.

## How to Run

## 1. Obtaining the Initial Training Data
Before running the main training loop, you need to initialize the training set using KMeans++ clustering on the extracted feature vectors. Follow the instructions below based on the model you wish to use for feature extraction:

### Option 1: Using Vision Transformer (ViT) Model
If you want to initialize the training set using the pre-trained Vision Transformer (ViT) model for feature extraction:

Open and run the `kmeans_google_vit.ipynb` notebook.
This notebook will:

- Extract feature vectors for all images using the ViT model.
- Apply KMeans++ clustering to select the initial set of training samples.

Ensure the required dependencies (such as torch, sklearn, and the ViT pre-trained model) are installed.
After running the notebook, the initialized training set will be ready for use in subsequent steps.

### Option 2: Using Autoencoder Model
If you want to initialize the training set using an autoencoder that we trained:

Download the pre-trained autoencoder model from our Google Drive (link provided in the "How to Get the Data" section).
Place the pre-trained model in the appropriate directory.
Open and run the `kmeans_auto_encoder.ipynb` notebook.

This notebook will:

- Load the pre-trained autoencoder model.
- Extract feature vectors for all images using the autoencoder.
- Apply KMeans++ clustering to select the initial set of training samples.
  
#### Note:

In all strategies, we initialize the training set using KMeans++ with the kmeans_google_vit notebook by default. If you want to use the autoencoder for initialization, you must modify the corresponding cells in the training script accordingly.


## 2. Running Different Sampling Strategies

All custom sampling strategies for the project are located in the `Strategies` folder. Each strategy is implemented in its own notebook, allowing you to easily experiment with different approaches to active learning.

#### Steps to Run a Strategy:

**Choose a Strategy**: 
   - Navigate to the `Strategies` folder.
   - Each notebook in this folder corresponds to a different active learning strategy. Review the strategies and select the one you want to run.
   - Notebooks are named to reflect the specific strategy they implement (e.g., `random_sampling.ipynb`, `uncertainty_sampling.ipynb`, `custom_sampling_strategy.ipynb`, etc.).
     
#### Note:

- Each strategy has its own parameters and methods for selecting samples from the unlabeled pool, so you may want to experiment with different strategies to see which performs best for your task.
- The **default initialization** of the training set is performed using the Vision Transformer (ViT) model with KMeans++. If you wish to use the Autoencoder for initialization, you’ll need to modify the relevant cells in the notebook as explained in the previous section.

## 3. Visualizing Results

   - After running a strategy, you can visualize the results.
   - If you would like to generate the plots that were used in our paper, you need to run the `plots.ipynb` notebook.
   
   The `plots.ipynb` file generates all the relevant visualizations, including:
   - Model performance over iterations.
   - Comparisons of different strategies.
   - Accuracy vs. number of samples selected, and more.
     
 #### Note:
 
   Ensure that the output of your chosen strategy is saved correctly, as the plotting notebook will rely on this data to create the figures.

---
This completes the instructions for running the notebooks and processing the data for our project. We encourage you to explore the final results, consider potential improvements, and possibly extend the project with your ideas.

**Thank you for following through with our project workflow!**
