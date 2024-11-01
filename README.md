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

   Clone this repository to your local machine using:

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

Before running any scripts, make sure the dataset is set up correctly alongside the project directory and that you unzip the files you downloaded.

## How to Run

## 1. Obtaining the Initial Training Data
Before running the main training loop, you must initialize the training set using KMeans++ clustering on the extracted feature vectors. Follow the instructions below:

- Run the `kmeans_initials.py` script.
  
   ```sh
   python Initials/kmeans_initials.py
   ```
By running this command you will get two pickle files: `ae_initials.pkl` and `vit_initials.pkl`

- The two pickle files were provided in the Initials directory, so you can avoid running the script above.

Please make sure the required dependencies (such as torch, sklearn, and the ViT pre-trained model) are installed.
After running the script, the initialized training set will be ready for use in subsequent steps.

#### Note:
- Download the pre-trained autoencoder model from our Google Drive (link provided in the "How to Get the Data" section).
Place the pre-trained model in the appropriate directory.
- In all strategies, we initialize the training set using KMeans++ with the vit_initials.pkl by default. If you want to use the autoencoder for initialization, you must modify the corresponding commands in the active learning pipeline script accordingly.
- To compare the two performances using the initials mentioned above, Run the `vit_vs_ae.py` script.
  
   ```sh
   python vit_vs_ae.py
   ```

## 2. Running Different Sampling Strategies

All custom sampling strategies for the project are located in the `Strategies` folder. the `Strategies` folder consists of 3 folders each approach in one folder, in each approach folder you will find the corresponding strategies.

#### Run an approach:
  chosen approach = Uncertainty, Diversity, Hybrid, Adversial
  ```sh
   python Run_PipeLine/Run_PipeLine_{chosen approach}.py
   ```
     
#### Note:

- Each strategy has its parameters and methods for selecting samples from the unlabeled pool, so you may want to experiment with different strategies to see which performs best for your task.
- The **default initialization** of the training set is performed using the Vision Transformer (ViT) model with KMeans++. If you wish to use the Autoencoder for initialization, you’ll need to modify the relevant commands in the scripts as explained in the previous section.
- In the kmeans++ strategy we tested 3 different techniques, you can test them by choosing **{chosen approach} = Kmeans**.
  
## 3. Hyperparameter Testing
To optimize our model’s performance in active learning, we conducted several experiments with specific hyperparameters, which are documented in the following files:

- **Budget Testing**:  
  The `Run_PipeLine_Budget_30.py` file tests the effect of different budgets (30 and 60) selected for each active learning iteration, allowing us to analyze how varying sample sizes impact model performance over time.
  
   ```sh
   python Run_PipeLine/Run_PipeLine_Budget_30.py
   ```
   By running this command you will get a pickle file named `Budget_30_accuracy.pkl` that has the result of the BADGE, Prediction probability, and Uncertainty K-Means++ strategies.

  **NOTE**:
  - The default budget per iteration in the project is set to 60
  

- **Competence Strategy (`c0` constant)**:  
  The `Run_PipeLine_competence.py` file explores the `c0` constant in our competence-based strategy. The constant `c0` affects the model's selection criteria for samples, impacting the difficulty level of chosen samples as training progresses.

   ```sh
   python Run_PipeLine/Run_PipeLine_competence.py
   ```
   By running this command you will get a pickle file named `competence_based_C0_accuracy.pkl` that has the result of the competence strategy with c0 = 0.25, 0.5, 0.75.

   **NOTE**:
   - The default c0 = 0.5
     
## 4. Test our hypothesis

To check the hypothesis that we mentioned in the short discussion, Follow the instructions below:
- Run the `check_embeddings.py` script.
  
   ```sh
   python Check_Features/check_embeddings.py
   ```
   By running this command you will get the percentage of the mismatch.


## 5. Visualizing Results

   - After running a strategy, you can visualize the results.
   - If you would like to generate the plots that were used in our paper, you need to run the `plots.ipynb` notebook.
   
   The `plots.ipynb` file generates all the relevant visualizations, including:
   - Model performance over iterations.
   - Comparisons of different strategies.
   - Accuracy vs. number of samples selected, and more.
     
 #### Note:
 
   Ensure that the output of your chosen approach is saved correctly, as the plotting script will rely on this data to create the figures.

---
This completes the instructions for running the scripts and processing the data for our project. We encourage you to explore the final results, consider potential improvements, and possibly extend the project with your ideas.

**Thank you for following through with our project workflow!**
