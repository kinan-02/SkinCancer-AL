import os.path
from transformers import ViTFeatureExtractor, ViTModel
import numpy as np
from collections import defaultdict
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from Strategies.Random import _random_sampling
from Strategies.Uncertainty_Approach.Uncertainty_entropy_based import _uncertainty_sampling
from Strategies.Uncertainty_Approach.competence_based import _competence_based_sampling
from Strategies.Uncertainty_Approach.deepfool import _adversial_attack_sampling
from Strategies.Uncertainty_Approach.prediction_probability_based import _pred_prob_based_sampling
from Strategies.Uncertainty_Approach.ceal import _uncertainty_ceal_sampling
from Strategies.Diversity_Approach.kmeans_budget_vit import _kmeans_sampling
from Strategies.Diversity_Approach.kmeans_NumClasses import _kmeans_NumClasses_sampling
from Strategies.Diversity_Approach.similarity_based import _similarity_based_sampling
from Strategies.Hybrid_Approach.BADGE import _badge_sampling
from Strategies.Hybrid_Approach.Uncertainty_KMeans import _uncertainty_kmeans_sampling
from Strategies.Hybrid_Approach.custom_sampling import _custom_sampling
from Strategies.Uncertainty_Approach.Uncertainty_Combination import _custom_2_sampling
from Strategies.Hybrid_Approach.Pred_Prob_Kmeans import _Pred_prob_kmeans_sampling
from torchvision import transforms
from Initials.ViT import get_vit_model


def set_seed():
    random.seed(0)  # Set seed for NumPy
    np.random.seed(0)  # Set seed for PyTorch (for both CPU and GPU)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class_mapping = {
    "actinic keratosis": 0,
    "basal cell carcinoma": 1,
    "dermatofibroma": 2,
    "melanoma": 3,
    "nevus": 4,
    "pigmented benign keratosis": 5,
    "squamous cell carcinoma": 6,
    "vascular lesion": 7
}


class ActiveLearningPipeline:
    def __init__(self, model,
                 available_pool_indices,
                 train_indices,
                 selection_criterion,
                 iterations,
                 budget_per_iter,
                 num_epochs,
                 device, optimizer, val_loader, test_loader, train_df, appraoch,C0):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        self.available_pool_indices = available_pool_indices
        self.train_indices = train_indices
        self.selection_criterion = selection_criterion
        self.num_epochs = num_epochs
        self.train_df = train_df
        self.approach = appraoch
        self.C0 =C0
        # CEAL
        self.high_confidence_labels = []
        self.high_confidence_indices = []
        # Diversity Approach
        self.pool_features = []
        self.pool_indices = []
        self.train_features = []

    def run_pipeline(self):
        """
        Run the active learning pipeline
        :return
        accuracy_scores: list, accuracy scores at each iteration
        """
        accuracy_scores = []
        confidence_threshold = 0.05
        dr = 0.0033
        if self.approach == "Diversity" or (self.approach == "Hybrid" and self.selection_criterion != 'BADGE'):
            self._get_features()
        for iteration in range(self.iterations + 1):
            print(f"--------- Number of Iteration {iteration} ---------")
            # Preparing train data
            train_images = [self.train_df.__getitem__(index)[0] for index in self.train_indices]
            label_df = [self.train_df.__getitem__(index)[1] for index in self.train_indices]
            if self.selection_criterion == 'ceal':
                train_images = train_images + [self.train_df.__getitem__(index)[0] for index in
                                               self.high_confidence_indices]
                label_df = label_df + list(self.high_confidence_labels)
            self._train_model(train_images, label_df)
            # loading the best model weights in each iteration
            if iteration != 0:
                path = os.path.join('best_models', f"best_{self.selection_criterion}_model.pth")
                self.model.load_state_dict(torch.load(path))
            # Evaluate Model
            accuracy = self._evaluate_model()
            accuracy_scores.append(accuracy)
            # Selecting Strategy
            if self.approach == "Uncertainty":
                self._uncertainty_sampling_strategy(iteration, confidence_threshold)
            elif self.approach == "Diversity":
                self._diversity_sampling_strategy()
            else:
                self._hybrid_sampling_strategy()
            confidence_threshold -= dr * iteration
        return accuracy_scores

    def _hybrid_sampling_strategy(self):
        if self.selection_criterion in ['BADGE','BADGE_30'] :
            self.available_pool_indices, self.train_indices = _badge_sampling(self.model, self.train_df,
                                                                              self.available_pool_indices,
                                                                              self.train_indices, self.device,
                                                                              self.budget_per_iter)
        if self.selection_criterion == 'Uncertainty_KMeans':
            self.available_pool_indices, self.train_indices, self.pool_features, self.pool_indices = _uncertainty_kmeans_sampling(
                self.model, self.train_df, self.available_pool_indices, self.device, self.budget_per_iter,
                self.iterations,
                self.pool_features, self.pool_indices, self.train_indices)
        if self.selection_criterion == 'Custom':
            self.available_pool_indices, self.train_indices, self.train_features, self.pool_features, self.pool_indices = _custom_sampling(
                self.model, self.available_pool_indices, self.train_df, self.train_indices, self.train_features,
                self.pool_features,
                self.budget_per_iter, self.pool_indices, self.device)
        if self.selection_criterion == 'Pred_Prob_Kmeans':
            self.available_pool_indices, self.train_indices, self.pool_features, self.pool_indices = _Pred_prob_kmeans_sampling(
                self.model, self.train_df, self.available_pool_indices, self.device, self.budget_per_iter, self.iterations,
                self.pool_features, self.pool_indices, self.train_indices)

    def _diversity_sampling_strategy(self):
        if self.selection_criterion == 'kmeans_budget':
            self.available_pool_indices, self.train_indices, self.pool_features, self.pool_indices = (
                _kmeans_sampling(self.available_pool_indices, self.budget_per_iter, self.pool_features,
                                 self.pool_indices, self.train_indices))
        if self.selection_criterion == 'KMeans_Nearest':
            self.available_pool_indices, self.train_indices, self.pool_features, self.pool_indices = _kmeans_NumClasses_sampling(
                self.pool_features, self.pool_indices, self.train_indices, self.available_pool_indices, True)
        if self.selection_criterion == 'KMeans_NearestFarthest':
            self.available_pool_indices, self.train_indices, self.pool_features, self.pool_indices = _kmeans_NumClasses_sampling(
                self.pool_features, self.pool_indices, self.train_indices, self.available_pool_indices, False)
        if self.selection_criterion == 'Similarity_based':
            self.available_pool_indices, self.train_indices, self.train_features, self.pool_features, self.pool_indices = _similarity_based_sampling(
                self.pool_features, self.pool_indices, self.train_features, self.train_indices, self.budget_per_iter,
                self.available_pool_indices)

    def _uncertainty_sampling_strategy(self, itr, confidence_threshold):
        if self.selection_criterion == 'random':
            self.available_pool_indices, self.train_indices = _random_sampling(self.available_pool_indices,
                                                                               self.budget_per_iter,
                                                                               self.train_indices)
        elif self.selection_criterion in ['uncertainty_ae','uncertainty_vit']:
            self.available_pool_indices, self.train_indices = _uncertainty_sampling(self.model, self.train_df,
                                                                                    self.available_pool_indices,
                                                                                    self.train_indices, self.device,
                                                                                    self.budget_per_iter)
        elif self.selection_criterion in ['competence_based','competence_based'+str(self.C0*100)]:
            self.available_pool_indices, self.train_indices = _competence_based_sampling(self.model, itr,
                                                                                         self.train_df,
                                                                                         self.available_pool_indices,
                                                                                         self.device,
                                                                                         self.iterations + 1,
                                                                                         self.budget_per_iter,
                                                                                         self.train_indices,self.C0)
        elif self.selection_criterion in ['deepfool_100','deepfool_200']:
            self.available_pool_indices, self.train_indices = _adversial_attack_sampling(
                self.available_pool_indices,
                self.train_df, self.model,
                self.device,
                self.budget_per_iter,
                self.train_indices)
        elif self.selection_criterion in ['pred_prob','pred_prob_30']:
            self.available_pool_indices, self.train_indices = _pred_prob_based_sampling(self.model, self.train_df,
                                                                                        self.available_pool_indices,
                                                                                        self.device,
                                                                                        self.budget_per_iter,
                                                                                        self.train_indices)
        elif self.selection_criterion == 'ceal':
            self.available_pool_indices, self.train_indices, self.high_confidence_labels, self.high_confidence_indices = _uncertainty_ceal_sampling(
                confidence_threshold, self.model, self.device, self.available_pool_indices, self.train_df,
                self.budget_per_iter,
                self.train_indices)
        elif self.selection_criterion == 'uncertainty_comb':
            self.available_pool_indices, self.train_indices = _custom_2_sampling(self.model, itr, 0.5,
                                                                                 self.available_pool_indices,
                                                                                 self.device, self.train_df,
                                                                                 self.iterations,
                                                                                 self.budget_per_iter,
                                                                                 self.train_indices)

    def calculate_class_weights(self, label_counts, num_classes=8):
        """
        this function is to caculate the inverse probability of each class in the data
        """
        total_samples = sum(label_counts.values())
        class_weights = torch.zeros(num_classes)

        for cls in range(num_classes):
            if cls in label_counts:
                class_weights[cls] = total_samples / (num_classes * label_counts[cls])
            else:
                class_weights[cls] = 1.0  # Handle the case where a class has zero samples in the current epoch

        return class_weights

    def _train_epoch(self, train_loader, optimizer, loss_f):
        self.model.train()
        running_loss = 0.0  # Track the running loss
        correct_predictions = 0
        total_predictions = 0
        # Training loop
        for inputs, labels in train_loader:
            inputs = inputs
            inputs = inputs.to(self.device)
            labels = torch.tensor(labels).to(self.device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = loss_f(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += inputs.shape[0]
        return running_loss, correct_predictions, total_predictions

    def _get_train_data(self, train_images, label_df):
        train_images_tensor = torch.stack(train_images)
        label_df_tensor = torch.tensor(label_df)
        train_dataset = TensorDataset(train_images_tensor, label_df_tensor)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader

    def _train_model(self, train_images, label_df):
        label_counts = defaultdict(int)
        for label in label_df:
            label_counts[label] += 1
        class_weights = self.calculate_class_weights(label_counts, 8).to(self.device)
        # Giving higher weight for the loss of samples that their class is a minority in the data while giving less weight
        # to the loss for classes that are majority
        loss_f = nn.CrossEntropyLoss(weight=class_weights)
        train_loader = self._get_train_data(train_images, label_df)
        best_acc = 0
        for epoch in range(self.num_epochs):
            running_loss, correct_predictions, total_predictions = self._train_epoch(train_loader, self.optimizer,
                                                                                     loss_f)
            # Print loss and accuracy at the end of each epoch
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct_predictions.double() / total_predictions
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
            # saving the best model weights on the validation
            val_acc = self._check_model()
            if val_acc > best_acc:
                best_acc = val_acc
                path = os.path.join('best_models', f"best_{self.selection_criterion}_model.pth")
                torch.save(self.model.state_dict(), path)
        print("--" * 30)

    def _check_model(self):
        """
        this function is used to evaluate the model on the validation set
        """
        self.model.eval()
        running_corrects = 0
        total_predictions = 0.0

        with torch.no_grad():
            for inputs, labels, _ in self.val_loader:
                inputs = inputs.to(self.device)
                labels = torch.tensor(labels).to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels)
                total_predictions += inputs.shape[0]
        val_acc = running_corrects.double() / total_predictions
        return val_acc.item()

    def _evaluate_model(self):
        """
        Evaluate the model on the test set
        :return:
        accuracy: float, accuracy of the model
        """
        self.model.eval()
        running_corrects = 0
        total_predictions = 0.0
        with torch.no_grad():
            for inputs, labels, _ in self.test_loader:
                inputs = inputs.to(self.device)
                labels = torch.tensor(labels).to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels)
                total_predictions += inputs.shape[0]
        test_acc = running_corrects.double() / total_predictions
        return test_acc.item()

    def extract_features(self, dataloader, model, feature_extractor):
        """
        Return the latent vector for each image and the corresponding indices.
        """
        features_list = []
        indices_list = []

        with torch.no_grad():
            for images, indices in dataloader:
                images = images.to(self.device)
                images_list = [transforms.ToPILImage()(img) for img in images]
                inputs = feature_extractor(images=images_list, return_tensors="pt")
                with torch.no_grad():
                    inputs = inputs.to(self.device)
                    outputs = model(**inputs)

                x = outputs.last_hidden_state[:, 0, :]
                features_list.append(x.cpu().numpy())

                # Collect indices
                indices_list.extend(indices)

        # Stack all features into a 2D array (n_samples, hidden_dim)
        features = np.vstack(features_list)

        return features, indices_list

    def _get_features(self):
        """
        Creates latent feature vectors for each image in the available pool using a pre-trained Vision Transformer (ViT) model from Google.
        """
        model, feature_extractor = get_vit_model()
        # feature_extractor = feature_extractor.to(self.device)
        model = model.to(self.device)

        train_images = [self.train_df.__getitem__(index)[0] for index in self.train_indices]
        train_images_tensor = torch.stack(train_images)
        label_df_tensor = torch.tensor(self.train_indices)
        train_dataset = TensorDataset(train_images_tensor, label_df_tensor)
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.train_features, self.train_indices = self.extract_features(train_loader, model, feature_extractor)

        X_unlabeled = [self.train_df.__getitem__(index)[0] for index in self.available_pool_indices]
        pool_images_tensor = torch.stack(X_unlabeled)
        pool_indices_tensor = torch.tensor(self.available_pool_indices)
        pool_dataset = TensorDataset(pool_images_tensor, pool_indices_tensor)

        batch_size = 32
        pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)

        self.pool_features, self.pool_indices = self.extract_features(pool_loader, model, feature_extractor)
