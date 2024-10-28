import numpy as np
from collections import defaultdict
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from Strategies.Random import _random_sampling
from Strategies.Uncertainty_Approach.Uncertainty_entropy_based import _uncertainty_sampling


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
                 test_indices,
                 selection_criterion,
                 iterations,
                 budget_per_iter,
                 num_epochs, device, optimizer, val_loader, test_loader, train_df):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        self.available_pool_indices = available_pool_indices
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.selection_criterion = selection_criterion
        self.num_epochs = num_epochs
        self.train_df = train_df

    def run_pipeline(self):
        """
        Run the active learning pipeline
        :return
        accuracy_scores: list, accuracy scores at each iteration
        """
        accuracy_scores = []
        for iteration in range(self.iterations + 1):
            print(f"--------- Number of Iteration {iteration} ---------")
            train_images = [self.train_df.__getitem__(index)[0] for index in self.train_indices]
            label_df = [class_mapping[self.train_df.__getitem__(index)[1]] for index in self.train_indices]
            self._train_model(train_images, label_df)
            # loading the best model weights in each iteration
            if iteration != 0:
                self.model.load_state_dict(torch.load(f"best_{self.selection_criterion}_model.pth"))
            accuracy = self._evaluate_model()
            accuracy_scores.append(accuracy)
            self._sampling_strategy()
        return accuracy_scores

    def _sampling_strategy(self):
        if self.selection_criterion == 'random':
            self.available_pool_indices, self.train_indices = _random_sampling(self.available_pool_indices,
                                                                               self.budget_per_iter,
                                                                               self.train_indices)
        elif self.selection_criterion == 'uncertainty_sampling':
            self.available_pool_indices, self.train_indices = _uncertainty_sampling(self.model, self.train_df,
                                                                                    self.available_pool_indices,
                                                                                    self.train_indices, self.device,
                                                                                    self.budget_per_iter)

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
                torch.save(self.model.state_dict(), f"best_{self.selection_criterion}_model.pth")
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
                # outputs = outputs.logits
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
