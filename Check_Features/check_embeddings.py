import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DataSet.data import import_data_loaders
import numpy as np
import torch, pickle, random
from torchvision import transforms
from Initials.ViT import get_vit_model
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def set_seed():
    random.seed(0)  # Set seed for NumPy
    np.random.seed(0)  # Set seed for PyTorch (for both CPU and GPU)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features(dataloader, model, feature_extractor, device):
        """
        Return the latent vector for each image and the corresponding indices.
        """
        features_list = []
        indices_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels, indices in dataloader:
                images = images.to(device)
                images_list = [transforms.ToPILImage()(img) for img in images]
                inputs = feature_extractor(images=images_list, return_tensors="pt")
                with torch.no_grad():
                    inputs = inputs.to(device)
                    outputs = model(**inputs)

                x = outputs.last_hidden_state[:, 0, :]
                features_list.append(x.cpu().numpy())

                # Collect indices
                indices_list.extend(indices)
                labels_list.extend(labels)
        # Stack all features into a 2D array (n_samples, hidden_dim)
        features = np.vstack(features_list)

        return features, labels_list, indices_list


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading data...")
    train_df, val_loader, test_loader = import_data_loaders()
    model, feature_extractor = get_vit_model()
    model = model.to(device)
    train_loader = DataLoader(train_df, batch_size=32, shuffle=False)
    print("Generating features...")
    features, labels, indices = extract_features(train_loader, model, feature_extractor, device)
    print("KNN classifier...")
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)
    distances, indices = knn.kneighbors(features)
    k_nearest_labels = np.array(labels)[indices]

    weighted_percentages = []
    for i in range(len(labels)):
        sample_label = labels[i]
        neighbors_labels = np.array(k_nearest_labels[i])
        neighbors_distances = distances[i]
        # Avoid division by zero by adding a small value to distances
        weights = 1 / (neighbors_distances + 1e-8)  # Inverse of the distance
        # Normalize weights to sum to 1
        normalized_weights = weights / np.sum(weights)
        # Create a boolean array where True means the label matches the sample's label
        unmatches = (neighbors_labels != sample_label).astype(float)
        # Calculate the weighted sum of matches
        weighted_unmatch_percentage = np.sum(unmatches * normalized_weights) * 100
        weighted_percentages.append(weighted_unmatch_percentage)
    # Now `percentages` contains the percentage of matching neighbors for each sample
    percentages = np.mean(weighted_percentages)
    print(percentages)
    # y_pred = knn.predict(features)
    # accuracy = accuracy_score(labels, y_pred)
    # print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
