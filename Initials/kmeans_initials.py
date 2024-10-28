import numpy as np
import pickle
from ..DataSet import import_data_loaders
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from Initials.autoEncoder import Autoencoder
import torch
import torchvision.transforms as transforms
from Initials.ViT import get_vit_model
import random

def set_seed():
    random.seed(0)  # Set seed for NumPy
    np.random.seed(0)  # Set seed for PyTorch (for both CPU and GPU)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_features(dataloader, model, feature_extractor, device, vit_flag=True):
    """
    Return the latent vector for each image and the corresponding indices.

    """
    features_list = []
    indices_list = []

    with torch.no_grad():
        for images, _, indices in dataloader:
            if vit_flag == False:
                x = images.to(device)
                with torch.no_grad():
                    x = model.encoder(x)
                    x = model.flatten(x)
                    x = model.fc1(x)
            else:
                images_list = [transforms.ToPILImage()(img) for img in images]
                inputs = feature_extractor(images=images_list, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)

                x = outputs.last_hidden_state[:, 0, :]

            features_list.append(x.cpu().numpy())
            indices_list.extend(indices)

    features = np.vstack(features_list)
    return features, indices_list


def get_representative_images(train_features, indices, kmeans):
    """
    Get the representative images (indices) for each cluster.
    """
    cluster_to_images = {}
    for i in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_features = train_features[cluster_indices]
        distances = np.linalg.norm(cluster_features - kmeans.cluster_centers_[i], axis=1)
        nearest_indices = cluster_indices[np.argsort(distances)[:1]]
        cluster_to_images[i] = [indices[idx] for idx in nearest_indices]
    return cluster_to_images


def kmeans_rep(train_features, train_indices):
    n_clusters = 30
    # Apply K-Means clustering on the train_features
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    kmeans.fit(train_features)
    representative_images = get_representative_images(train_features, train_indices, kmeans)
    print(representative_images)
    # Get a list of the initial indices for the train dataset.
    return list(ids.item() for l in representative_images.values() for ids in l)


def main():
    set_seed()
    train_df, val_loader, test_loader = import_data_loaders()
    ae_model = Autoencoder()
    ae_model.load_state_dict(torch.load(f"ae_model.pth"))
    vit_model, feature_extractor = get_vit_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model.to(device)
    train_loader = DataLoader(train_df, batch_size=32, shuffle=True)
    ae_train_features, ae_train_indices = extract_features(train_loader, ae_model, None, device, False)
    ae_initials = kmeans_rep(ae_train_features, ae_train_indices)
    vit_train_features, vit_train_indices = extract_features(train_loader, vit_model, feature_extractor, device, True)
    vit_initials = kmeans_rep(vit_train_features, vit_train_indices)
    with open('ae_initials.pkl', 'wb') as file:
        # Write the list to the file using pickle
        pickle.dump(ae_initials, file)

    with open('vit_initials.pkl', 'wb') as file:
        # Write the list to the file using pickle
        pickle.dump(vit_initials, file)
