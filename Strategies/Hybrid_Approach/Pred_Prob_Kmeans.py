import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import entropy


def get_representative_images(kmeans, pool_features, pool_indices):
    """
   returns a dictionary where the keys are the index of the cluster and the values are the closest images to
   each centroid note that the K of the KMeans is the budget per iteration.
    """

    cluster_to_images = {}
    for i in range(kmeans.n_clusters):
        # Get the indices of all images in the current cluster
        cluster_indices = np.where(kmeans.labels_ == i)[0]

        # Extract features of the images in the current cluster
        cluster_features = pool_features[cluster_indices]

        # Compute distances between each feature and the cluster centroid
        distances = np.linalg.norm(cluster_features - kmeans.cluster_centers_[i], axis=1)

        nearest_indices = cluster_indices[np.argsort(distances)[:4]]
        farthest_indices = cluster_indices[np.argsort(distances)[-3:]]
        # Map the cluster number to the index of the representative image
        cluster_to_images[i] = [pool_indices[idx] for idx in nearest_indices] + [pool_indices[idx] for idx in
                                                                                 farthest_indices]

    return cluster_to_images


def _kmean_uncertin_samples(selected_indices, budget_per_iter, pool_features, pool_indices):
    """
    returns the selected indices
    """
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    kmeans.fit(pool_features[selected_indices])

    pool_indices_np = np.array(pool_indices)
    representative_images = get_representative_images(kmeans, pool_features[selected_indices],
                                                      pool_indices_np[selected_indices])
    selected_indices = list(ids.item() for l in representative_images.values() for ids in l)

    for i in selected_indices:
        index = pool_indices.index(i)
        pool_features = np.delete(pool_features, index, axis=0)
        pool_indices.pop(index)
    return selected_indices, pool_features, pool_indices


def get_DataLoader(train_df, available_pool_indices):
    X_unlabeled = [train_df.__getitem__(index)[0] for index in available_pool_indices]

    pool_images_tensor = torch.stack(X_unlabeled)
    pool_dataset = TensorDataset(pool_images_tensor)

    batch_size = 32
    pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)
    return pool_loader

def calculate_p_score(outputs, c):
    probabilities = torch.cat(outputs, dim=0)
    # ignoring the paddings
    if c != 0:
        probabilities = probabilities[:-c]
    probabilities_cpu = probabilities.cpu().numpy()
    p_scores = []
    # caculating the p_score for each sample
    for prob in probabilities_cpu:
        max_prob = np.max(prob)
        p_score = (1 - max_prob) / max_prob
        p_scores.append(p_score)
    p_scores = np.array(p_scores)
    return p_scores

def select_indices(outputs, c, budget_per_iter, iterations, pool_features, pool_indices, train_indices,
                   available_pool_indices):

    p_scores = calculate_p_score(outputs, c)

    selected_indices = np.argsort(p_scores)[-(budget_per_iter * iterations):]

    selected_indices, pool_features, pool_indices = _kmean_uncertin_samples(selected_indices, budget_per_iter, pool_features,
                                                                            pool_indices)

    train_indices = train_indices + selected_indices

    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices, pool_features, pool_indices


def _Pred_prob_kmeans_sampling(model, train_df, available_pool_indices, device, budget_per_iter, iterations,
                                 pool_features, pool_indices, train_indices):
    """
    Adds samples to the training set using a uncertainty-kmeans sampling strategy.
    """

    batch_size = 32
    pool_loader = get_DataLoader(train_df, available_pool_indices)
    model.eval()
    outputs = []
    with torch.no_grad():
        for inputs in pool_loader:
            inputs = inputs[0]
            inputs = inputs.to(device)
            x = model(inputs)
            if x.shape[0] != batch_size:
                padding_tensor = torch.full((batch_size - x.shape[0], 8), 0).to(device)
                c = batch_size - x.shape[0]
                x = torch.cat([x, padding_tensor])
            outputs.append(x)

    return select_indices(outputs, c, budget_per_iter, iterations, pool_features, pool_indices, train_indices,
                          available_pool_indices)
