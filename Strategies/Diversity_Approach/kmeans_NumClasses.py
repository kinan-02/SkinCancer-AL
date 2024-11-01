import numpy as np
from sklearn.cluster import KMeans


def get_representative_images(kmeans, pool_features, pool_indices, flag_Nearest):
    """

    :return:  returns a dictionary where the keys are the index of the cluster and the values are the closest images to
     each centroid if the flag_Nearest is True else the values are the 4 closest
    and 3 farthest images to each centroid  the note that the K of the KMeans is the number of classes.
    """
    cluster_to_images = {}
    for i in range(kmeans.n_clusters):
        # Get the indices of all images in the current cluster
        cluster_indices = np.where(kmeans.labels_ == i)[0]

        # Extract features of the images in the current cluster
        cluster_features = pool_features[cluster_indices]

        # Compute distances between each feature and the cluster centroid
        distances = np.linalg.norm(cluster_features - kmeans.cluster_centers_[i], axis=1)

        # Map the cluster number to the index of the representative image
        if flag_Nearest:
            nearest_indices = cluster_indices[np.argsort(distances)[:7]]
            # Map the cluster number to the indices of the top k nearest images
            cluster_to_images[i] = [pool_indices[idx] for idx in nearest_indices]

        else:
            # Another approach
            nearest_indices = cluster_indices[np.argsort(distances)[:4]]
            farthest_indices = cluster_indices[np.argsort(distances)[-3:]]
            cluster_to_images[i] = [pool_indices[idx] for idx in nearest_indices] + [pool_indices[idx] for idx in
                                                                                     farthest_indices]
    return cluster_to_images


def _kmeans_NumClasses_sampling(pool_features, pool_indices, train_indices, available_pool_indices, flag_Nearest):
    """
    Kmeans sampling strategy
    """
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    kmeans.fit(pool_features)

    representative_images = get_representative_images(kmeans, pool_features, pool_indices, flag_Nearest)
    selected_indices = list(ids.item() for l in representative_images.values() for ids in l)

    # After Selecting the relevant indices (samples), we need to delete them from the pool and them to the train.
    for i in selected_indices:
        index = pool_indices.index(i)
        pool_features = np.delete(pool_features, index, axis=0)
        pool_indices.pop(index)

    train_indices = train_indices + selected_indices

    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices, pool_features, pool_indices
