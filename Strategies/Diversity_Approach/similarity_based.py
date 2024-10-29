from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def _similarity_based_sampling(pool_features, pool_indices, train_features, train_indices, budget_per_iter, available_pool_indices):
    """
    Adds samples to the training set using similarity sampling strategy.
    """
    max_similarities = []

    for pool_feature in pool_features:
        similarities = cosine_similarity([pool_feature], train_features)
        max_similarity = np.max(similarities)
        max_similarities.append(max_similarity)

    selected_indices = np.argsort(max_similarities)[:budget_per_iter]
    temp = np.array(available_pool_indices)
    selected_indices = temp[selected_indices]

    for i in selected_indices:
        index = pool_indices.index(i)

        train_features = np.append(train_features, [pool_features[index]], axis=0)
        train_indices.append(i)

        pool_features = np.delete(pool_features, index, axis=0)
        pool_indices.pop(index)

    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices, train_features, pool_features, pool_indices
