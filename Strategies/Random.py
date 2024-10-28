import numpy as np


def _random_sampling(available_pool_indices, budget_per_iter, train_indices):
    """
    Random sampling strategy
    """
    selected_indices = np.random.choice(available_pool_indices, budget_per_iter, replace=False)
    selected_indices = selected_indices.tolist()
    train_indices = train_indices + selected_indices

    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices
