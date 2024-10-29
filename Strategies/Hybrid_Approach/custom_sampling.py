import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_train_loader(train_df, train_indices):
    train_images = [train_df.__getitem__(index)[0] for index in train_indices]
    label_df = [train_df.__getitem__(index)[1] for index in train_indices]
    train_images_tensor = torch.stack(train_images)
    label_df_tensor = torch.tensor(label_df)
    indices_tensor = torch.tensor(train_indices)
    train_dataset = TensorDataset(train_images_tensor, label_df_tensor, indices_tensor)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_loader


def get_selected_indices(pool_features, misclassified_features, budget_per_iter,
                         available_pool_indices):
    max_similarities = []
    for pool_feature in pool_features:
        # Compute cosine similarity between pool features and misclassified features
        similarities = cosine_similarity([pool_feature], misclassified_features)
        max_similarity = np.max(similarities)
        max_similarities.append(max_similarity)

        # selected_indices is the indices of the samples from the pool that have the highest similarity to the misclassified samples
    selected_indices = np.argsort(max_similarities)[:budget_per_iter]
    temp = np.array(available_pool_indices)
    selected_indices = temp[selected_indices]
    return selected_indices


def update_indices(selected_indices, pool_indices, train_indices, available_pool_indices, train_features,
                   pool_features):
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


def _custom_sampling(model, available_pool_indices, train_df, train_indices, train_features, pool_features,
                     budget_per_iter, pool_indices, device):
    """
    Adds samples to the training set using a custom sampling strategy.
    """
    misclassified_indices = []
    train_loader = get_train_loader(train_df, train_indices)
    with torch.no_grad():
        for idx, (inputs, labels, indices) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            incorrect = (preds != labels).cpu().numpy()
            misclassified_indices += [i for i, incorrect_flag in zip(indices, incorrect) if incorrect_flag]
    # An array containing the feature vectors of the samples that the model misclassified.
    misclassified_features = np.array([train_features[train_indices.index(i)] for i in misclassified_indices])

    selected_indices = get_selected_indices(pool_features, misclassified_features,
                                            budget_per_iter,
                                            available_pool_indices)
    return update_indices(selected_indices, pool_indices, train_indices, available_pool_indices, train_features,
                   pool_features)
