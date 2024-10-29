from sklearn.cluster import KMeans
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_loader(train_df, available_pool_indices):
    X_unlabeled = [train_df.__getitem__(index)[0] for index in available_pool_indices]

    pool_images_tensor = torch.stack(X_unlabeled)
    pool_dataset = TensorDataset(pool_images_tensor)

    batch_size = 32
    pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)
    return pool_loader

def update_indices(train_indices, available_pool_indices, selected_indices):
    temp = np.array(available_pool_indices)
    selected_indices = temp[selected_indices]

    train_indices = train_indices + selected_indices.tolist()

    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices

def grad_kmeans(gradient_embeddings, budget_per_iter, available_pool_indices, train_indices):
    gradient_embeddings = np.concatenate(gradient_embeddings, axis=0)
    num_samples_to_select = budget_per_iter
    # applying KMenas++ on the gradients embeddings
    kmeans = KMeans(n_clusters=num_samples_to_select, init='k-means++', random_state=0).fit(gradient_embeddings)
    # updating the pool and train indices
    centroids = kmeans.cluster_centers_
    selected_indices = []
    for i in range(num_samples_to_select):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        closest_sample_index = min(cluster_indices,
                                   key=lambda idx: np.linalg.norm(gradient_embeddings[idx] - centroids[i]))
        selected_indices.append(closest_sample_index)
    return update_indices(train_indices, available_pool_indices, selected_indices)

def _badge_sampling(model, train_df, available_pool_indices, train_indices, device, budget_per_iter):
    """
    BADGE sampling strategy
    """

    pool_loader = get_loader(train_df, available_pool_indices)
    model.eval()
    gradient_embeddings = []
    def extract_grad_hook(module, grad_input, grad_output):
        gradient_embeddings.append(grad_output[0].detach().cpu().numpy())

    # calculating the gradients based on the last layer of the model
    handle = model.fc[3].register_full_backward_hook(extract_grad_hook)

    for inputs in pool_loader:
        inputs = inputs[0].to(device)
        inputs.requires_grad = False
        outputs = model(inputs)

        predicted_labels = torch.argmax(outputs, dim=1).to(device)
        loss = torch.nn.functional.cross_entropy(outputs, predicted_labels)

        model.zero_grad()

        loss.backward()
    handle.remove()
    return grad_kmeans(gradient_embeddings, budget_per_iter, available_pool_indices, train_indices)