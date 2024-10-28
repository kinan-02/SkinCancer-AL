import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
def get_pool_loader(available_pool_indices, train_df):
    X_unlabeled = [train_df.__getitem__(index)[0] for index in self.available_pool_indices]

    pool_images_tensor = torch.stack(X_unlabeled)
    pool_dataset = TensorDataset(pool_images_tensor)

    batch_size = 32
    pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)
    return pool_loader


def calculate_uncertainty(outputs, c):
    probabilities = torch.cat(outputs, dim=0)
    # ignoring the paddings
    if c != 0:
        probabilities = probabilities[:-c]
    probabilities_cpu = probabilities.cpu().numpy()
    uncertainties = entropy(probabilities_cpu, axis=1)
    selected_indices = np.argsort(uncertainties)[-budget_per_iter:]
    temp = np.array(available_pool_indices)
    selected_indices = temp[selected_indices]
    # taking the indices of the samples that their uncertainty is less than the threshold
    high_confidence_indices = np.where(uncertainties < confidence_threshold)[0]
    high_confidence_labels = np.argmax(probabilities_cpu[high_confidence_indices], axis=1)
    high_confidence_indices = temp[high_confidence_indices]
    return selected_indices, high_confidence_indices, high_confidence_labels


def _uncertainty_ceal_sampling(confidence_threshold, model, device, available_pool_indices, train_df, budget_per_iter,
                               train_indices):
    """
    The Cost effective active learning strategy
    """
    pool_loader = get_pool_loader(available_pool_indices, train_df)
    model.eval()
    outputs = []
    c = 0
    with torch.no_grad():
        for inputs in pool_loader:
            inputs = inputs[0]
            inputs = inputs.to(device)
            x = model(inputs)
            # padding the batch with zero victors to ensure that all the batches have the same size
            if x.shape[0] != batch_size:
                padding_tensor = torch.full((batch_size - x.shape[0], 8), 0).to(device)
                c = batch_size - x.shape[0]
                x = torch.cat([x, padding_tensor])
            outputs.append(x)
    selected_indices, high_confidence_indices, high_confidence_labels = calculate_uncertainty(outputs, c)

    train_indices = train_indices + selected_indices.tolist()

    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices, high_confidence_labels, high_confidence_indices
