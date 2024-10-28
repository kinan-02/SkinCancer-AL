import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.stats import entropy


def _uncertainty_sampling(model, train_df, available_pool_indices, train_indices, device, budget_per_iter):
    """
    Uncertainty-Entropy sampling strategy
    """
    X_unlabeled = [train_df.__getitem__(index)[0] for index in available_pool_indices]

    pool_images_tensor = torch.stack(X_unlabeled)
    pool_dataset = TensorDataset(pool_images_tensor)

    batch_size = 32
    pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    outputs = []
    with torch.no_grad():
        for inputs in pool_loader:
            inputs = inputs[0].to(device)
            x = model(inputs)
            if x.shape[0] != batch_size:
                padding_tensor = torch.full((batch_size - x.shape[0], 8), 0).to(device)
                c = batch_size - x.shape[0]
                x = torch.cat([x, padding_tensor])
            outputs.append(x)
    probabilities = torch.cat(outputs, dim=0)
    probabilities = probabilities[:-c]
    probabilities_cpu = probabilities.cpu().numpy()
    uncertainties = entropy(probabilities_cpu, axis=1)

    selected_indices = np.argsort(uncertainties)[-budget_per_iter:]
    temp = np.array(available_pool_indices)
    selected_indices = temp[selected_indices]

    train_indices = train_indices + selected_indices.tolist()

    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices
