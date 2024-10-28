import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_pool_loader(train_df, available_pool_indices):
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

def _pred_prob_based_sampling(model, train_df, available_pool_indices, device, budget_per_iter, train_indices):
    """
    prediction probability based strategy
    """
    # Get pool Loader
    batch_size = 32
    pool_loader = get_pool_loader(train_df, available_pool_indices)
    model.eval()
    outputs = []
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

    # Calculate p_score
    p_scores = calculate_p_score(outputs, c)

    # taking budget per iteration samples that have the highest p_score
    selected_indices = np.argsort(p_scores)[-budget_per_iter:]
    temp = np.array(available_pool_indices)
    selected_indices = temp[selected_indices]

    train_indices = train_indices + selected_indices.tolist()

    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices