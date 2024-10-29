import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def _get_poolLoader(available_pool_indices, train_df):
    X_unlabeled = [train_df.__getitem__(index)[0] for index in available_pool_indices]

    pool_images_tensor = torch.stack(X_unlabeled)
    pool_dataset = TensorDataset(pool_images_tensor)

    batch_size = 32
    pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)
    return pool_loader


def calc_pscores_cdf(outputs, c):
    probabilities = torch.cat(outputs, dim=0)
    if c != 0 :
        probabilities = probabilities[:-c]
    probabilities_cpu = probabilities.cpu().numpy()
    p_scores = []
    for prob in probabilities_cpu:
        max_prob = np.max(prob)
        p_score = (1 - max_prob) / max_prob
        p_scores.append(p_score)

    p_scores = np.array(p_scores)
    cdf = np.cumsum(p_scores) / np.sum(p_scores)
    return p_scores, cdf


def sample_indices(selected_indices, budget_per_iter, p_scores, cdf):
    if len(selected_indices) > 0:
        if selected_indices.shape[0] < budget_per_iter:
            sampled_indices = selected_indices.tolist()
            sorted_p_scores_indices = np.argsort(p_scores)
            remaining_indices = np.setdiff1d(sorted_p_scores_indices, selected_indices)
            additional_samples = remaining_indices[:budget_per_iter - len(selected_indices)]
            sampled_indices.extend(additional_samples)
        else:
            selected_cdf_values = cdf[selected_indices]
            selected_cdf_norm = selected_cdf_values / np.sum(selected_cdf_values)
            sampled_indices = np.random.choice(selected_indices, budget_per_iter, p=selected_cdf_norm,
                                               replace=False)
    else:
        sampled_indices = np.random.choice(len(p_scores), budget_per_iter, replace=False)
    return sampled_indices


def update_indices(available_pool_indices, sampled_indices, train_indices):
    temp = np.array(available_pool_indices)
    selected_indices = temp[sampled_indices]
    train_indices = train_indices + selected_indices.tolist()
    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices


def _competence_based_sampling(model, itr, train_df, available_pool_indices, device, iterations, budget_per_iter,
                               train_indices):
    pool_loader = _get_poolLoader(available_pool_indices, train_df)
    batch_size = 32
    model.eval()
    outputs = []
    c=0
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
    p_scores, cdf = calc_pscores_cdf(outputs, c)
    C0 = 0.5
    c_t = min(1, np.sqrt((itr / iterations) * (1 - C0 ** 2) / iterations + C0 ** 2))

    selected_indices = np.where(p_scores > c_t)[0]
    # update train_indices and available pool indices
    sampled_indices = sample_indices(selected_indices, budget_per_iter, p_scores, cdf)
    available_pool_indices, train_indices = update_indices(available_pool_indices, sampled_indices, train_indices)
    return available_pool_indices, train_indices
