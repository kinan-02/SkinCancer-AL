import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.stats import entropy


def get_pool_loader(train_df, available_pool_indices):
    X_unlabeled = [train_df.__getitem__(index)[0] for index in available_pool_indices]

    pool_images_tensor = torch.stack(X_unlabeled)
    pool_dataset = TensorDataset(pool_images_tensor)

    batch_size = 32
    pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)
    return pool_loader


def calculate_p_score(probabilities_cpu, uncertainties, w):
    p_scores = []
    for prob in probabilities_cpu:
        max_prob = np.max(prob)
        p_score = (1 - max_prob)
        p_scores.append(p_score)
    p_scores = np.array(p_scores)
    p_scores = w * p_scores + (1 - w) * uncertainties
    return p_scores


def select_indices(itr, iterations, p_scores, budget_per_iter):
    C0 = 0.5
    c_t = min(1, np.sqrt((itr / iterations) * (1 - C0 ** 2) / iterations + C0 ** 2))
    cdf = np.cumsum(p_scores) / np.sum(p_scores)
    selected_indices = np.where(p_scores > c_t)[0]

    if len(selected_indices) > 0:
        if selected_indices.shape[0] < budget_per_iter:
            sampled_indices = selected_indices.tolist()
            sorted_p_scores_indices = np.argsort(p_scores)
            remaining_indices = np.setdiff1d(sorted_p_scores_indices, selected_indices)
            additional_samples = remaining_indices[:budget_per_iter - len(selected_indices)]
            sampled_indices.extend(additional_samples)
        else:
            #if we have samples that are more than the budget we randomly pick samples from them
            #using the cdf distribution in the sampling
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


def _custom_2_sampling(model, itr, w, available_pool_indices, device, train_df, iterations, budget_per_iter,
                       train_indices):
    batch_size = 32
    pool_loader = get_pool_loader(train_df, available_pool_indices)
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
    probabilities = torch.cat(outputs, dim=0)
    if c != 0:
        probabilities = probabilities[:-c]
    probabilities_cpu = probabilities.cpu().numpy()
    uncertainties = entropy(probabilities_cpu, axis=1)
    p_scores = calculate_p_score(probabilities_cpu, uncertainties, w)
    sampled_indices = select_indices(itr, iterations, p_scores, budget_per_iter)
    return update_indices(available_pool_indices, sampled_indices, train_indices)
