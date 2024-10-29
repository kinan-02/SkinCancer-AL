import foolbox as fb
import numpy as np


def adversial_attack(adversarial_images, device, inputs, model, index, fmodel, epsilons):
    inputs = inputs.to(device)
    x = model(inputs)
    _, preds = torch.max(x, 1)
    for image, pred, i in zip(inputs, preds, index):
        image = image.unsqueeze(0).to(device)
        # Convert `pred` into a tensor with the correct shape for Foolbox
        pred_tensor = torch.tensor([pred.item()], dtype=torch.int64).to(device)
        # Run DeepFool attack with epsilons
        adversarial_images_eps, perturbations, success_status = attack(fmodel, image,
                                                                       criterion=fb.criteria.Misclassification(
                                                                           pred_tensor), epsilons=epsilons)
        adversarial_image = \
            attack(fmodel, image, criterion=fb.criteria.Misclassification(pred_tensor), epsilons=0.01)[0]
        # Calculating the norm between each image and it's adversial.
        k = torch.norm((adversarial_image - image).view(image.size(0), -1), dim=1)
        adversarial_images.append((k, i))


def get_pool_loader(available_pool_indices, train_df):
    X_unlabeled = [train_df.__getitem__(index)[0] for index in available_pool_indices]
    pool_images_tensor = torch.stack(X_unlabeled)
    pool_indices_tensor = torch.tensor(available_pool_indices)
    pool_dataset = TensorDataset(pool_images_tensor, pool_indices_tensor)
    batch_size = 32
    pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)
    return pool_loader


def _adversial_attack_sampling(available_pool_indices, train_df, model, device, budget_per_iter, train_indices):
    """
    Adversial attack (DeepFool) sampling strategy.
    """
    pool_loader = get_pool_loader(available_pool_indices, train_df)
    # Load the attack strategy.
    fmodel = fb.PyTorchModel(model, bounds=(0, 255))
    attack = fb.attacks.FGSM()
    adversarial_images = []
    outputs = []
    model.eval()
    epsilons = [0.01]
    for inputs, index in pool_loader:
        adversial_attack(adversarial_images, device, inputs, model, index, fmodel, epsilons)
    # Sort the images according to the norm with the adversial, and taking the samples with the lowest norm.
    adversarial_images = sorted(adversarial_images, key=lambda x: x[0])
    selected_indices = [t[1] for t in adversarial_images[:budget_per_iter]]
    train_indices = train_indices + selected_indices
    available_pool_set = set(available_pool_indices)
    train_set = set(train_indices)
    available_pool_indices = list(available_pool_set - train_set)
    return available_pool_indices, train_indices
