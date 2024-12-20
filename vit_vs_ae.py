from AL_Pipeline import ActiveLearningPipeline
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DataSet.data import import_data_loaders
import numpy as np
import torch, pickle, random
from collections import defaultdict
from ResNet import ourResNet
def set_seed():
    random.seed(0)  # Set seed for NumPy
    np.random.seed(0)  # Set seed for PyTorch (for both CPU and GPU)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print("Starting the PipeLine...")
    train_df, val_loader, test_loader = import_data_loaders()
    with open('Initials/vit_initials.pkl', 'rb') as file:
        vit_initials = pickle.load(file)
    with open('Initials/ae_initials.pkl', 'rb') as file:
        ae_initials = pickle.load(file)
    available_pool_indices = []
    for i in range(len(train_df)):
        image, label, index = train_df[i]
        available_pool_indices.append(index)

    iterations = 20
    budget_per_iter = 60
    num_epoch = 15
    selection_criteria = ['uncertainty_vit', 'uncertainty_ae']
    accuracy_scores_dict = defaultdict(list)

    for i, criterion in enumerate(selection_criteria):
        if i == 0:
            initials = vit_initials
        else:
            initials = ae_initials
        set_seed()
        resnet = ourResNet()
        model, optimizer, device = resnet.get_model()

        AL_class = ActiveLearningPipeline(model=model,
                                          available_pool_indices=available_pool_indices,
                                          train_indices=initials,
                                          selection_criterion=criterion,
                                          iterations=iterations,
                                          budget_per_iter=budget_per_iter,
                                          num_epochs=num_epoch,
                                          device=device,
                                          optimizer=optimizer,
                                          val_loader=val_loader,
                                          test_loader=test_loader,
                                          train_df=train_df)
        accuracy_scores_dict[criterion] = AL_class.run_pipeline()

    with open('vit_vs_ae_accuracy.pkl', 'wb') as file:
        # Write the list to the file using pickle
        pickle.dump(accuracy_scores_dict, file)

if __name__ == "__main__":
    main()
