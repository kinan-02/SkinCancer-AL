import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DataSet.data import import_data_loaders
import numpy as np
import torch, pickle, random
from torchvision import transforms
from Initials.ViT import get_vit_model
from torch.utils.data import TensorDataset, DataLoader

def set_seed():
    random.seed(0)  # Set seed for NumPy
    np.random.seed(0)  # Set seed for PyTorch (for both CPU and GPU)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features(dataloader, model, feature_extractor, device):
        """
        Return the latent vector for each image and the corresponding indices.
        """
        features_list = []
        indices_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels, indices in dataloader:
                images = images.to(device)
                images_list = [transforms.ToPILImage()(img) for img in images]
                inputs = feature_extractor(images=images_list, return_tensors="pt")
                with torch.no_grad():
                    inputs = inputs.to(device)
                    outputs = model(**inputs)

                x = outputs.last_hidden_state[:, 0, :]
                features_list.append(x.cpu().numpy())

                # Collect indices
                indices_list.extend(indices)
                labels_list.extend(labels)
        # Stack all features into a 2D array (n_samples, hidden_dim)
        features = np.vstack(features_list)

        return features, labels_list, indices_list


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_loader, test_loader = import_data_loaders()
    model, feature_extractor = get_vit_model()
    train_loader = DataLoader(train_df, batch_size=32, shuffle=True)
    features, labels, indices = extract_features(train_loader, model, feature_extractor, device)


if __name__ == "__main__":
    main()
