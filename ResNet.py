import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.optim import Adam

class ourResNet():
    def __init__(self):
        # Load pre-trained ResNet50 model from torchvision
        self.base_model = models.resnet50(pretrained=True)
        # Add a fully connected layer witht the number of classes (for the prediction).
        num_classes = 8
        self.base_model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    def get_model(self):
        # Freeze all layers except the fully connected ones
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the final fully connected layer
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

        optimizer = Adam(self.base_model.fc.parameters(), lr=0.0008)

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(device)
        return self.base_model, optimizer