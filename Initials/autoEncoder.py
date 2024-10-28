import torch
import torch.nn as nn

# This is the class of the AutoEncoder that we  build to extract the latent vector for each image.

class Autoencoder(nn.Module):
    def __init__(self, encoded_dim=256):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # input channels=3 for RGB, output channels=32
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (112x112)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (56x56)
        )

        # Flatten and fully connected layer to get 1D vector
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 56 * 56, encoded_dim)  # From 32x56x56 to encoded_dim

        # Decoder
        self.fc2 = nn.Linear(encoded_dim, 32 * 56 * 56)  # Fully connected to expand back
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (32, 56, 56)),  # Reshape 1D vector back to (32x56x56)
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (112x112)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (224x224)
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Output channels=3 for RGB
            nn.Sigmoid()  # Output should be between 0 and 1 for normalized RGB
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        x = self.flatten(x)  # Flatten to 1D vector
        x = self.fc1(x)  # Project to encoded_dim

        # Decoding
        x = self.fc2(x)  # Expand back to match the flattened shape of feature maps
        x = self.decoder(x)  # Pass through the decoder
        return x

