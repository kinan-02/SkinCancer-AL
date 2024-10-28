from PIL import Image
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Define image transformations (resize, convert to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_mapping = {
    "actinic keratosis": 0,
    "basal cell carcinoma": 1,
    "dermatofibroma": 2,
    "melanoma": 3,
    "nevus": 4,
    "pigmented benign keratosis": 5,
    "squamous cell carcinoma": 6,
    "vascular lesion": 7
}

class dataset(Dataset):
    def __init__(self, dataframe, transform, train='train'):
        self.dataframe = dataframe
        self.train = train
        self.transform = transform
        self.path_to_image = self._create_path_to_image_dict()
        self.paths = list(self.path_to_image.keys())
        self.labels = list(self.path_to_image.values())

    def _create_path_to_image_dict(self):
        """
        Create a dictionary that maps image paths to their corresponding labels.
        """
        path_to_image = {}
        for index, row in self.dataframe.iterrows():
            if self.train == 'train':
                img_path = os.path.join('../train_dataset/', row['isic_id'] + '.jpg')
            elif self.train == 'test':
                img_path = os.path.join('../test_dataset/', row['isic_id'] + '.jpg')
            else:
                img_path = os.path.join('../validation_dataset/', row['isic_id'] + '.jpg')
            label = row['diagnosis']
            path_to_image[img_path] = label
        return path_to_image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """
        Get an image and its corresponding label and index.
        """
        img_path = self.paths[index]
        img_label = self.labels[index]
        image = Image.open(img_path)
        image = self.transform(image)
        label = class_mapping[img_label]  # Convert label to integer regardless of train/val/test
        return image, label, index

def import_data_loaders():
    train_df = pd.read_csv('../train_dataset/metadata.csv')
    test_df = pd.read_csv('../test_dataset/metadata.csv')
    val_df = pd.read_csv('../validation_dataset/metadata.csv')

    # Wrap datasets in DataLoaders
    train_dataset = dataset(train_df, transform, train='train')
    val_dataset = dataset(val_df, transform, train='val')
    test_dataset = dataset(test_df, transform, train='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader
