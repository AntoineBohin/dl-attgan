import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd



class CustomDataset(Dataset):
    """
    Custom dataset loader for facial attribute datasets.
    Loads images and their corresponding attribute labels.
    """
    def __init__(self, dataset_path, attribute_labels_path, image_size, selected_attributes):
        self.dataset_path = dataset_path

        #attribute_data = pd.read_csv(attribute_labels_path)
        #attribute_names = list(attribute_data.columns[1:])
        #attribute_indices = [attribute_names.index(attr) + 1 for attr in selected_attributes]

        # Extract image filenames and corresponding attribute labels
        #self.image_names = attribute_data.iloc[:, 0].values
        #self.attribute_labels = attribute_data.iloc[:, attribute_indices].values

        attribute_names = open(attribute_labels_path, 'r', encoding='utf-8').readlines()[1].split()
        attribute_indices = [attribute_names.index(att) + 1 for att in selected_attributes]
        self.image_names = np.loadtxt(attribute_labels_path, skiprows=2, usecols=[0], dtype=str)
        self.attribute_labels = np.loadtxt(attribute_labels_path, skiprows=2, usecols=attribute_indices, dtype=int)
        print(self.image_names)
        print(self.attribute_labels)
        # Convert attribute labels from {-1, 1} to {0, 1}
        self.attribute_labels = (self.attribute_labels + 1) // 2

        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        """
        Returns the image and its corresponding attribute labels.
        """
        image = self.transforms(Image.open(os.path.join(self.dataset_path, self.image_names[index])))
        attributes = torch.tensor((self.attribute_labels[index] + 1) // 2)  
        return image, attributes
    
    def __len__(self):
        return len(self.image_names)



class CelebA(Dataset):
    def __init__(self, dataset_path, attribute_labels_path, image_size, dataset_mode, selected_attributes):
        """
        Args:
            dataset_path (str): Path to the dataset images.
            attribute_labels_path (str): Path to the attribute label file.
            image_size (int): Target size of the images.
            dataset_mode (str): Dataset split - 'train', 'valid', or 'test'.
            selected_attributes (list): List of selected attributes.
        """
        super(CelebA, self).__init__()
        self.dataset_path = dataset_path

        attribute_data = pd.read_csv(attribute_labels_path)
        attribute_names = list(attribute_data.columns[1:])
        attribute_indices = [attribute_names.index(attr) + 1 for attr in selected_attributes]

        # Extract image filenames and corresponding attribute labels
        image_names = attribute_data.iloc[:, 0].values
        attribute_labels = attribute_data.iloc[:, attribute_indices].values

        # Convert attribute labels from {-1, 1} to {0, 1}
        attribute_labels = (attribute_labels + 1) // 2
        
        if dataset_mode == 'train':
            self.image_names = image_names[:182000]
            self.attribute_labels = attribute_labels[:182000]
        if dataset_mode == 'valid':
            self.image_names = image_names[182000:182637]
            self.attribute_labels = attribute_labels[182000:182637]
        if dataset_mode == 'test':
            self.image_names = image_names[182637:]
            self.attribute_labels = attribute_labels[182637:]
        self.transforms = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])                    
        self.dataset_length = len(self.image_names)

    def __getitem__(self, index):
        image = self.transforms(Image.open(os.path.join(self.dataset_path, self.image_names[index])))
        attributes = torch.tensor((self.attribute_labels[index] + 1) // 2)  # Normalize to {0,1}
        return image, attributes
    
    def __len__(self):
        return self.dataset_length

def check_attribute_conflict(attribute_batch, target_attribute, attribute_list):
    """
    Ensures that selected attributes in a batch do not conflict with each other.
    Some attributes are mutually exclusive, meaning that setting one should disable the other.
    Example conflicts:
        - "Bald" vs. "Bangs"
        - "Black_Hair" vs. "Blond_Hair" vs. "Brown_Hair" vs. "Gray_Hair"
        - "Mustache" vs. "No_Beard"
    Args:
        attribute_batch (torch.Tensor): Batch of attribute vectors (one per image).
        target_attribute (str): The attribute to check for conflicts.
        attribute_list (list): List of all possible attributes.
    """
    def _get(attribute_vector, attribute_name):
        """Retrieve the value of a specific attribute from the attribute vector."""
        if attribute_name in attribute_list:
            return attribute_vector[attribute_list.index(attribute_name)]
        return None
    def _set(attribute_vector, value, attribute_name):
        """Set the value of a specific attribute in the attribute vector."""
        if attribute_name in attribute_list:
            attribute_vector[attribute_list.index(attribute_name)] = value
    
    target_index = attribute_list.index(target_attribute)
    for attributes in attribute_batch:
        if target_attribute in ['Bald', 'Receding_Hairline'] and attributes[target_index] != 0:
            if _get(attributes, 'Bangs') != 0:
                _set(attributes, 1 - attributes[target_index], 'Bangs')
        elif target_attribute == 'Bangs' and attributes[target_index] != 0:
            for conflicting_attr in ['Bald', 'Receding_Hairline']:
                if _get(attributes, conflicting_attr) != 0:
                    _set(attributes, 1 - attributes[target_index], conflicting_attr)
        elif target_attribute in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and attributes[target_index] != 0:
            for conflicting_attr in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if conflicting_attr != target_attribute and _get(attributes, conflicting_attr) != 0:
                    _set(attributes, 1 - attributes[target_index], conflicting_attr)
        elif target_attribute in ['Straight_Hair', 'Wavy_Hair'] and attributes[target_index] != 0:
            for conflicting_attr in ['Straight_Hair', 'Wavy_Hair']:
                if conflicting_attr != target_attribute and _get(attributes, conflicting_attr) != 0:
                    _set(attributes, 1 - attributes[target_index], conflicting_attr)
        elif target_attribute in ['Mustache', 'No_Beard'] and attributes[target_index] != 0:
            for conflicting_attr in ['Mustache', 'No_Beard']:
                if conflicting_attr != target_attribute and _get(attributes, conflicting_attr) != 0:
                    _set(attributes, 1 - attributes[target_index], conflicting_attr)
    return attribute_batch
