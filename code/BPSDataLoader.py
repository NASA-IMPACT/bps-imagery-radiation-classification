import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from config import NUM_CLASS, RAD_LABEL
from utils import compute_padding


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class RadDataLoader(Dataset):
    def __init__(self, data_file, label, num_class, transform=None):
        self.data_path = data_file
        self.label = label
        self.blank_label = np.zeros((label.shape[0], num_class))
        self.transform = transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        data_path = self.data_path.iloc[idx]
        class_label = self.label.iloc[idx]
        self.blank_label[idx][class_label] = 1
        raw_img = imread(data_path)
        ascol = raw_img.reshape(-1, 1)
        ascol_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(ascol)
        scaled_img = ascol_scaled.reshape(raw_img.shape)
        # compute padding required
        padding = compute_padding(scaled_img.shape)
        # padding
        resized_img = np.pad(scaled_img, padding, 'symmetric')
        scaled_img = np.reshape(resized_img, (1, resized_img.shape[0], resized_img.shape[1]))
        torch_data = torch.tensor(scaled_img, dtype=torch.float32)
        torch_label = torch.tensor(self.blank_label[idx, :], dtype=torch.float32)

        if self.transform:
            torch_data = self.transform(torch_data)

        return torch_data, torch_label


def cnn_data_processor(data_dict):
    """
    Generates a pandas dataframe with image paths and labels
    :param data_dict:(dict) {radiation:[file_paths]}
    :return: (pandas dataframe) contains path to the images and their corresponding
    labels
    """
    rad_label_map = {rad: RAD_LABEL[rad] for rad in data_dict}
    data = {'Data': [], 'Label': []}
    for radiation, files in data_dict.items():
        for file in files:
            data['Data'].append(file)
            data['Label'].append(rad_label_map[radiation])
    pd_data = pd.DataFrame(data=data)
    return pd_data


def load_torch_data(data_path):
    """
    Load train/validation/test data for the model
    :param data_path: (dict) {radiation:[file_paths]}
    :return: train_data: Train object of Dataset class
             val_data: Validation object of Dataset class
             test_data: Test object of Dataset class
    """
    rotation_transform = RotationTransform(angles=[-180, -90, 0, 90, 180])
    data_label_df = cnn_data_processor(data_path)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        data_label_df.iloc[:, 0], data_label_df.iloc[:, 1], test_size=0.1, random_state=10
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.2, random_state=10
    )

    train_data = RadDataLoader(train_paths, train_labels, NUM_CLASS, rotation_transform)
    val_data = RadDataLoader(val_paths, val_labels, NUM_CLASS, rotation_transform)
    test_data = RadDataLoader(test_paths, test_labels, NUM_CLASS, rotation_transform)

    return train_data, val_data, test_data
