import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
from PIL import Image
from torch.utils.data import DataLoader

print_config()
#### Load training data ######

# Define the directory containing training data
data_dir = "data/raw/Training/"

# Get sorted list of class directories
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)

# Collect image file paths organized by class
image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]

# Flatten image paths and corresponding class labels
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i])
num_total = len(image_class)

# Get image dimensions from first image
image_width, image_height = Image.open(image_files_list[0]).size

print(f"Training data loaded from {data_dir}")
print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")

# Create training datasets
train_x = [image_files_list[i] for i in range(num_total)]
train_y = [image_class[i] for i in range(num_total)]


#### Load testing data ######

# Define the directory containing testing data
test_data_dir = "data/raw/Testing/"

# Get sorted list of class directories
test_class_names = sorted(x for x in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, x)))
num_test_class = len(test_class_names)

# Collect image file paths organized by class
test_image_files = [
    [os.path.join(test_data_dir, test_class_names[i], x) for x in os.listdir(os.path.join(test_data_dir, test_class_names[i]))]
    for i in range(num_test_class)
]
num_test_each = [len(test_image_files[i]) for i in range(num_test_class)]   
test_image_files_list = []
test_image_class = []
for i in range(num_test_class):
    test_image_files_list.extend(test_image_files[i])
    test_image_class.extend([i] * num_test_each[i])
num_test_total = len(test_image_class)
# Get image dimensions from first test image
test_image_width, test_image_height = Image.open(test_image_files_list[0]).size

print(f"Testing data loaded from {test_data_dir}")
print(f"Total test image count: {num_test_total}")
print(f"Test image dimensions: {test_image_width} x {test_image_height}")
print(f"Test label names: {test_class_names}")
print(f"Test label counts: {num_test_each}")

# Create testing datasets

test_x = [test_image_files_list[i] for i in range(num_test_total)]
test_y = [test_image_class[i] for i in range(num_test_total)]


### PREPROCESSING 


## Define transformations for training and validation
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ]
)

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])

### FINAL DATA LOADERS AND PREP OF DATASETS ###

class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_ds = BrainTumorDataset(train_x, train_y, train_transforms)
train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)

#val_ds = BrainTumorDataset(val_x, val_y, val_transforms)
#val_loader = DataLoader(val_ds, batch_size=300, num_workers=10)

test_ds = BrainTumorDataset(test_x, test_y, train_transforms)
test_loader = DataLoader(test_ds, batch_size=300, num_workers=10)

## check type and shape of data items
print("TRAIN")
print(f"Data item type: {type(train_ds[0])}")
print(f"Image shape: {train_ds[0][0].shape}, Label: {train_ds[0][1]}")
print(type(train_loader))
print("TEST")
print(f"Data item type: {type(test_ds[0])}")
print(f"Image shape: {test_ds[0][0].shape}, Label: {test_ds[0][1]}")



