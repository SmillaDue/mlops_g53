import os
from pathlib import Path

import cv2
import imutils
import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirst, RandFlip, RandRotate, RandZoom, Resize, ToTensor, Transform
from PIL import Image

# hyperparameter
IMG_SIZE = 256


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def crop_img(img):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[
        extTop[1] - ADD_PIXELS : extBot[1] + ADD_PIXELS, extLeft[0] - ADD_PIXELS : extRight[0] + ADD_PIXELS
    ].copy()

    return new_img


data_transforms = Compose(
    [
        lambda img: cv2.imread(str(img)),
        crop_img,
        ToTensor(),
        EnsureChannelFirst(channel_dim=-1),
        Resize((IMG_SIZE, IMG_SIZE)),
        normalize,
    ]
)


#### Load training data ######
def load_data(raw_data_dir: str):
    """Load training and testing data from raw data directory.
    Args:
        raw_data_dir (str): Path to the raw data directory containing 'Training' and 'Testing' subdirectories.

    Returns:
        train_x (list): List of file paths for training images.
        train_y (list): List of class labels for training images, [0,1,2,3]
        test_x (list): List of file paths for testing images.
        test_y (list): List of class labels for testing images, [0,1,2,3]
        label_names (list): List of class names corresponding to labels.
    """

    train_data_dir = os.path.join(raw_data_dir, "Training")
    test_data_dir = os.path.join(raw_data_dir, "Testing")

    # Get sorted list of class directories
    class_names = sorted(x for x in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, x)))
    num_class = len(class_names)

    # Collect image file paths organized by class
    image_files = [
        [
            os.path.join(train_data_dir, class_names[i], x)
            for x in os.listdir(os.path.join(train_data_dir, class_names[i]))
        ]
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

    # Get image dimensions from first image, the images are processed to be of same size
    image_width, image_height = Image.open(image_files_list[0]).size

    # Create training datasets
    train_x = [image_files_list[i] for i in range(num_total)]
    train_y = [image_class[i] for i in range(num_total)]

    print(f"Training data loaded from {train_data_dir}")
    print(f"Total image count: {num_total}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")

    # Get sorted list of class directories
    test_class_names = sorted(x for x in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, x)))
    num_test_class = len(test_class_names)

    # Collect image file paths organized by class
    test_image_files = [
        [
            os.path.join(test_data_dir, test_class_names[i], x)
            for x in os.listdir(os.path.join(test_data_dir, test_class_names[i]))
        ]
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

    return train_x, train_y, test_x, test_y, class_names


def preprocess_data(raw_data_dir: str, processed_data_dir: str):
    """
    Process raw data and save to processed data directory.

    Preprocessing steps:
    - Load images and labels from raw data directory.
    - Apply transformations: cropping, resizing, normalization, cropping
    - Save processed images and labels as .pt files.

    Reference for preprocessing function: https://github.com/masoudnick/Brain-Tumor-MRI-Classification/blob/main/Preprocessing.py
    """

    #### Load training data ######
    train_x, train_y, test_x, test_y, class_names = load_data(raw_data_dir)  # lists containing file paths and labels

    # TRAINING DATA
    # ### PREPROCESSING
    processed_x, processed_y = [], []
    for path_str, label in zip(train_x, train_y):
        processed_img = data_transforms(path_str)  # (3,H, W)
        processed_img = processed_img.float()
        processed_x.append(processed_img)
        processed_y.append(label)

    X = torch.stack(processed_x, dim=0)  # (N,3,H,W)
    y = torch.tensor(processed_y, dtype=torch.long)  # (N,), long

    out_dir = Path(processed_data_dir)
    torch.save(X, out_dir / "train_images.pt")
    torch.save(y, out_dir / "train_targets.pt")

    # TESTING DATA
    processed_test_x, processed_test_y = [], []
    for path_str, label in zip(test_x, test_y):
        img_path = Path(path_str)
        processed_img = data_transforms(img_path)  # (3,H, W)
        processed_img = processed_img.float()

        processed_test_x.append(processed_img)
        processed_test_y.append(label)

    X_test = torch.stack(processed_test_x, dim=0)  # (N,3,H,W)
    y_test = torch.tensor(processed_test_y, dtype=torch.long)  # (N,), long

    torch.save(X_test, out_dir / "test_images.pt")
    torch.save(y_test, out_dir / "test_targets.pt")

    with open(out_dir / "labels.txt", "w") as output:
        output.write(str(class_names))


def brain_tumor() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt", weights_only=False)
    train_target = torch.load("data/processed/train_targets.pt", weights_only=False)
    test_images = torch.load("data/processed/test_images.pt", weights_only=False)
    test_target = torch.load("data/processed/test_targets.pt", weights_only=False)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    preprocess_data("data/raw", "data/processed")
