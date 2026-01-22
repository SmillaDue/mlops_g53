import os
import subprocess
from pathlib import Path

import torch
from google.cloud import storage

from mlops_project.utils import ArrayPreprocessing, TensorsPreprocessing, ensure_data_and_model

# subprocess.run(["dvc", "pull", "data"], check=True)

MODEL_BUCKET, MODEL_PREFIX, DATA_PREFIX, LOCAL_DATA, LOCAL_MODEL = ensure_data_and_model()

# hyperparameter
IMG_SIZE = 256

BASE_DIR = Path(os.environ.get("DATA_DIR", "data/processed"))


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

    # Create training datasets
    train_x = [image_files_list[i] for i in range(num_total)]
    train_y = [image_class[i] for i in range(num_total)]
    print(f"Training data loaded from {train_data_dir}")
    print(f"Total image count: {num_total}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")
    print("")

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

    print(f"Testing data loaded from {test_data_dir}")
    print(f"Total test image count: {num_test_total}")
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
    array_preprocessing = ArrayPreprocessing(img_size=IMG_SIZE)
    tensor_preprocessing = TensorsPreprocessing()

    imgs = [array_preprocessing(img_path) for img_path in train_x]  # list of tensors (1,H,W)
    imgs = tensor_preprocessing(imgs)

    X_train = imgs
    y_train = torch.tensor(train_y, dtype=torch.long)  # (N,), long

    out_dir = Path(processed_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(X_train, out_dir / "train_images.pt")
    torch.save(y_train, out_dir / "train_targets.pt")

    # TESTING DATA
    # ### PREPROCESSING
    array_preprocessing = ArrayPreprocessing(img_size=IMG_SIZE)
    tensor_preprocessing = TensorsPreprocessing()

    imgs = [array_preprocessing(img_path) for img_path in test_x]  # list of tensors (1,H,W)
    imgs = tensor_preprocessing(imgs)

    X_test = imgs
    y_test = torch.tensor(test_y, dtype=torch.long)  # (N,), long

    torch.save(X_test, out_dir / "test_images.pt")
    torch.save(y_test, out_dir / "test_targets.pt")

    with open(out_dir / "labels.txt", "w") as output:
        output.write(str(class_names))


def brain_tumor() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""

    train_images = torch.load(BASE_DIR / "train_images.pt", weights_only=False)
    train_target = torch.load(BASE_DIR / "train_targets.pt", weights_only=False)
    test_images = torch.load(BASE_DIR / "test_images.pt", weights_only=False)
    test_target = torch.load(BASE_DIR / "test_targets.pt", weights_only=False)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    preprocess_data("data/raw", "data/processed")

