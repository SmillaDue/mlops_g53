# mlops_project

## Framework

This project is built using **MONAI**, a PyTorch-based open-source framework designed specifically for medical and healthcare imaging. MONAI provides standardized components for data ingestion, preprocessing, model architectures, training workflows, and visualization. These abstractions help ensure that the pipeline is both **reproducible and maintainable**, which is essential for medical imaging research and deployment.

## Project Structure

The project is organized into modular components to support training, evaluation, deployment, and visualization:

- `data.py` — Data loading and preprocessing logic  
- `dataset.py` — Dataset definitions and dataset-specific utilities  
- `model.py` — Model architecture and initialization  
- `train.py` — Training loop and experiment configuration  
- `evaluate.py` — Model evaluation and metrics computation  
- `api.py` — Inference API for serving the trained model  
- `visualize.py` — Visualization utilities for data and predictions  
- `utils.py` — Shared helper functions and utilities  

## Data

The project uses the **Brain Tumor MRI Dataset** from Kaggle, which contains a total of **7,023 two-dimensional (2D) brain MRI images** distributed across four diagnostic categories:

* Glioma
* Meningioma
* Pituitary tumor
* No tumor

The dataset is partitioned into **training, validation, and test splits** to support model development, hyperparameter tuning, and unbiased performance evaluation.

Source:
[https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

::: mlops_project.dataset
    options:
      show_root_heading: true
      heading_level: 3

## Data Processing

::: mlops_project.data
    options:
      show_root_heading: true
      heading_level: 3

## Model Architecture

As a baseline, the project employs **DenseNet121**, initialized with **ImageNet pre-trained weights** and adapted for **four-class classification**. DenseNet121 is a widely used convolutional neural network architecture in medical imaging due to its strong feature reuse, stable gradient flow, and favorable trade-off between performance and computational efficiency.

::: mlops_project.model.DenseNetModel
    options:
      show_root_heading: true
      heading_level: 3

## Training

Model performance is assessed using standard classification metrics, including:

* **Accuracy**
* **F1 score**

::: mlops_project.train
    options:
      show_root_heading: true
      heading_level: 3


## Model Evaluation

This module provides functionality for evaluating trained models on the test dataset, including metric computation and performance reporting.

::: mlops_project.evaluate
    options:
      show_root_heading: true
      heading_level: 3


## Inference API

The project includes a lightweight API for serving trained models and performing inference on new MRI images.

::: mlops_project.api
    options:
      show_root_heading: true
      heading_level: 3

## Visualization

This module provides utilities for visualizing input data, model predictions, and training performance.

::: mlops_project.visualize
    options:
      show_root_heading: true
      heading_level: 3
