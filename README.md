# mlops_project

A short description of the project.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

Project Description

Goal

This is the project description for group 53 in the 02476 Machine Learning Operations course on DTU. The goal is to design, implement, and evaluate a reproducible medical image classification pipeline, demonstrating practical understanding of machine learning workflows, model training, evaluation, and operational considerations such as experiment tracking and explainability.

Framework

We use the MONAI framework, a PyTorch-based open-source library specialized for healthcare imaging. MONAI provides standardized components for data loading, preprocessing, model architectures, and visualization, making it suitable for building reliable and reproducible medical imaging pipelines.

Data

The project uses the Brain Tumor MRI Dataset, consisting of 7,023 2D brain MRI images categorized into four classes: glioma, meningioma, pituitary tumor, and no tumor.
The dataset will be split into training, validation, and test sets. 

Models
As a baseline model, we use DenseNet121, initialized with ImageNet pre-trained weights and adapted for four-class classification. DenseNet121 is a well-established architecture in medical imaging tasks and provides a strong balance between performance and computational efficiency. For evaluation basic things like accuracy and F1 scores will be calculated.
