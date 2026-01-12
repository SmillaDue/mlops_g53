from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mlops_project.data import MyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../../configs", config_name="default_config.yaml", version_base=None)
def train(config: DictConfig) -> None:
    """Train model"""
    print("Training day and night;)")

    # Extract hyperparameters from config
    batch_size = config.batch_size
    epochs = config.epochs

    print(f"lr={config.optimizer.lr}, {batch_size=}, {epochs=}")

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    train_set, _ = MyDataset("data/raw")  # NEEDS TO BE CHANGED WHEN DATA IS READY
    model = hydra.utils.instantiate(config.model).to(DEVICE)

    # Create a DataLoader to batch and shuffle the training data
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize optimizer from config
    optimizer_class = getattr(torch.optim, config.optimizer.name, torch.optim.Adam)
    optimizer = optimizer_class(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    for epoch in range(epochs):
        model.train()
        preds = []
        targets = []
        # Iterate over batches in the training data
        for i, (img, target) in enumerate(train_dataloader):
            # Move data to the appropriate device
            img, target = img.to(DEVICE), target.to(DEVICE)
            # Clear gradients from previous iteration
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            # Backward pass: compute gradients
            loss.backward()
            # Update model parameters
            optimizer.step()

            # Calculate and record accuracy for this batch
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
                # wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

    print("Training complete")

    # Save the trained model's parameters to models folder
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")

    # getting the model performance metrics
    preds = torch.cat(preds, 0)
    targets = torch.cat(targets, 0)
    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    # printing model performane metrics for now, but need to log them in wandb later
    print("\n" + "=" * 50)
    print("FINAL TRAINING METRICS")
    print("=" * 50)
    print(f"Accuracy:  {final_accuracy:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall:    {final_recall:.4f}")
    print(f"F1 Score:  {final_f1:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    train()
