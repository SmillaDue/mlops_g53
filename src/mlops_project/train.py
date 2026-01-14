from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mlops_project.data import brain_tumor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../../configs", config_name="default_config.yaml", version_base=None)
def train(config: DictConfig) -> None:
    """Train a neural network model on brain tumor classification data.
    
    Loads data, splits into train/validation sets, trains the model with the specified
    hyperparameters, and saves the trained model. Optionally logs metrics to Weights & Biases.
    
    Args:
        config: Hydra configuration containing:
            - batch_size: Number of samples per batch
            - epochs: Number of training epochs
            - seed: Random seed for reproducibility
            - optimizer: Optimizer configuration (name, lr, weight_decay)
            - model: Model architecture configuration
            - wandb: Optional W&B logging configuration (enabled, project, entity, etc.)
    
    Returns:
        None. Saves model to models/model.pth and logs metrics to W&B if enabled.
    """
    print("Training day and night;)")
    print(f"Using device: {DEVICE}")

    # Extract hyperparameters from config
    batch_size = config.batch_size
    epochs = config.epochs

    print(f"lr={config.optimizer.lr}, {batch_size=}, {epochs=}")

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    use_wandb = bool(getattr(config, "wandb", None) and config.wandb.get("enabled", False))
    log_every = int(config.wandb.get("log_every", 50)) if getattr(config, "wandb", None) else 50

    run = None
    if use_wandb:


        # Hydra configs are not plain dicts; convert safely
        cfg_dict = OmegaConf.to_container(config, resolve=True)

        run = wandb.init(
            project=config.wandb.get("project", "brainy_mlops"),
            entity=config.wandb.get("entity", None),  
            name=config.wandb.get("name", None),      
            tags=config.wandb.get("tags", None),      
            config=cfg_dict,
        )
        
        wandb.config.update({"device": str(DEVICE)}, allow_val_change=True)

    train_set, _ = brain_tumor()
    
    # Split into 90% train, 10% validation
    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )
    
    model = hydra.utils.instantiate(config.model).to(DEVICE)

    # Create DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize optimizer from config
    optimizer_class = getattr(torch.optim, config.optimizer.name, torch.optim.Adam)
    optimizer = optimizer_class(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    global_step = 0
    all_preds = []
    all_targets = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
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

            batch_loss = float(loss.item())
            batch_acc = float((y_pred.argmax(dim=1) == target).float().mean().item())

            running_loss += batch_loss
            running_acc += batch_acc
            num_batches += 1

            # Calculate and record accuracy for this batch
            if epoch == epochs - 1:
                all_preds.append(y_pred.detach().cpu())
                all_targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
                # wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            
            if use_wandb and (global_step % log_every == 0):
                wandb.log(
                    {
                        "train/loss_step": batch_loss,
                        "train/acc_step": batch_acc,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            global_step += 1

        epoch_loss = running_loss / max(1, num_batches)
        epoch_acc = running_acc / max(1, num_batches)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for img, target in val_dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                y_pred = model(img)
                val_loss += loss_fn(y_pred, target).item()
                val_acc += (y_pred.argmax(dim=1) == target).float().mean().item()
                val_batches += 1
        val_loss /= max(1, val_batches)
        val_acc /= max(1, val_batches)
        
        print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if use_wandb:
            wandb.log(
                {"train/loss_epoch": epoch_loss, "train/acc_epoch": epoch_acc,
                 "val/loss_epoch": val_loss, "val/acc_epoch": val_acc},
                step=global_step,
            )

    print("Training complete")

    # Save the trained model's parameters to models folder
    Path("models").mkdir(parents=True, exist_ok=True)
    model_path = Path("models/model.pth")
    torch.save(model.state_dict(), model_path)

    # getting the model performance metrics
    preds = torch.cat(all_preds, 0)
    targets = torch.cat(all_targets, 0)
    pred_labels = preds.argmax(dim=1)

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted", zero_division=0)
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted", zero_division=0)
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted", zero_division=0)

    if use_wandb:
        wandb.log(
            {
                "train/accuracy_final": final_accuracy,
                "train/precision_final": final_precision,
                "train/recall_final": final_recall,
                "train/f1_final": final_f1,
            },
            step=global_step,
        )

        if config.wandb.get("log_model", True):
            artifact = wandb.Artifact(
                name="model",
                type="checkpoint",
                metadata={"framework": "pytorch"},
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    train()
