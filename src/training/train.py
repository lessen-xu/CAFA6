"""
Training script for CAFA6 protein function prediction.
FIXED:
- IA.tsv path: data/IA.tsv (not data/Train/IA.tsv)
- Other paths correctly set for Train/ subdirectory

Location: src/training/train.py
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (
    GOAnnotationProcessor,
    ProteinDataset,
    collate_fn,
)
from models.esm_model import ESMForGOPrediction, ESMLightning


def parse_args():
    parser = argparse.ArgumentParser(description="Train CAFA6 model")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to config file")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode with small data subset")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model(batch["sequences"], batch["labels"])
            total_loss += outputs["loss"].item()
            n_batches += 1

            probs = torch.sigmoid(outputs["logits"])
            all_preds.append(probs.cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate metrics at different thresholds
    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        preds_binary = (all_preds > threshold).astype(float)

        # Micro F1
        tp = ((preds_binary == 1) & (all_labels == 1)).sum()
        fp = ((preds_binary == 1) & (all_labels == 0)).sum()
        fn = ((preds_binary == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    metrics = {
        "val_loss": total_loss / n_batches,
        "val_f1": best_f1,
        "best_threshold": best_threshold,
    }

    model.train()
    return metrics


def train(config, debug=False, use_wandb=True):
    """Main training function."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    if use_wandb and not debug:
        try:
            import wandb
            wandb.init(project="cafa6", config=config)
        except ImportError:
            print("wandb not installed, skipping logging")
            use_wandb = False

    # Data paths - CORRECT STRUCTURE
    # data/
    # ├── IA.tsv                    <- directly under data/
    # ├── sample_submission.tsv     <- directly under data/
    # ├── Test/
    # │   ├── testsuperset-taxon-list.tsv
    # │   └── testsuperset.fasta
    # └── Train/
    #     ├── go-basic.obo
    #     ├── train_sequences.fasta
    #     ├── train_taxonomy.tsv
    #     └── train_terms.tsv
    
    data_dir = config["data"]["data_dir"]
    train_dir = os.path.join(data_dir, "Train")

    # Initialize GO processor
    print("Initializing GO processor...")
    go_processor = GOAnnotationProcessor(
        obo_path=os.path.join(train_dir, "go-basic.obo"),
        ia_path=os.path.join(data_dir, "IA.tsv"),  # IA.tsv is directly under data/
        terms_path=os.path.join(train_dir, "train_terms.tsv"),
    )

    # Save processor
    processor_path = os.path.join(config["output"]["save_dir"], "go_processor.pkl")
    os.makedirs(config["output"]["save_dir"], exist_ok=True)
    go_processor.save(processor_path)

    # Create datasets
    print("Creating datasets...")
    train_dataset = ProteinDataset(
        fasta_path=os.path.join(train_dir, "train_sequences.fasta"),
        terms_path=os.path.join(train_dir, "train_terms.tsv"),
        go_processor=go_processor,
        max_length=config["data"]["max_length"],
        split="train",
        val_ratio=config["data"]["val_ratio"],
    )

    val_dataset = ProteinDataset(
        fasta_path=os.path.join(train_dir, "train_sequences.fasta"),
        terms_path=os.path.join(train_dir, "train_terms.tsv"),
        go_processor=go_processor,
        max_length=config["data"]["max_length"],
        split="val",
        val_ratio=config["data"]["val_ratio"],
    )

    # Debug mode: use small subset
    if debug:
        train_dataset.protein_ids = train_dataset.protein_ids[:100]
        val_dataset.protein_ids = val_dataset.protein_ids[:50]
        print(f"Debug mode: using {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    print("Creating model...")
    num_labels = len(go_processor.term2idx)
    print(f"Number of labels: {num_labels}")

    if config["model"]["type"] == "esm_light":
        model = ESMLightning(
            model_name=config["model"]["name"],
            num_labels=num_labels,
            dropout=config["model"]["dropout"],
        )
    else:
        model = ESMForGOPrediction(
            model_name=config["model"]["name"],
            num_labels=num_labels,
            dropout=config["model"]["dropout"],
            freeze_backbone=config["model"].get("freeze_backbone", False),
        )

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    num_training_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=config["training"]["learning_rate"] * 0.01,
    )

    # Training loop
    best_f1 = 0
    global_step = 0

    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        model.train()
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            optimizer.zero_grad()

            outputs = model(batch["sequences"], batch["labels"])
            loss = outputs["loss"]

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["max_grad_norm"]
            )

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

            # Log to wandb
            if use_wandb and not debug and global_step % 100 == 0:
                import wandb
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                })

        avg_loss = epoch_loss / n_batches
        print(f"Average training loss: {avg_loss:.4f}")

        # Evaluation
        print("Evaluating...")
        metrics = evaluate(model, val_loader, device)
        print(f"Val loss: {metrics['val_loss']:.4f}, Val F1: {metrics['val_f1']:.4f}")

        if use_wandb and not debug:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": metrics["val_loss"],
                "val_f1": metrics["val_f1"],
            })

        # Save best model
        if metrics["val_f1"] > best_f1:
            best_f1 = metrics["val_f1"]
            save_path = os.path.join(
                config["output"]["save_dir"],
                "best_model.pt"
            )
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": config,
            }, save_path)
            print(f"Saved best model with F1: {best_f1:.4f}")

    print(f"\nTraining complete! Best F1: {best_f1:.4f}")

    if use_wandb and not debug:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, debug=args.debug, use_wandb=not args.no_wandb)

