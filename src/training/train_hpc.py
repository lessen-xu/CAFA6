"""
HPC-optimized training script for CAFA6 protein function prediction.

Location: src/training/train_hpc.py

Features:
- Mixed precision training (AMP)
- Gradient accumulation
- Gradient checkpointing
- Weighted loss with IA
- CAFA-specific evaluation metrics
- Early stopping
- Better logging and checkpointing
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
from datetime import datetime

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available, logging to console only")

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import (
    GOAnnotationProcessor,
    ProteinDataset,
    collate_fn,
)
from models.esm_model import get_model
from utils.losses import get_loss_function, compute_pos_weight
from utils.metrics import CAFAEvaluator, convert_predictions_to_dict, convert_labels_to_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train CAFA6 model on HPC")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode with small data subset")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_lr_scheduler(optimizer, config, num_training_steps):
    """Get learning rate scheduler."""
    warmup_steps = int(num_training_steps * config["training"].get("warmup_ratio", 0.1))
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["training"]["learning_rate"],
        total_steps=num_training_steps,
        pct_start=config["training"].get("warmup_ratio", 0.1),
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0,
    )
    
    return scheduler


def evaluate(model, dataloader, go_processor, config, device, loss_fn=None):
    """Evaluate model with CAFA metrics."""
    model.eval()
    
    all_protein_ids = []
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Forward pass
            if config["training"].get("use_amp", False):
                with autocast():
                    outputs = model(batch["sequences"], batch["labels"], loss_fn)
            else:
                outputs = model(batch["sequences"], batch["labels"], loss_fn)
            
            if "loss" in outputs:
                total_loss += outputs["loss"].item()
                n_batches += 1
            
            probs = torch.sigmoid(outputs["logits"])
            all_preds.append(probs.cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            all_protein_ids.extend(batch["protein_ids"])
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Convert to dictionary format for CAFA evaluation
    pred_dict = convert_predictions_to_dict(
        all_protein_ids, all_preds, go_processor.idx2term
    )
    label_dict = convert_labels_to_dict(
        all_protein_ids, all_labels, go_processor.idx2term
    )
    
    # CAFA evaluation
    evaluator = CAFAEvaluator(go_processor)
    thresholds = config.get("evaluation", {}).get(
        "threshold_range",
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    )
    
    final_score, per_ont_results = evaluator.compute_final_score(
        pred_dict, label_dict, thresholds
    )
    
    # Also compute simple metrics for quick comparison
    best_simple_f1 = 0
    best_threshold = 0.5
    for threshold in thresholds:
        preds_binary = (all_preds > threshold).astype(float)
        tp = ((preds_binary == 1) & (all_labels == 1)).sum()
        fp = ((preds_binary == 1) & (all_labels == 0)).sum()
        fn = ((preds_binary == 0) & (all_labels == 1)).sum()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        if f1 > best_simple_f1:
            best_simple_f1 = f1
            best_threshold = threshold
    
    metrics = {
        "val_loss": total_loss / max(n_batches, 1),
        "cafa_score": final_score,
        "simple_f1": best_simple_f1,
        "best_threshold": best_threshold,
        "per_ontology": per_ont_results,
    }
    
    model.train()
    return metrics


def train(config, args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    os.makedirs(config["output"]["save_dir"], exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(config["output"]["save_dir"], "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    
    # Initialize wandb
    use_wandb = (
        WANDB_AVAILABLE and 
        config.get("wandb", {}).get("enabled", False) and 
        not args.no_wandb and 
        not args.debug
    )
    
    if use_wandb:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"].get("name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config=config
        )
    
    # Data paths
    data_dir = config["data"]["data_dir"]
    train_dir = os.path.join(data_dir, "Train")
    
    # Initialize GO processor
    print("Initializing GO processor...")
    go_processor = GOAnnotationProcessor(
        obo_path=os.path.join(train_dir, "go-basic.obo"),
        ia_path=os.path.join(data_dir, "IA.tsv"),
        terms_path=os.path.join(train_dir, "train_terms.tsv"),
    )
    
    # Save processor
    processor_path = os.path.join(config["output"]["save_dir"], "go_processor.pkl")
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
    
    # Debug mode
    if args.debug:
        train_dataset.protein_ids = train_dataset.protein_ids[:100]
        val_dataset.protein_ids = val_dataset.protein_ids[:50]
        print("DEBUG MODE: Using small subset")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"] * 2,  # Larger batch for eval
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Create model
    print("Creating model...")
    num_labels = len(go_processor.term2idx)
    model = get_model(config["model"], num_labels)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_fn = None
    if config["training"].get("use_weighted_loss", False):
        print("Using IA-weighted loss...")
        # Get IA weights for each term in vocabulary
        ia_weights = np.array([
            go_processor.ia_dict.get(go_processor.idx2term[i], 0.0)
            for i in range(num_labels)
        ])
        loss_fn = get_loss_function(
            "weighted_bce",
            ia_weights=ia_weights,
            device=device
        )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Scheduler
    gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_training_steps = (
        len(train_loader) // gradient_accumulation_steps * config["training"]["epochs"]
    )
    scheduler = get_lr_scheduler(optimizer, config, num_training_steps)
    
    # Mixed precision scaler
    scaler = GradScaler() if config["training"].get("use_amp", False) else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_score = 0
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_score = checkpoint.get("best_score", 0)
        print(f"Resumed from epoch {start_epoch}, best score: {best_score:.4f}")
    
    # Early stopping
    patience = config["training"].get("early_stopping_patience", 5)
    patience_counter = 0
    
    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    global_step = 0
    
    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        model.train()
        epoch_loss = 0
        n_batches = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Training")
        start_time = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            if config["training"].get("use_amp", False):
                with autocast():
                    outputs = model(batch["sequences"], batch["labels"], loss_fn)
                    loss = outputs["loss"] / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(batch["sequences"], batch["labels"], loss_fn)
                loss = outputs["loss"] / gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if config["training"].get("use_amp", False):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config["training"]["max_grad_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config["training"]["max_grad_norm"]
                    )
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += outputs["loss"].item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{outputs['loss'].item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if use_wandb and global_step % config["output"].get("log_every_n_steps", 50) == 0:
                wandb.log({
                    "train_loss": outputs["loss"].item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                })
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / n_batches
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"Average training loss: {avg_loss:.4f}")
        
        # Evaluation
        eval_every = config.get("evaluation", {}).get("eval_every_n_epochs", 1)
        if (epoch + 1) % eval_every == 0:
            print("\nEvaluating...")
            metrics = evaluate(model, val_loader, go_processor, config, device, loss_fn)
            
            print(f"Validation Results:")
            print(f"  Loss: {metrics['val_loss']:.4f}")
            print(f"  CAFA Score: {metrics['cafa_score']:.4f}")
            print(f"  Simple F1: {metrics['simple_f1']:.4f}")
            print(f"  Best Threshold: {metrics['best_threshold']:.2f}")
            print(f"\nPer-Ontology Results:")
            for ont, results in metrics['per_ontology'].items():
                print(f"  {ont}: F1={results['f1']:.4f}, P={results['precision']:.4f}, "
                      f"R={results['recall']:.4f}, T={results['threshold']:.2f}")
            
            if use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "val_loss": metrics["val_loss"],
                    "cafa_score": metrics["cafa_score"],
                    "simple_f1": metrics["simple_f1"],
                    "best_threshold": metrics["best_threshold"],
                }
                for ont, results in metrics['per_ontology'].items():
                    log_dict[f"{ont}_f1"] = results["f1"]
                    log_dict[f"{ont}_precision"] = results["precision"]
                    log_dict[f"{ont}_recall"] = results["recall"]
                wandb.log(log_dict)
            
            # Check for improvement
            current_score = metrics["cafa_score"]
            if current_score > best_score:
                best_score = current_score
                patience_counter = 0
                
                # Save best model
                save_path = os.path.join(config["output"]["save_dir"], "best_model.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_score": best_score,
                    "config": config,
                    "metrics": metrics,
                }, save_path)
                print(f"✓ Saved best model with CAFA score: {best_score:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Periodic checkpoint
        save_every = config["output"].get("save_every_n_epochs", 2)
        if (epoch + 1) % save_every == 0:
            save_path = os.path.join(
                config["output"]["save_dir"],
                f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_score": best_score,
                "config": config,
            }, save_path)
            print(f"Saved checkpoint: {save_path}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best CAFA Score: {best_score:.4f}")
    print(f"{'='*60}")
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, args)

