"""
HPC-optimized training script for CAFA6 - A800 Optimized
Location: src/training/train_hpc.py
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import GOAnnotationProcessor, ProteinDataset, collate_fn
from models.esm_model import get_model
from utils.losses import get_loss_function
from utils.metrics import CAFAEvaluator, convert_predictions_to_dict, convert_labels_to_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, dataloader, go_processor, config, device, loss_fn=None):
    model.eval()
    all_protein_ids, all_preds, all_labels = [], [], []
    total_loss, n_batches = 0, 0
    
    # 确定AMP dtype
    amp_dtype = torch.bfloat16 if config["training"].get("amp_dtype") == "bf16" else torch.float16
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            with autocast(dtype=amp_dtype):
                outputs = model(batch["sequences"], batch["labels"], loss_fn)
            
            if "loss" in outputs:
                total_loss += outputs["loss"].item()
                n_batches += 1
            
            probs = torch.sigmoid(outputs["logits"]).float()
            all_preds.append(probs.cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            all_protein_ids.extend(batch["protein_ids"])
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    pred_dict = convert_predictions_to_dict(all_protein_ids, all_preds, go_processor.idx2term)
    label_dict = convert_labels_to_dict(all_protein_ids, all_labels, go_processor.idx2term)
    
    evaluator = CAFAEvaluator(go_processor)
    thresholds = config.get("evaluation", {}).get("threshold_range", [0.1, 0.2, 0.3, 0.4, 0.5])
    final_score, per_ont_results = evaluator.compute_final_score(pred_dict, label_dict, thresholds)
    
    model.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "cafa_score": final_score,
        "per_ontology": per_ont_results,
    }


def train(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 确定AMP配置
    use_amp = config["training"].get("use_amp", True)
    amp_dtype = torch.bfloat16 if config["training"].get("amp_dtype") == "bf16" else torch.float16
    print(f"AMP enabled: {use_amp}, dtype: {amp_dtype}")
    
    os.makedirs(config["output"]["save_dir"], exist_ok=True)
    
    # 保存配置
    with open(os.path.join(config["output"]["save_dir"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # wandb
    use_wandb = WANDB_AVAILABLE and config.get("wandb", {}).get("enabled", False) and not args.no_wandb and not args.debug
    if use_wandb:
        wandb.init(project=config["wandb"]["project"], name=config["wandb"].get("name"), config=config)
    
    # 数据
    data_dir = config["data"]["data_dir"]
    train_dir = os.path.join(data_dir, "Train")
    
    print("Initializing GO processor...")
    go_processor = GOAnnotationProcessor(
        obo_path=os.path.join(train_dir, "go-basic.obo"),
        ia_path=os.path.join(data_dir, "IA.tsv"),
        terms_path=os.path.join(train_dir, "train_terms.tsv"),
    )
    go_processor.save(os.path.join(config["output"]["save_dir"], "go_processor.pkl"))
    
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
    
    if args.debug:
        train_dataset.protein_ids = train_dataset.protein_ids[:100]
        val_dataset.protein_ids = val_dataset.protein_ids[:50]
    
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
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # 模型
    print("Creating model...")
    num_labels = len(go_processor.term2idx)
    model = get_model(config["model"], num_labels)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss
    loss_fn = None
    if config["training"].get("use_weighted_loss", False):
        print("Using IA-weighted loss...")
        ia_weights = np.array([go_processor.ia_dict.get(go_processor.idx2term[i], 0.0) for i in range(num_labels)])
        loss_fn = get_loss_function("weighted_bce", ia_weights=ia_weights, device=device)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    num_training_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["training"]["learning_rate"],
        total_steps=num_training_steps,
        pct_start=config["training"].get("warmup_ratio", 0.1),
    )
    
    # AMP Scaler - 只有fp16需要scaler，bf16不需要
    scaler = GradScaler() if (use_amp and amp_dtype == torch.float16) else None
    
    # Resume
    start_epoch, best_score = 0, 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_score = checkpoint.get("best_score", 0)
    
    # Early stopping
    patience = config["training"].get("early_stopping_patience", 5)
    patience_counter = 0
    
    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print(f"Batch size: {config['training']['batch_size']}, Iterations per epoch: {len(train_loader)}")
    
    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"\n{'='*60}\nEpoch {epoch + 1}/{config['training']['epochs']}\n{'='*60}")
        
        model.train()
        epoch_loss, n_batches = 0, 0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)  # 更快的梯度清零
            
            # Forward with AMP
            if use_amp:
                with autocast(dtype=amp_dtype):
                    outputs = model(batch["sequences"], batch["labels"], loss_fn)
                    loss = outputs["loss"]
                
                if scaler:  # fp16
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:  # bf16
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
                    optimizer.step()
            else:
                outputs = model(batch["sequences"], batch["labels"], loss_fn)
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
                optimizer.step()
            
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / n_batches
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s ({epoch_time/n_batches:.2f}s/it)")
        print(f"Average training loss: {avg_loss:.4f}")
        
        # Evaluation
        print("\nEvaluating...")
        metrics = evaluate(model, val_loader, go_processor, config, device, loss_fn)
        
        print(f"Validation Results:")
        print(f"  Loss: {metrics['val_loss']:.4f}")
        print(f"  CAFA Score: {metrics['cafa_score']:.4f}")
        print(f"\nPer-Ontology Results:")
        for ont, results in metrics['per_ontology'].items():
            print(f"  {ont}: F1={results['f1']:.4f}, P={results['precision']:.4f}, R={results['recall']:.4f}")
        
        # Save best
        current_score = metrics["cafa_score"]
        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "config": config,
                "metrics": metrics,
            }, os.path.join(config["output"]["save_dir"], "best_model.pt"))
            print(f"✓ Saved best model with CAFA score: {best_score:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        # Checkpoint
        if (epoch + 1) % config["output"].get("save_every_n_epochs", 1) == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "config": config,
            }, os.path.join(config["output"]["save_dir"], f"checkpoint_epoch_{epoch + 1}.pt"))
    
    print(f"\n{'='*60}\nTraining complete! Best CAFA Score: {best_score:.4f}\n{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, args)
