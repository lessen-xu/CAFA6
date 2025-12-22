"""
Inference script to generate submission file for CAFA6.
FIXED:
- Write predictions on-the-fly instead of storing all in memory
- torch.load with weights_only=False for PyTorch 2.6+

Location: src/inference.py
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import TestDataset, GOAnnotationProcessor, collate_fn
from models.esm_model import ESMForGOPrediction, ESMLightning


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--processor", type=str, required=True,
                        help="Path to GO processor pickle")
    parser.add_argument("--test_fasta", type=str, required=True,
                        help="Path to test FASTA file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output submission file")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Minimum probability threshold")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_terms", type=int, default=1500,
                        help="Maximum terms per protein")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load GO processor
    print("Loading GO processor...")
    go_processor = GOAnnotationProcessor.__new__(GOAnnotationProcessor)
    go_processor.load(args.processor)
    print(f"Loaded {len(go_processor.term2idx)} terms")

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    num_labels = len(go_processor.term2idx)

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
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} with F1 {checkpoint['best_f1']:.4f}")

    # Create test dataset
    print("Loading test data...")
    test_dataset = TestDataset(
        fasta_path=args.test_fasta,
        max_length=config["data"]["max_length"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Reduce memory usage
        collate_fn=collate_fn,
    )

    # Run inference and write on-the-fly
    print("Running inference...")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    total_lines = 0
    total_proteins = 0

    with open(args.output, "w") as f:
        with torch.no_grad():
            for batch in tqdm(test_loader):
                outputs = model(batch["sequences"])
                probs = torch.sigmoid(outputs["logits"]).cpu().numpy()

                # Process each protein in batch immediately
                for i, protein_id in enumerate(batch["protein_ids"]):
                    protein_probs = probs[i]

                    # Get terms above threshold
                    above_threshold = np.where(protein_probs > args.threshold)[0]

                    # Sort by probability and limit to max_terms
                    sorted_indices = above_threshold[
                        np.argsort(protein_probs[above_threshold])[::-1]
                    ][:args.max_terms]

                    for idx in sorted_indices:
                        term = go_processor.idx2term[idx]
                        prob = protein_probs[idx]
                        f.write(f"{protein_id}\t{term}\t{prob:.3f}\n")
                        total_lines += 1

                    total_proteins += 1

                # Clear memory periodically
                del probs
                if total_proteins % 10000 == 0:
                    torch.cuda.empty_cache()

    print(f"Submission saved to {args.output}")
    print(f"Total proteins: {total_proteins:,}")
    print(f"Total lines: {total_lines:,}")


if __name__ == "__main__":
    main()

