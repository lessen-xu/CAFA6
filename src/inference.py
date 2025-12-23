"""
Inference script to generate submission file for CAFA6.

Location: src/inference.py

UPDATED:
- Proper GO term propagation to ancestors
- Better threshold handling
- Memory-efficient batch processing
- Support for different model types
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import TestDataset, GOAnnotationProcessor, collate_fn
from models.esm_model import get_model


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
    parser.add_argument("--propagate", action="store_true", default=True,
                        help="Propagate predictions to ancestor terms")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed precision for inference")
    return parser.parse_args()


def propagate_predictions(
    predictions: dict,
    go_processor: GOAnnotationProcessor
) -> dict:
    """
    Propagate predictions to ancestor GO terms.
    
    For each predicted term, assign the same or higher probability
    to all ancestor terms.
    
    Args:
        predictions: {protein_id: {go_term: probability}}
        go_processor: GO processor with ontology graph
    
    Returns:
        Propagated predictions
    """
    propagated = {}
    
    for protein_id, term_probs in tqdm(predictions.items(), desc="Propagating"):
        new_probs = defaultdict(float)
        
        for term, prob in term_probs.items():
            # Include the term itself
            new_probs[term] = max(new_probs[term], prob)
            
            # Propagate to ancestors
            ancestors = go_processor.get_ancestors(term)
            for ancestor in ancestors:
                new_probs[ancestor] = max(new_probs[ancestor], prob)
        
        propagated[protein_id] = dict(new_probs)
    
    return propagated


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load GO processor
    print("Loading GO processor...")
    go_processor = GOAnnotationProcessor.__new__(GOAnnotationProcessor)
    go_processor.load(args.processor)
    
    # Also need to load the ontology graph for propagation
    # Try to find the obo file
    checkpoint_dir = os.path.dirname(args.checkpoint)
    possible_obo_paths = [
        os.path.join(checkpoint_dir, "..", "..", "data", "Train", "go-basic.obo"),
        "data/Train/go-basic.obo",
        "../data/Train/go-basic.obo",
    ]
    
    obo_path = None
    for path in possible_obo_paths:
        if os.path.exists(path):
            obo_path = path
            break
    
    if obo_path and args.propagate:
        import obonet
        go_processor.graph = obonet.read_obo(obo_path)
        print(f"Loaded ontology from {obo_path}")
    elif args.propagate:
        print("Warning: Could not find go-basic.obo, skipping propagation")
        args.propagate = False
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    
    num_labels = len(go_processor.term2idx)
    model = get_model(config["model"], num_labels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best score: {checkpoint.get('best_score', 'unknown')}")
    
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
        num_workers=4,
        collate_fn=collate_fn,
    )
    
    # Run inference
    print("Running inference...")
    predictions = {}  # {protein_id: {go_term: probability}}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            if args.use_amp:
                with autocast():
                    outputs = model(batch["sequences"])
            else:
                outputs = model(batch["sequences"])
            
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            
            for i, protein_id in enumerate(batch["protein_ids"]):
                # Get terms above threshold
                above_threshold = np.where(probs[i] > args.threshold)[0]
                
                if len(above_threshold) > 0:
                    predictions[protein_id] = {
                        go_processor.idx2term[idx]: float(probs[i, idx])
                        for idx in above_threshold
                    }
    
    print(f"Generated predictions for {len(predictions)} proteins")
    
    # Propagate to ancestors
    if args.propagate:
        print("Propagating predictions to ancestor terms...")
        predictions = propagate_predictions(predictions, go_processor)
    
    # Generate submission file
    print("Generating submission file...")
    
    with open(args.output, "w") as f:
        for protein_id in tqdm(sorted(predictions.keys()), desc="Writing"):
            term_probs = predictions[protein_id]
            
            # Sort by probability and limit to max_terms
            sorted_terms = sorted(
                term_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:args.max_terms]
            
            for term, prob in sorted_terms:
                if prob > args.threshold:
                    # Format: protein_id \t GO_term \t probability
                    f.write(f"{protein_id}\t{term}\t{prob:.3f}\n")
    
    # Statistics
    total_predictions = sum(
        len([p for p in probs.values() if p > args.threshold])
        for probs in predictions.values()
    )
    print(f"\nSubmission saved to {args.output}")
    print(f"Total proteins: {len(predictions)}")
    print(f"Total predictions: {total_predictions}")
    print(f"Average predictions per protein: {total_predictions / max(len(predictions), 1):.1f}")


if __name__ == "__main__":
    main()

