"""
Quick test script to verify the code works before running on HPC.

Location: scripts/quick_test.py

Usage:
    python scripts/quick_test.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    
    try:
        from data.dataset import GOAnnotationProcessor, ProteinDataset, TestDataset
        print("  ✓ data.dataset")
    except Exception as e:
        print(f"  ✗ data.dataset: {e}")
        return False
    
    try:
        from models.esm_model import ESMForGOPrediction, ESMLightning, get_model
        print("  ✓ models.esm_model")
    except Exception as e:
        print(f"  ✗ models.esm_model: {e}")
        return False
    
    try:
        from utils.metrics import CAFAEvaluator, evaluate_batch
        print("  ✓ utils.metrics")
    except Exception as e:
        print(f"  ✗ utils.metrics: {e}")
        return False
    
    try:
        from utils.losses import get_loss_function, WeightedBCELoss
        print("  ✓ utils.losses")
    except Exception as e:
        print(f"  ✗ utils.losses: {e}")
        return False
    
    return True


def test_data_loading():
    """Test data loading."""
    print("\nTesting data loading...")
    
    data_dir = "data"
    train_dir = os.path.join(data_dir, "Train")
    
    # Check files exist
    required_files = [
        os.path.join(train_dir, "train_sequences.fasta"),
        os.path.join(train_dir, "train_terms.tsv"),
        os.path.join(train_dir, "go-basic.obo"),
        os.path.join(data_dir, "IA.tsv"),
        os.path.join(data_dir, "Test", "testsuperset.fasta"),
    ]
    
    for f in required_files:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} not found")
            return False
    
    # Try loading
    from data.dataset import GOAnnotationProcessor
    
    try:
        go_processor = GOAnnotationProcessor(
            obo_path=os.path.join(train_dir, "go-basic.obo"),
            ia_path=os.path.join(data_dir, "IA.tsv"),
            terms_path=os.path.join(train_dir, "train_terms.tsv"),
        )
        print(f"  ✓ GO processor loaded: {len(go_processor.term2idx)} terms")
    except Exception as e:
        print(f"  ✗ GO processor failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    import torch
    from models.esm_model import ESMLightning
    
    try:
        model = ESMLightning(
            model_name="facebook/esm2_t6_8M_UR50D",  # Use smallest model for testing
            num_labels=100,
            dropout=0.1,
        )
        print("  ✓ ESMLightning created")
        
        # Test forward pass
        test_sequences = ["MKTAYIAKQRQISFVKSHFSRQ"]
        outputs = model(test_sequences)
        print(f"  ✓ Forward pass: logits shape {outputs['logits'].shape}")
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False
    
    return True


def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics...")
    
    import numpy as np
    from utils.metrics import CAFAEvaluator
    
    # Create mock data
    class MockProcessor:
        def __init__(self):
            self.term2idx = {"GO:0001": 0, "GO:0002": 1, "GO:0003": 2}
            self.idx2term = {0: "GO:0001", 1: "GO:0002", 2: "GO:0003"}
            self.term2ont = {"GO:0001": "MFO", "GO:0002": "BPO", "GO:0003": "CCO"}
            self.ia_dict = {"GO:0001": 1.0, "GO:0002": 2.0, "GO:0003": 1.5}
    
    try:
        processor = MockProcessor()
        evaluator = CAFAEvaluator(processor)
        
        predictions = {
            "P1": {"GO:0001": 0.9, "GO:0002": 0.3},
            "P2": {"GO:0002": 0.8, "GO:0003": 0.6},
        }
        ground_truth = {
            "P1": {"GO:0001", "GO:0002"},
            "P2": {"GO:0002"},
        }
        
        score, results = evaluator.compute_final_score(predictions, ground_truth)
        print(f"  ✓ Metrics calculated: CAFA score = {score:.4f}")
        
    except Exception as e:
        print(f"  ✗ Metrics failed: {e}")
        return False
    
    return True


def test_losses():
    """Test loss functions."""
    print("\nTesting loss functions...")
    
    import torch
    import numpy as np
    from utils.losses import get_loss_function
    
    try:
        # Test weighted BCE
        ia_weights = np.array([1.0, 2.0, 1.5])
        loss_fn = get_loss_function("weighted_bce", ia_weights=ia_weights)
        
        logits = torch.randn(4, 3)
        labels = torch.randint(0, 2, (4, 3)).float()
        
        loss = loss_fn(logits, labels)
        print(f"  ✓ Weighted BCE loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"  ✗ Loss function failed: {e}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("CAFA6 Quick Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Metrics", test_metrics()))
    results.append(("Losses", test_losses()))
    results.append(("Model Creation", test_model_creation()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Ready for HPC training.")
    else:
        print("Some tests failed. Please fix issues before running on HPC.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

