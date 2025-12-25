"""
Data inspection script - shows raw file contents.
Usage: python src/data/inspect_data.py
"""

import os
import pandas as pd
from Bio import SeqIO


def inspect_file(filepath, name, num_lines=5):
    print(f"\n{'='*60}")
    print(f"FILE: {name}")
    print(f"Path: {filepath}")
    
    if not os.path.exists(filepath):
        print("FILE NOT FOUND")
        return
    
    print(f"\nFirst {num_lines} lines:")
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            print(f"  {repr(line.strip())}")


def main():
    data_dir = "data"
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")
    
    print("CAFA6 DATA INSPECTION")
    
    inspect_file(os.path.join(train_dir, "train_sequences.fasta"), "train_sequences.fasta")
    inspect_file(os.path.join(train_dir, "train_terms.tsv"), "train_terms.tsv")
    inspect_file(os.path.join(train_dir, "train_taxonomy.tsv"), "train_taxonomy.tsv")
    inspect_file(os.path.join(test_dir, "testsuperset.fasta"), "testsuperset.fasta")
    inspect_file(os.path.join(data_dir, "IA.tsv"), "IA.tsv")
    inspect_file(os.path.join(data_dir, "sample_submission.tsv"), "sample_submission.tsv")
    
    print(f"\n{'='*60}")
    print("INSPECTION COMPLETE")


if __name__ == "__main__":
    main()
