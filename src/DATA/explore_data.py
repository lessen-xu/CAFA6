"""
Data exploration script for CAFA6 competition.
Usage: python src/data/explore_data.py
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from Bio import SeqIO
import obonet
import matplotlib.pyplot as plt

ASPECT_MAP = {'F': 'MFO', 'P': 'BPO', 'C': 'CCO'}


def load_train_sequences(fasta_path):
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        header_parts = record.id.split("|")
        if len(header_parts) >= 2:
            uniprot_id = header_parts[1]
        else:
            uniprot_id = record.id
        sequences[uniprot_id] = str(record.seq)
    return sequences


def load_test_sequences(fasta_path):
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def load_terms(terms_path):
    df = pd.read_csv(terms_path, sep="\t")
    df.columns = ["protein_id", "go_term", "aspect"]
    df["ontology"] = df["aspect"].map(ASPECT_MAP)
    return df


def load_ia(ia_path):
    df = pd.read_csv(ia_path, sep="\t", header=None, names=["go_term", "ia_weight"])
    return df


def main():
    data_dir = "data"
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")
    
    print("Loading data...")
    train_sequences = load_train_sequences(os.path.join(train_dir, "train_sequences.fasta"))
    test_sequences = load_test_sequences(os.path.join(test_dir, "testsuperset.fasta"))
    terms_df = load_terms(os.path.join(train_dir, "train_terms.tsv"))
    ia_df = load_ia(os.path.join(data_dir, "IA.tsv"))
    
    print(f"\nTrain sequences: {len(train_sequences):,}")
    print(f"Test sequences: {len(test_sequences):,}")
    print(f"Total annotations: {len(terms_df):,}")
    print(f"Unique proteins: {terms_df['protein_id'].nunique():,}")
    print(f"Unique GO terms: {terms_df['go_term'].nunique():,}")
    
    print("\nPer ontology:")
    for ont in ["MFO", "BPO", "CCO"]:
        ont_df = terms_df[terms_df["ontology"] == ont]
        print(f"  {ont}: {len(ont_df):,} annotations, {ont_df['go_term'].nunique():,} terms")
    
    print(f"\nIA weights: {len(ia_df):,} terms")
    print(f"IA range: [{ia_df['ia_weight'].min():.4f}, {ia_df['ia_weight'].max():.4f}]")


if __name__ == "__main__":
    main()
