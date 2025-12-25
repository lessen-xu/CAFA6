"""
Dataset classes for CAFA6 protein function prediction.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import obonet
from tqdm import tqdm
from collections import defaultdict

ASPECT_MAP = {
    'F': 'MFO',
    'P': 'BPO',
    'C': 'CCO',
}


class GOAnnotationProcessor:
    """Process GO annotations and handle ontology relationships."""
    
    def __init__(self, obo_path, ia_path, terms_path=None):
        self.graph = obonet.read_obo(obo_path)
        self.ia_df = pd.read_csv(ia_path, sep="\t", header=None,
                                  names=["go_term", "ia_weight"])
        self.ia_dict = dict(zip(self.ia_df["go_term"], self.ia_df["ia_weight"]))
        
        self.roots = {
            "BPO": "GO:0008150",
            "CCO": "GO:0005575",
            "MFO": "GO:0003674"
        }
        
        self.term2idx = {}
        self.idx2term = {}
        self.term2ont = {}
        
        if terms_path:
            self._build_vocabulary(terms_path)
    
    def _build_vocabulary(self, terms_path):
        terms_df = pd.read_csv(terms_path, sep="\t")
        terms_df.columns = ["protein_id", "go_term", "aspect"]
        terms_df["ontology"] = terms_df["aspect"].map(ASPECT_MAP)
        
        for ont in ["MFO", "BPO", "CCO"]:
            ont_terms = terms_df[terms_df["ontology"] == ont]["go_term"].unique()
            for term in ont_terms:
                if term not in self.term2idx:
                    idx = len(self.term2idx)
                    self.term2idx[term] = idx
                    self.idx2term[idx] = term
                    self.term2ont[term] = ont
        
        print(f"Vocabulary size: {len(self.term2idx)}")
        print(f"  MFO: {sum(1 for t in self.term2ont.values() if t == 'MFO')}")
        print(f"  BPO: {sum(1 for t in self.term2ont.values() if t == 'BPO')}")
        print(f"  CCO: {sum(1 for t in self.term2ont.values() if t == 'CCO')}")
    
    def get_ancestors(self, term):
        ancestors = set()
        if term not in self.graph:
            return ancestors
        queue = [term]
        while queue:
            current = queue.pop(0)
            for parent in self.graph.successors(current):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        return ancestors
    
    def propagate_annotations(self, terms):
        propagated = set(terms)
        for term in terms:
            ancestors = self.get_ancestors(term)
            propagated.update(ancestors)
        return propagated
    
    def get_ia(self, term):
        return self.ia_dict.get(term, 0.0)
    
    def terms_to_vector(self, terms, propagate=True):
        if propagate:
            terms = self.propagate_annotations(terms)
        vector = np.zeros(len(self.term2idx), dtype=np.float32)
        for term in terms:
            if term in self.term2idx:
                vector[self.term2idx[term]] = 1.0
        return vector
    
    def save(self, path):
        state = {
            "term2idx": self.term2idx,
            "idx2term": self.idx2term,
            "term2ont": self.term2ont,
            "ia_dict": self.ia_dict,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def load(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.term2idx = state["term2idx"]
        self.idx2term = state["idx2term"]
        self.term2ont = state["term2ont"]
        self.ia_dict = state["ia_dict"]


class ProteinDataset(Dataset):
    """Dataset for protein sequences and GO annotations."""
    
    def __init__(
        self,
        fasta_path,
        terms_path,
        go_processor,
        max_length=1024,
        split="train",
        val_ratio=0.1,
        seed=42,
    ):
        self.go_processor = go_processor
        self.max_length = max_length
        
        self.sequences = self._load_train_sequences(fasta_path)
        self.annotations = self._load_annotations(terms_path)
        
        self.protein_ids = list(
            set(self.sequences.keys()) & set(self.annotations.keys())
        )
        
        print(f"Proteins with both sequence and annotations: {len(self.protein_ids)}")
        
        np.random.seed(seed)
        np.random.shuffle(self.protein_ids)
        n_val = int(len(self.protein_ids) * val_ratio)
        
        if split == "train":
            self.protein_ids = self.protein_ids[n_val:]
        elif split == "val":
            self.protein_ids = self.protein_ids[:n_val]
        
        print(f"{split.capitalize()} set: {len(self.protein_ids)} proteins")
    
    def _load_train_sequences(self, fasta_path):
        sequences = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            header_parts = record.id.split("|")
            if len(header_parts) >= 2:
                uniprot_id = header_parts[1]
            else:
                uniprot_id = record.id
            sequences[uniprot_id] = str(record.seq)
        return sequences
    
    def _load_annotations(self, terms_path):
        df = pd.read_csv(terms_path, sep="\t")
        df.columns = ["protein_id", "go_term", "aspect"]
        annotations = defaultdict(list)
        for _, row in df.iterrows():
            annotations[row["protein_id"]].append(row["go_term"])
        return dict(annotations)
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[protein_id]
        terms = self.annotations[protein_id]
        
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        labels = self.go_processor.terms_to_vector(terms, propagate=True)
        
        return {
            "protein_id": protein_id,
            "sequence": sequence,
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


class TestDataset(Dataset):
    """Dataset for test sequences (no labels)."""
    
    def __init__(self, fasta_path, max_length=1024):
        self.max_length = max_length
        self.sequences = {}
        self.protein_ids = []
        
        for record in SeqIO.parse(fasta_path, "fasta"):
            protein_id = record.id
            self.sequences[protein_id] = str(record.seq)
            self.protein_ids.append(protein_id)
        
        print(f"Test set: {len(self.protein_ids)} proteins")
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[protein_id]
        
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        return {
            "protein_id": protein_id,
            "sequence": sequence,
        }


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    protein_ids = [item["protein_id"] for item in batch]
    sequences = [item["sequence"] for item in batch]
    
    result = {
        "protein_ids": protein_ids,
        "sequences": sequences,
    }
    
    if "labels" in batch[0]:
        labels = torch.stack([item["labels"] for item in batch])
        result["labels"] = labels
    
    return result
