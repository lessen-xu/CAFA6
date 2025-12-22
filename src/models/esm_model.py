"""
ESM-2 based model for protein function prediction.
ESM-2 is a state-of-the-art protein language model from Meta AI.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer


class ESMForGOPrediction(nn.Module):
    """
    ESM-2 model with classification head for GO term prediction.
    
    ESM-2 comes in different sizes:
    - esm2_t6_8M_UR50D: 8M parameters (fast, for testing)
    - esm2_t12_35M_UR50D: 35M parameters
    - esm2_t30_150M_UR50D: 150M parameters
    - esm2_t33_650M_UR50D: 650M parameters (recommended for 4060)
    - esm2_t36_3B_UR50D: 3B parameters (may need gradient checkpointing)
    """
    
    def __init__(
        self,
        model_name="facebook/esm2_t33_650M_UR50D",
        num_labels=None,
        dropout=0.1,
        freeze_backbone=False,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load ESM-2 model and tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Get hidden size from config
        self.hidden_size = self.esm.config.hidden_size
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )
    
    def forward(self, sequences, labels=None):
        """
        Forward pass.
        
        Args:
            sequences: List of protein sequences (amino acid strings)
            labels: Optional tensor of shape (batch_size, num_labels)
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Tokenize sequences
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get ESM embeddings
        outputs = self.esm(**inputs)
        
        # Use mean pooling over sequence length
        # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = outputs.last_hidden_state * attention_mask
        pooled = embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            result["loss"] = loss_fn(logits, labels)
        
        return result
    
    def predict(self, sequences, threshold=0.5):
        """
        Make predictions for sequences.
        
        Args:
            sequences: List of protein sequences
            threshold: Probability threshold for predictions
        
        Returns:
            Numpy array of shape (batch_size, num_labels)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(sequences)
            probs = torch.sigmoid(outputs["logits"])
        return probs.cpu().numpy()


class ESMLightning(nn.Module):
    """
    Lighter version using smaller ESM-2 model for faster experimentation.
    Good for initial testing on 4060.
    """
    
    def __init__(
        self,
        model_name="facebook/esm2_t12_35M_UR50D",  # 35M model
        num_labels=None,
        dropout=0.2,
    ):
        super().__init__()
        
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)
        self.hidden_size = self.esm.config.hidden_size
        
        # Simpler classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )
    
    def forward(self, sequences, labels=None):
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Shorter for speed
        )
        
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.esm(**inputs)
        
        # Mean pooling
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = outputs.last_hidden_state * attention_mask
        pooled = embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        
        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            result["loss"] = loss_fn(logits, labels)
        
        return result
