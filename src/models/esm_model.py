"""
ESM-2 based models for protein function prediction.

Location: src/models/esm_model.py

Optimized for A800 80GB with BF16 support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer
from typing import Optional, Dict, List


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence length."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attention(hidden_states).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled


class ESMForGOPrediction(nn.Module):
    """
    ESM-2 model for GO term prediction.
    
    Sizes:
    - esm2_t6_8M_UR50D: 8M (testing)
    - esm2_t12_35M_UR50D: 35M
    - esm2_t30_150M_UR50D: 150M
    - esm2_t33_650M_UR50D: 650M (recommended)
    - esm2_t36_3B_UR50D: 3B
    """
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        num_labels: int = None,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = False,
        pooling_type: str = "mean",
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.pooling_type = pooling_type
        
        # Load model
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)
        self.hidden_size = self.esm.config.hidden_size
        
        # Gradient checkpointing
        if use_gradient_checkpointing:
            self.esm.gradient_checkpointing_enable()
            print(f"Gradient checkpointing enabled for {model_name}")
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False
            print(f"Backbone frozen")
        
        # Pooling
        if pooling_type == "attention":
            self.pooler = AttentionPooling(self.hidden_size)
        else:
            self.pooler = None
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling_type == "attention":
            return self.pooler(hidden_states, attention_mask)
        else:  # mean
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    
    def forward(
        self,
        sequences: List[str],
        labels: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Tokenize
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # ESM forward (不手动转dtype，让autocast处理)
        outputs = self.esm(**inputs)
        
        # Pool
        pooled = self.pool(outputs.last_hidden_state, inputs["attention_mask"])
        
        # Classify
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        
        # Loss
        if labels is not None:
            labels = labels.to(device)
            if loss_fn is not None:
                result["loss"] = loss_fn(logits, labels)
            else:
                result["loss"] = nn.BCEWithLogitsLoss()(logits, labels)
        
        return result


class ESMLightning(nn.Module):
    """Lighter version for testing."""
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        num_labels: int = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)
        self.hidden_size = self.esm.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )
    
    def forward(
        self,
        sequences: List[str],
        labels: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.esm(**inputs)
        
        # Mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        
        if labels is not None:
            labels = labels.to(device)
            if loss_fn is not None:
                result["loss"] = loss_fn(logits, labels)
            else:
                result["loss"] = nn.BCEWithLogitsLoss()(logits, labels)
        
        return result


def get_model(config: dict, num_labels: int) -> nn.Module:
    """Factory function."""
    model_type = config.get("type", "esm")
    
    if model_type == "esm_light":
        return ESMLightning(
            model_name=config["name"],
            num_labels=num_labels,
            dropout=config.get("dropout", 0.2),
        )
    else:
        return ESMForGOPrediction(
            model_name=config["name"],
            num_labels=num_labels,
            dropout=config.get("dropout", 0.1),
            freeze_backbone=config.get("freeze_backbone", False),
            use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
            pooling_type=config.get("pooling_type", "mean"),
        )
