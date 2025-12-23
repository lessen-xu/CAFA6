"""
Evaluation metrics for CAFA6 competition.
Implements weighted F1-measure using Information Accretion (IA) weights.

Location: src/utils/metrics.py

Reference: Jiang Y, et al. Genome Biol. (2016) 17(1):184
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class CAFAEvaluator:
    """
    Evaluator for CAFA competition metrics.
    
    Uses Information Accretion (IA) weighted precision and recall.
    """
    
    def __init__(self, go_processor, ontologies: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            go_processor: GOAnnotationProcessor with IA weights
            ontologies: List of ontologies to evaluate ['MFO', 'BPO', 'CCO']
        """
        self.go_processor = go_processor
        self.ontologies = ontologies or ['MFO', 'BPO', 'CCO']
        
        # Precompute IA weights for each term
        self.ia_weights = {}
        for term, ia in go_processor.ia_dict.items():
            self.ia_weights[term] = ia
    
    def get_ia(self, term: str) -> float:
        """Get Information Accretion weight for a term."""
        return self.ia_weights.get(term, 0.0)
    
    def weighted_precision_recall(
        self,
        predictions: Dict[str, Dict[str, float]],
        ground_truth: Dict[str, set],
        threshold: float
    ) -> Tuple[float, float]:
        """
        Calculate weighted precision and recall at a given threshold.
        
        Args:
            predictions: {protein_id: {go_term: probability}}
            ground_truth: {protein_id: set of go_terms}
            threshold: Probability threshold for predictions
            
        Returns:
            Tuple of (weighted_precision, weighted_recall)
        """
        total_weighted_tp = 0.0
        total_weighted_pred = 0.0
        total_weighted_true = 0.0
        
        for protein_id in ground_truth:
            if protein_id not in predictions:
                # No predictions for this protein
                true_terms = ground_truth[protein_id]
                for term in true_terms:
                    total_weighted_true += self.get_ia(term)
                continue
            
            # Get predictions above threshold
            pred_terms = {
                term for term, prob in predictions[protein_id].items()
                if prob >= threshold
            }
            true_terms = ground_truth[protein_id]
            
            # Calculate weighted TP, FP, FN
            for term in pred_terms:
                ia = self.get_ia(term)
                total_weighted_pred += ia
                if term in true_terms:
                    total_weighted_tp += ia
            
            for term in true_terms:
                ia = self.get_ia(term)
                total_weighted_true += ia
        
        # Avoid division by zero
        precision = total_weighted_tp / (total_weighted_pred + 1e-10)
        recall = total_weighted_tp / (total_weighted_true + 1e-10)
        
        return precision, recall
    
    def f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def find_optimal_threshold(
        self,
        predictions: Dict[str, Dict[str, float]],
        ground_truth: Dict[str, set],
        thresholds: List[float] = None
    ) -> Tuple[float, float, float, float]:
        """
        Find optimal threshold that maximizes F1.
        
        Returns:
            Tuple of (best_threshold, best_f1, precision, recall)
        """
        if thresholds is None:
            thresholds = np.arange(0.01, 0.99, 0.01)
        
        best_f1 = 0.0
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0
        
        for threshold in thresholds:
            precision, recall = self.weighted_precision_recall(
                predictions, ground_truth, threshold
            )
            f1 = self.f1_score(precision, recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        
        return best_threshold, best_f1, best_precision, best_recall
    
    def evaluate_per_ontology(
        self,
        predictions: Dict[str, Dict[str, float]],
        ground_truth: Dict[str, set],
        thresholds: List[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate predictions for each ontology separately.
        
        Returns:
            Dictionary with metrics for each ontology
        """
        results = {}
        
        for ont in self.ontologies:
            # Filter predictions and ground truth for this ontology
            ont_predictions = {}
            ont_ground_truth = {}
            
            for protein_id in ground_truth:
                # Filter ground truth terms
                ont_terms = {
                    term for term in ground_truth[protein_id]
                    if self.go_processor.term2ont.get(term) == ont
                }
                if ont_terms:
                    ont_ground_truth[protein_id] = ont_terms
            
            for protein_id in predictions:
                # Filter prediction terms
                ont_preds = {
                    term: prob for term, prob in predictions[protein_id].items()
                    if self.go_processor.term2ont.get(term) == ont
                }
                if ont_preds:
                    ont_predictions[protein_id] = ont_preds
            
            if not ont_ground_truth:
                results[ont] = {
                    'threshold': 0.5,
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'n_proteins': 0
                }
                continue
            
            # Find optimal threshold
            threshold, f1, precision, recall = self.find_optimal_threshold(
                ont_predictions, ont_ground_truth, thresholds
            )
            
            results[ont] = {
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'n_proteins': len(ont_ground_truth)
            }
        
        return results
    
    def compute_final_score(
        self,
        predictions: Dict[str, Dict[str, float]],
        ground_truth: Dict[str, set],
        thresholds: List[float] = None
    ) -> Tuple[float, Dict]:
        """
        Compute the final CAFA score.
        
        The final score is the arithmetic mean of the F1 scores
        for MFO, BPO, and CCO.
        
        Returns:
            Tuple of (final_score, detailed_results)
        """
        per_ont_results = self.evaluate_per_ontology(
            predictions, ground_truth, thresholds
        )
        
        # Compute mean F1 across ontologies
        f1_scores = [
            per_ont_results[ont]['f1']
            for ont in self.ontologies
            if per_ont_results[ont]['n_proteins'] > 0
        ]
        
        if f1_scores:
            final_score = np.mean(f1_scores)
        else:
            final_score = 0.0
        
        return final_score, per_ont_results


def convert_predictions_to_dict(
    protein_ids: List[str],
    predictions: np.ndarray,
    idx2term: Dict[int, str],
    threshold: float = 0.0
) -> Dict[str, Dict[str, float]]:
    """
    Convert numpy prediction array to dictionary format.
    
    Args:
        protein_ids: List of protein IDs
        predictions: numpy array of shape (n_proteins, n_terms)
        idx2term: Mapping from index to GO term
        threshold: Minimum probability to include (for memory efficiency)
    
    Returns:
        Dictionary {protein_id: {go_term: probability}}
    """
    result = {}
    
    for i, protein_id in enumerate(protein_ids):
        probs = predictions[i]
        # Only include terms above threshold
        above_thresh = np.where(probs > threshold)[0]
        
        if len(above_thresh) > 0:
            result[protein_id] = {
                idx2term[idx]: float(probs[idx])
                for idx in above_thresh
            }
    
    return result


def convert_labels_to_dict(
    protein_ids: List[str],
    labels: np.ndarray,
    idx2term: Dict[int, str]
) -> Dict[str, set]:
    """
    Convert numpy label array to dictionary format.
    
    Args:
        protein_ids: List of protein IDs
        labels: numpy array of shape (n_proteins, n_terms)
        idx2term: Mapping from index to GO term
    
    Returns:
        Dictionary {protein_id: set of go_terms}
    """
    result = {}
    
    for i, protein_id in enumerate(protein_ids):
        positive_indices = np.where(labels[i] > 0.5)[0]
        if len(positive_indices) > 0:
            result[protein_id] = {
                idx2term[idx] for idx in positive_indices
            }
    
    return result


def evaluate_batch(
    protein_ids: List[str],
    predictions: np.ndarray,
    labels: np.ndarray,
    go_processor,
    thresholds: List[float] = None
) -> Tuple[float, Dict]:
    """
    Convenience function to evaluate a batch of predictions.
    
    Returns:
        Tuple of (final_score, detailed_results)
    """
    # Convert to dictionary format
    pred_dict = convert_predictions_to_dict(
        protein_ids, predictions, go_processor.idx2term
    )
    label_dict = convert_labels_to_dict(
        protein_ids, labels, go_processor.idx2term
    )
    
    # Evaluate
    evaluator = CAFAEvaluator(go_processor)
    return evaluator.compute_final_score(pred_dict, label_dict, thresholds)

