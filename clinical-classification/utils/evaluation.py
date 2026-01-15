"""
Evaluation utilities for clinical classification models.

Provides functions for computing metrics, generating reports,
and creating visualizations.
"""

import json
import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names (default: ['Neither', 'Cancer Only', 
                'Diabetes Only', 'Both'])
    
    Returns:
        Dictionary containing accuracy, precision, recall, F1 scores
    """
    if labels is None:
        labels = ['Neither', 'Cancer Only', 'Diabetes Only', 'Both']
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'per_class': {}
    }
    
    # Per-class breakdown
    for i, label in enumerate(labels):
        metrics['per_class'][label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return metrics


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "EVALUATION REPORT"
) -> None:
    """
    Print formatted evaluation report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names
        title: Report title
    """
    if labels is None:
        labels = ['Neither', 'Cancer Only', 'Diabetes Only', 'Both']
    
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, labels)
    
    # Overall metrics
    print(f"\nOverall Accuracy: {metrics['accuracy']:.3f}")
    print(f"\nMacro Averages:")
    print(f"  Precision: {metrics['macro_precision']:.3f}")
    print(f"  Recall:    {metrics['macro_recall']:.3f}")
    print(f"  F1 Score:  {metrics['macro_f1']:.3f}")
    
    print(f"\nWeighted Averages:")
    print(f"  Precision: {metrics['weighted_precision']:.3f}")
    print(f"  Recall:    {metrics['weighted_recall']:.3f}")
    print(f"  F1 Score:  {metrics['weighted_f1']:.3f}")
    
    # Per-class metrics
    print("\n" + "-"*80)
    print("Per-Class Metrics:")
    print("-"*80)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-"*80)
    
    for label in labels:
        if label in metrics['per_class']:
            m = metrics['per_class'][label]
            print(f"{label:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} "
                  f"{m['f1']:>10.3f} {m['support']:>10}")
    
    print("="*80)


def save_evaluation_results(
    metrics: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[OK] Saved evaluation results to: {output_path}")


def create_confusion_matrix_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create confusion matrix as DataFrame.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names
    
    Returns:
        Confusion matrix as DataFrame
    """
    if labels is None:
        labels = ['Neither', 'Cancer Only', 'Diabetes Only', 'Both']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    return cm_df


def print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> None:
    """
    Print formatted confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names
    """
    cm_df = create_confusion_matrix_report(y_true, y_pred, labels)
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    print("\nRows: True Labels | Columns: Predicted Labels\n")
    print(cm_df.to_string())
    print("\n" + "="*80)
