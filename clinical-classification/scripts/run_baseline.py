#!/usr/bin/env python3
"""
Run Baseline Ensemble Pipeline

This script trains and evaluates the baseline ensemble model on clinical
discharge summaries. The ensemble combines regex patterns, logistic regression,
random forest, and semi-supervised learning.

Usage:
    python scripts/run_baseline.py --data_file layer_health_data.csv

Output:
    - results/baseline_test_output.csv: Predictions on test set
    - results/baseline_evaluation.json: Evaluation metrics
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.baseline import BaselineEnsemble
from utils.preprocessing import load_data, get_combined_label, create_train_val_test_splits
from sklearn.metrics import classification_report, accuracy_score


def save_results_to_file(results, y_test, ensemble, baseline_acc, semisup_acc, output_dir='results'):
    """
    Save evaluation results to JSON file
    
    Parameters:
    -----------
    results : dict
        Evaluation results from ensemble
    y_test : np.ndarray
        True test labels
    ensemble : BaselineEnsemble
        Trained ensemble model
    baseline_acc : float
        Baseline model accuracy
    semisup_acc : float
        Semi-supervised model accuracy
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Per-class comparison
    comparison = []
    for cls in ['Neither', 'Cancer Only', 'Diabetes Only', 'Both']:
        true_count = (y_test == cls).sum()
        if true_count > 0:
            comparison.append({
                'class': cls,
                'count': int(true_count),
                'baseline_recall': float(baseline_acc if cls in y_test else 0),
                'semisup_recall': float(semisup_acc if cls in y_test else 0),
                'ensemble_recall': float(results['per_class_recall'][cls])
            })
    
    # Build output structure
    output = {
        'timestamp': datetime.now().isoformat(),
        'overall_accuracy': {
            'baseline': float(baseline_acc),
            'semi_supervised': float(semisup_acc),
            'ensemble': float(results['accuracy'])
        },
        'per_class_performance': comparison,
        'classification_report': results['classification_report'],
        'training_metadata': {
            'n_baseline_train': int(ensemble.n_baseline_train),
            'n_semisup_train': int(ensemble.n_semisup_train),
            'n_synthetic': int(ensemble.n_synthetic),
            'baseline_neither_pct': float(ensemble.baseline_neither_pct)
        },
        'decision_strategy': {
            'tier_1': 'REGEX - Override when keywords present',
            'tier_2': 'AGREEMENT - Use when models agree',
            'tier_3': 'MODEL STRENGTHS - Route to specialized model',
            'tier_4': 'WEIGHTED VOTING - Fallback with balanced weights'
        }
    }
    
    # Save to file
    output_file = os.path.join(output_dir, 'baseline_evaluation.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to: {output_file}")
    return output_file


def save_test_predictions(df_test, predictions, decision_log, output_dir='results'):
    """
    Save test set predictions to CSV file
    
    Parameters:
    -----------
    df_test : pd.DataFrame
        Test dataframe
    predictions : np.ndarray
        Model predictions
    decision_log : list of dict
        Decision reasoning for each prediction
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output dataframe
    output_df = df_test[['patient_identifier', 'text', 'has_cancer', 'has_diabetes']].copy()
    output_df['true_label'] = df_test['combined_label']
    output_df['predicted_label'] = predictions
    output_df['correct'] = output_df['true_label'] == output_df['predicted_label']
    
    # Add decision details
    output_df['decision_reason'] = [log['reason'] for log in decision_log]
    output_df['confidence'] = [log['confidence'] for log in decision_log]
    output_df['baseline_pred'] = [log['baseline_pred'] for log in decision_log]
    output_df['baseline_conf'] = [log['baseline_conf'] for log in decision_log]
    output_df['semisup_pred'] = [log['semisup_pred'] for log in decision_log]
    output_df['semisup_conf'] = [log['semisup_conf'] for log in decision_log]
    output_df['has_diabetes_kw'] = [log['has_diabetes_kw'] for log in decision_log]
    output_df['has_cancer_kw'] = [log['has_cancer_kw'] for log in decision_log]
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'baseline_test_output.csv')
    output_df.to_csv(output_file, index=False)
    
    print(f"✓ Test predictions saved to: {output_file}")
    
  
    
    return output_file


def run_baseline(data_file='layer_health_data.csv', config=None, val_size=0.2):
    """
    Run complete baseline ensemble pipeline
    
    Parameters:
    -----------
    data_file : str
        Path to data file (default: 'layer_health_data.csv')
    config : dict, optional
        Configuration parameters for ensemble
    val_size : float
        Validation set size (default: 0.2)
    
    Returns:
    --------
    results : dict
        Evaluation results and predictions
    """
    print("="*80)
    print("FINAL ENSEMBLE: COMPLETE STANDALONE PIPELINE")
    print("="*80)
    
    # Initialize ensemble
    ensemble = BaselineEnsemble(config=config)
    
    # Load data with embeddings
    print("\nLoading data with embeddings...")
    df = load_data(data_file, generate_embeddings=True)
    
    # STEP 1: Create train/val/unlabeled/test splits using preprocessing helper
    print("\n" + "="*80)
    print("STEP 1: CREATING TRAIN/VAL/UNLABELED/TEST SPLITS")
    print("="*80)
    
    df_train, df_val, df_unlabeled, df_test = create_train_val_test_splits(
        df,
        val_size=val_size,
        random_state=config.get('random_state', 42) if config else 42
    )
    
    print(f"\nDataset splits:")
    print(f"  Train (labeled):     {len(df_train)} examples")
    print(f"  Validation (labeled): {len(df_val)} examples")
    print(f"  Unlabeled:           {len(df_unlabeled)} examples")
    print(f"  Test (test_set=1):   {len(df_test)} examples")
    
    # Combine train + val for baseline training
    df_train_combined = pd.concat([df_train, df_val], ignore_index=True)
    
    # Use ensemble's stratified split on the combined labeled data
    print("\n" + "="*80)
    print("STEP 2: CREATING STRATIFIED TRAIN/VALIDATION SPLIT FOR ENSEMBLE")
    print("="*80)
    
    train_labeled, val_labeled = ensemble.create_stratified_split(df_train_combined, get_combined_label)
    
    print(f"\nTrain set: {len(train_labeled)} examples")
    print(f"  Distribution: {train_labeled['combined_label'].value_counts().to_dict()}")
    print(f"\nValidation set: {len(val_labeled)} examples")
    print(f"  Distribution: {val_labeled['combined_label'].value_counts().to_dict()}")
    
    # STEPS 3-6: Generate synthetic labels
    print("\n" + "="*80)
    print("STEPS 3-6: GENERATING SYNTHETIC LABELS")
    print("="*80)
    
    final_expanded = ensemble.generate_synthetic_labels(
        df, train_labeled, val_labeled, get_combined_label
    )
    
    # STEP 7: Prepare training data
    print("\n" + "="*80)
    print("STEP 7: PREPARING TRAINING DATA")
    print("="*80)
    
    # Baseline = original labeled only
    X_baseline = np.vstack(train_labeled['embeddings'].values)
    y_baseline = train_labeled['combined_label'].values
    
    # Semi-supervised = labeled + synthetic
    X_semisup = np.vstack(final_expanded['embeddings'].values)
    y_semisup = final_expanded['combined_label'].values
    
    synthetic_only = final_expanded[final_expanded['source'] != 'labeled']
    
    print(f"Baseline: {len(y_baseline)} examples")
    print(f"  Distribution: {pd.Series(y_baseline).value_counts().to_dict()}")
    print(f"\nSemi-supervised: {len(y_semisup)} examples ({len(train_labeled)} labeled + {len(synthetic_only)} synthetic)")
    print(f"  Distribution: {pd.Series(y_semisup).value_counts().to_dict()}")
    
    # STEP 8: Prepare validation data
    print("\n" + "="*80)
    print("STEP 8: PREPARING VALIDATION DATA")
    print("="*80)
    
    X_val = np.vstack(val_labeled['embeddings'].values)
    y_val = val_labeled['combined_label'].values
    texts_val = val_labeled['text'].tolist()
    
    print(f"Validation set: {len(y_val)} examples")
    print(f"Distribution: {pd.Series(y_val).value_counts().to_dict()}")
    
    # STEP 9-10: Train ensemble and print strategy
    print("\n" + "="*80)
    print("STEPS 9-10: TRAINING ENSEMBLE")
    print("="*80)
    
    ensemble.train(X_baseline, y_baseline, X_semisup, y_semisup)
    
    print("✓ Baseline model trained")
    print("✓ Semi-supervised model trained")
    
    # Print detailed strategy
    ensemble.print_ensemble_strategy()
    
    # STEP 11: Evaluate on validation set
    print("\n" + "="*80)
    print("STEP 11: EVALUATION ON VALIDATION SET")
    print("="*80)
    
    results = ensemble.evaluate(X_val, y_val, texts_val)
    
    # Print ensemble performance
    print("\n" + "="*80)
    print("THEORY-DRIVEN ENSEMBLE PERFORMANCE (VALIDATION)")
    print("="*80)
    print(classification_report(y_val, results['predictions'], zero_division=0))
    
    # Get individual model predictions for comparison
    y_pred_baseline = ensemble.clf_baseline.predict(X_val)
    y_pred_semisup = ensemble.clf_semisup.predict(X_val)
    
    baseline_acc = accuracy_score(y_val, y_pred_baseline)
    semisup_acc = accuracy_score(y_val, y_pred_semisup)
    
    # Per-class comparison table
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON (VALIDATION)")
    print("="*80)
    
    comparison = []
    for cls in ['Neither', 'Cancer Only', 'Diabetes Only', 'Both']:
        true_count = (y_val == cls).sum()
        if true_count > 0:
            baseline_recall = ((y_val == cls) & (y_pred_baseline == cls)).sum() / true_count
            semisup_recall = ((y_val == cls) & (y_pred_semisup == cls)).sum() / true_count
            ensemble_recall = results['per_class_recall'][cls]
            
            comparison.append({
                'Class': cls,
                'Count': true_count,
                'Baseline': f"{baseline_recall:.0%}",
                'Semi-Sup': f"{semisup_recall:.0%}",
                'Ensemble': f"{ensemble_recall:.0%}"
            })
    
    print("\n", pd.DataFrame(comparison).to_string(index=False))
    
    # Overall accuracy comparison
    print(f"\n{'='*80}")
    print("OVERALL ACCURACY (VALIDATION)")
    print(f"{'='*80}")
    print(f"  Baseline:               {baseline_acc:.1%}")
    print(f"  Semi-Supervised:        {semisup_acc:.1%}")
    print(f"  Theory-Driven Ensemble: {results['accuracy']:.1%}")
    
    # Per-class recall summary
    print(f"\n{'='*80}")
    for cls in ['Diabetes Only', 'Cancer Only', 'Both', 'Neither']:
        if (y_val == cls).sum() > 0:
            correct = ((y_val == cls) & (results['predictions'] == cls)).sum()
            total = (y_val == cls).sum()
            print(f"✓ {cls} recall: {correct}/{total} = {correct/total:.0%}")
    
    # Save validation results
    save_results_to_file(results, y_val, ensemble, baseline_acc, semisup_acc)
    
    # STEP 12: Run on official test set (test_set == 1)
    print("\n" + "="*80)
    print("STEP 12: RUNNING ON OFFICIAL TEST SET (test_set=1)")
    print("="*80)
    
    if len(df_test) > 0:
        X_test = np.vstack(df_test['embeddings'].values)
        y_test = df_test['combined_label'].values
        texts_test = df_test['text'].tolist()
        
        print(f"\nOfficial test set: {len(y_test)} examples")
        print(f"Distribution: {pd.Series(y_test).value_counts().to_dict()}")
        
        # Get predictions
        predictions_test, decision_log_test = ensemble.predict(X_test, texts_test)
        
        # Calculate accuracy
        test_acc = accuracy_score(y_test, predictions_test)
        
        
        # Save test predictions
        save_test_predictions(df_test, predictions_test, decision_log_test)
    else:
        print("\n⚠ No official test set found (test_set=1)")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE ENSEMBLE PIPELINE FINISHED")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline ensemble pipeline')
    parser.add_argument('--data_file', type=str, default='layer_health_data.csv',
                        help='Name of data file')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation set size (default: 0.2)')
    args = parser.parse_args()
    
    config = {
        'random_state': args.random_state
    }
    
    results = run_baseline(data_file=args.data_file, config=config, val_size=args.val_size)