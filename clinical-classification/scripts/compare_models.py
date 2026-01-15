#!/usr/bin/env python3
"""
Compare Baseline Ensemble vs LLM Three-Agent System

This script compares predictions from the baseline ensemble and LLM agent system
on the test set. Since the test set has no ground truth labels, the comparison
focuses on prediction differences, agreement rates, and confidence metrics.

Usage:
    python scripts/compare_models.py

Input:
    - results/baseline_test_output.csv
    - results/llm_agents_test_output.csv

Output:
    - results/model_comparison_summary.json: Summary statistics
    - results/model_comparison_full.csv: Full comparison data
    - results/model_disagreements.csv: Cases where models disagree
    - results/high_confidence_disagreements.csv: High-confidence disagreements
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


def load_results():
    """Load both model results"""
    baseline_path = 'results/baseline_test_output.csv'
    llm_path = 'results/llm_agents_test_output.csv'
    
    print("Loading results...")
    baseline_df = pd.read_csv(baseline_path)
    llm_df = pd.read_csv(llm_path)
    
    print(f"Baseline results: {len(baseline_df)} cases")
    print(f"LLM results: {len(llm_df)} cases")
    
    return baseline_df, llm_df


def align_dataframes(baseline_df, llm_df):
    """Align dataframes by patient ID"""
    # Rename columns for consistency
    baseline_df = baseline_df.rename(columns={
        'patient_identifier': 'patient_id',
        'predicted_label': 'baseline_prediction',
        'confidence': 'baseline_confidence'
    })
    
    # Merge on patient_id
    merged = pd.merge(
        baseline_df[['patient_id', 'text', 'baseline_prediction', 'baseline_confidence']],
        llm_df[['patient_id', 'final_classification', 'pred_has_cancer', 
                'pred_has_diabetes', 'final_confidence', 
                'initial_classification', 'decision_rationale']],
        on='patient_id',
        suffixes=('_baseline', '_llm')
    )
    
    # Rename for clarity
    merged = merged.rename(columns={
        'final_classification': 'llm_prediction',
        'final_confidence': 'llm_confidence'
    })
    
    print(f"\nAligned {len(merged)} cases")
    return merged


def prediction_distribution(merged_df):
    """Compare prediction distributions"""
    print("\n" + "="*80)
    print("PREDICTION DISTRIBUTION COMPARISON")
    print("="*80)
    
    print("\nBaseline Predictions:")
    baseline_dist = merged_df['baseline_prediction'].value_counts()
    for label, count in baseline_dist.items():
        print(f"  {label:20s}: {count:3d} ({count/len(merged_df)*100:5.1f}%)")
    
    print("\nLLM Predictions:")
    llm_dist = merged_df['llm_prediction'].value_counts()
    for label, count in llm_dist.items():
        print(f"  {label:20s}: {count:3d} ({count/len(merged_df)*100:5.1f}%)")
    
    print("\n" + "-"*80)
    print("Distribution Differences:")
    print("-"*80)
    
    all_labels = set(baseline_dist.index) | set(llm_dist.index)
    differences = {}
    
    for label in sorted(all_labels):
        baseline_pct = baseline_dist.get(label, 0) / len(merged_df) * 100
        llm_pct = llm_dist.get(label, 0) / len(merged_df) * 100
        diff = llm_pct - baseline_pct
        differences[label] = {
            'baseline_pct': float(baseline_pct),
            'llm_pct': float(llm_pct),
            'difference': float(diff)
        }
        print(f"  {label:20s}: Baseline {baseline_pct:5.1f}% | LLM {llm_pct:5.1f}% | Diff {diff:+5.1f}%")
    
    return differences


def disagreement_analysis(merged_df):
    """Analyze cases where models disagree"""
    print("\n" + "="*80)
    print("DISAGREEMENT ANALYSIS")
    print("="*80)
    
    # Cases where predictions differ
    disagreements = merged_df[merged_df['baseline_prediction'] != merged_df['llm_prediction']]
    agreements = merged_df[merged_df['baseline_prediction'] == merged_df['llm_prediction']]
    
    print(f"\nTotal cases: {len(merged_df)}")
    print(f"Agreements:   {len(agreements)} ({len(agreements)/len(merged_df)*100:.1f}%)")
    print(f"Disagreements: {len(disagreements)} ({len(disagreements)/len(merged_df)*100:.1f}%)")
    
    if len(disagreements) > 0:
        print("\n" + "-"*80)
        print("Disagreement Patterns:")
        print("-"*80)
        
        disagreement_patterns = disagreements.groupby(['baseline_prediction', 'llm_prediction']).size()
        disagreement_patterns = disagreement_patterns.sort_values(ascending=False)
        
        for (baseline_pred, llm_pred), count in disagreement_patterns.items():
            print(f"  Baseline: {baseline_pred:20s} -> LLM: {llm_pred:20s} | {count:3d} cases ({count/len(disagreements)*100:5.1f}%)")
    
    return {
        'total_cases': int(len(merged_df)),
        'agreements': int(len(agreements)),
        'disagreements': int(len(disagreements)),
        'agreement_rate': float(len(agreements)/len(merged_df))
    }


def confidence_analysis(merged_df):
    """Analyze confidence scores"""
    print("\n" + "="*80)
    print("CONFIDENCE ANALYSIS")
    print("="*80)
    
    # Overall confidence statistics
    baseline_conf = merged_df['baseline_confidence'].describe()
    llm_conf = merged_df['llm_confidence'].describe()
    
    print("\nBaseline Confidence Statistics:")
    print(f"  Mean:   {baseline_conf['mean']:.3f}")
    print(f"  Median: {baseline_conf['50%']:.3f}")
    print(f"  Std:    {baseline_conf['std']:.3f}")
    print(f"  Min:    {baseline_conf['min']:.3f}")
    print(f"  Max:    {baseline_conf['max']:.3f}")
    
    print("\nLLM Confidence Statistics:")
    print(f"  Mean:   {llm_conf['mean']:.3f}")
    print(f"  Median: {llm_conf['50%']:.3f}")
    print(f"  Std:    {llm_conf['std']:.3f}")
    print(f"  Min:    {llm_conf['min']:.3f}")
    print(f"  Max:    {llm_conf['max']:.3f}")
    
    # Confidence when models agree vs disagree
    print("\n" + "-"*80)
    print("Confidence: Agreement vs Disagreement")
    print("-"*80)
    
    agreements = merged_df[merged_df['baseline_prediction'] == merged_df['llm_prediction']]
    disagreements = merged_df[merged_df['baseline_prediction'] != merged_df['llm_prediction']]
    
    if len(agreements) > 0:
        print(f"\nWhen models AGREE ({len(agreements)} cases):")
        print(f"  Baseline avg confidence: {agreements['baseline_confidence'].mean():.3f}")
        print(f"  LLM avg confidence:      {agreements['llm_confidence'].mean():.3f}")
    
    if len(disagreements) > 0:
        print(f"\nWhen models DISAGREE ({len(disagreements)} cases):")
        print(f"  Baseline avg confidence: {disagreements['baseline_confidence'].mean():.3f}")
        print(f"  LLM avg confidence:      {disagreements['llm_confidence'].mean():.3f}")
    
    # Confidence by prediction class
    print("\n" + "-"*80)
    print("Confidence by Prediction Class:")
    print("-"*80)
    
    print("\nBaseline:")
    for label in sorted(merged_df['baseline_prediction'].unique()):
        mask = merged_df['baseline_prediction'] == label
        avg_conf = merged_df[mask]['baseline_confidence'].mean()
        count = mask.sum()
        print(f"  {label:20s}: {avg_conf:.3f} ({count} cases)")
    
    print("\nLLM:")
    for label in sorted(merged_df['llm_prediction'].unique()):
        mask = merged_df['llm_prediction'] == label
        avg_conf = merged_df[mask]['llm_confidence'].mean()
        count = mask.sum()
        print(f"  {label:20s}: {avg_conf:.3f} ({count} cases)")
    
    return {
        'baseline_mean': float(baseline_conf['mean']),
        'baseline_std': float(baseline_conf['std']),
        'llm_mean': float(llm_conf['mean']),
        'llm_std': float(llm_conf['std']),
        'agreement_baseline_conf': float(agreements['baseline_confidence'].mean()) if len(agreements) > 0 else None,
        'agreement_llm_conf': float(agreements['llm_confidence'].mean()) if len(agreements) > 0 else None,
        'disagreement_baseline_conf': float(disagreements['baseline_confidence'].mean()) if len(disagreements) > 0 else None,
        'disagreement_llm_conf': float(disagreements['llm_confidence'].mean()) if len(disagreements) > 0 else None
    }


def high_confidence_disagreements(merged_df, conf_threshold=0.8):
    """Find cases where both models are confident but disagree"""
    print("\n" + "="*80)
    print(f"HIGH CONFIDENCE DISAGREEMENTS (both > {conf_threshold})")
    print("="*80)
    
    high_conf_disagree = merged_df[
        (merged_df['baseline_prediction'] != merged_df['llm_prediction']) &
        (merged_df['baseline_confidence'] > conf_threshold) &
        (merged_df['llm_confidence'] > conf_threshold)
    ]
    
    print(f"\nFound {len(high_conf_disagree)} cases where both models are confident but disagree")
    
    if len(high_conf_disagree) > 0:
        print("\nTop 10 high-confidence disagreements:")
        print("-"*80)
        
        for i, row in high_conf_disagree.head(10).iterrows():
            print(f"\nPatient {row['patient_id']}:")
            print(f"  Baseline: {row['baseline_prediction']:20s} (conf: {row['baseline_confidence']:.3f})")
            print(f"  LLM:      {row['llm_prediction']:20s} (conf: {row['llm_confidence']:.3f})")
    
    return high_conf_disagree


def save_detailed_comparison(merged_df, disagreements, high_conf_disagree, output_dir='results'):
    """Save detailed comparison results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all disagreements
    if len(disagreements) > 0:
        disagreements_file = os.path.join(output_dir, 'model_disagreements.csv')
        disagreements.to_csv(disagreements_file, index=False)
        print(f"\n[OK] Saved all disagreements to: {disagreements_file}")
    
    # Save high-confidence disagreements
    if len(high_conf_disagree) > 0:
        high_conf_file = os.path.join(output_dir, 'high_confidence_disagreements.csv')
        high_conf_disagree.to_csv(high_conf_file, index=False)
        print(f"[OK] Saved high-confidence disagreements to: {high_conf_file}")
    
    # Save full comparison
    comparison_file = os.path.join(output_dir, 'model_comparison_full.csv')
    merged_df.to_csv(comparison_file, index=False)
    print(f"[OK] Saved full comparison to: {comparison_file}")


def save_comparison_summary(summary_data, output_dir='results'):
    """Save comparison summary as JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_file = os.path.join(output_dir, 'model_comparison_summary.json')
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"[OK] Saved summary to: {summary_file}")


def main():
    """Run complete model comparison"""
    print("="*80)
    print("BASELINE ENSEMBLE vs LLM THREE-AGENT SYSTEM")
    print("MODEL COMPARISON REPORT (TEST SET - NO LABELS)")
    print("="*80)
    
    # Load data
    baseline_df, llm_df = load_results()
    
    # Align dataframes
    merged_df = align_dataframes(baseline_df, llm_df)
    
    # Run analyses
    distribution_results = prediction_distribution(merged_df)
    disagreement_results = disagreement_analysis(merged_df)
    confidence_results = confidence_analysis(merged_df)
    high_conf_disagree = high_confidence_disagreements(merged_df, conf_threshold=0.8)
    
    # Get disagreements for saving
    disagreements = merged_df[merged_df['baseline_prediction'] != merged_df['llm_prediction']]
    
    # Compile summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_cases': len(merged_df),
        'note': 'Test set has no ground truth labels - comparison focuses on prediction differences and confidence',
        'prediction_distribution': distribution_results,
        'disagreements': disagreement_results,
        'confidence': confidence_results,
        'high_confidence_disagreements': len(high_conf_disagree)
    }
    
    # Save results
    save_detailed_comparison(merged_df, disagreements, high_conf_disagree)
    save_comparison_summary(summary)
    
    print("\n" + "="*80)
    print("[OK] COMPARISON COMPLETE")
    print("="*80)
    
    # Final summary
    print("\nKEY FINDINGS:")
    print(f"  - Total cases analyzed: {len(merged_df)}")
    print(f"  - Agreement rate: {disagreement_results['agreement_rate']:.1%}")
    print(f"  - Disagreement rate: {(1-disagreement_results['agreement_rate']):.1%}")
    print(f"  - Baseline avg confidence: {confidence_results['baseline_mean']:.3f}")
    print(f"  - LLM avg confidence: {confidence_results['llm_mean']:.3f}")
    print(f"  - High-confidence disagreements: {len(high_conf_disagree)}")


if __name__ == "__main__":
    main()
