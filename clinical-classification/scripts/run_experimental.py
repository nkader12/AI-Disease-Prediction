#!/usr/bin/env python3
"""
Run LLM Three-Agent System Pipeline

This script trains and evaluates the LLM-based three-agent classification system.
The system combines similarity search (Agent 1), initial LLM classification (Agent 2),
and final decision synthesis (Agent 3) using OpenAI GPT-4 and Astra DB vector store.

Note: Vector store is always recreated to avoid duplicate records.

Usage:
    python scripts/run_experimental.py --data_file clinical_data.csv

Arguments:
    --data_file: Path to clinical data CSV file
    --collection_name: Name for Astra DB collection (default: patient_embeddings)
    --top_k: Number of similar cases to retrieve (default: 5)
    --random_state: Random seed for reproducibility (default: 42)
    --model: OpenAI model to use (default: gpt-4o)

Output:
    - results/llm_agents_test_output.csv: Predictions on test set
    - results/llm_agents_evaluation.json: Evaluation metrics
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os
import sys
from dotenv import load_dotenv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.experimental import LLMAgentSystem
from utils.preprocessing import load_data, get_combined_label
from utils.vectore_db_load import create_vector_store, load_vector_store

load_dotenv()


def save_llm_evaluation(results_df, output_dir='results'):
    """Save LLM agent evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = results_df['true_label'].tolist()
    y_pred = results_df['final_classification'].tolist()
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    per_class_metrics = {}
    for cls in ['Neither', 'Cancer Only', 'Diabetes Only', 'Both']:
        mask = results_df['true_label'] == cls
        if mask.sum() > 0:
            correct = ((results_df['true_label'] == cls) & 
                      (results_df['final_classification'] == cls)).sum()
            total = mask.sum()
            per_class_metrics[cls] = {
                'count': int(total),
                'correct': int(correct),
                'recall': float(correct / total)
            }
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'LLM Three-Agent System',
        'overall_accuracy': float(accuracy),
        'total_cases': len(results_df),
        'per_class_metrics': per_class_metrics,
        'avg_initial_confidence': float(results_df['initial_confidence'].mean()),
        'avg_final_confidence': float(results_df['final_confidence'].mean()),
        'cases_changed': int((results_df['initial_classification'] != 
                             results_df['final_classification']).sum()),
        'classification_report': classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    }
    
    output_file = os.path.join(output_dir, 'llm_agents_evaluation.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to: {output_file}")
    return output_file


def save_llm_test_predictions(results_df, output_dir='results', dataset_name='validation'):
    """Save LLM agent test predictions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output filename based on dataset
    if dataset_name == 'test':
        output_file = os.path.join(output_dir, 'llm_agents_test_output.csv')
        summary_file = os.path.join(output_dir, 'llm_agents_test_summary.txt')
    else:
        output_file = os.path.join(output_dir, 'llm_agents_validation_output.csv')
        summary_file = os.path.join(output_dir, 'llm_agents_validation_summary.txt')
    
    # Save detailed predictions
    results_df.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to: {output_file}")
    
    # Save summary
    y_true = results_df['true_label'].tolist()
    y_pred = results_df['final_classification'].tolist()
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"LLM THREE-AGENT SYSTEM - {dataset_name.upper()} SET RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total predictions: {len(results_df)}\n")
        f.write(f"Correct predictions: {results_df['correct'].sum()}\n")
        f.write(f"Accuracy: {results_df['correct'].mean():.1%}\n\n")
        
        f.write("Per-class performance:\n")
        for cls in ['Neither', 'Cancer Only', 'Diabetes Only', 'Both']:
            mask = results_df['true_label'] == cls
            if mask.sum() > 0:
                correct = ((results_df['true_label'] == cls) & 
                          (results_df['final_classification'] == cls)).sum()
                total = mask.sum()
                f.write(f"  {cls:20s}: {correct}/{total} = {correct/total:.1%}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Confidence statistics:\n")
        f.write("="*80 + "\n")
        f.write(f"  Average Initial Confidence: {results_df['initial_confidence'].mean():.3f}\n")
        f.write(f"  Average Final Confidence:   {results_df['final_confidence'].mean():.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Decision changes:\n")
        f.write("="*80 + "\n")
        changed = (results_df['initial_classification'] != results_df['final_classification']).sum()
        f.write(f"  Cases where final differed from initial: {changed} ({changed/len(results_df)*100:.1f}%)\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    return output_file, summary_file


def run_llm_agents(
    data_file='clinical_data.csv',
    collection_name='patient_embeddings',
    top_k=5,
    config=None
):
    """
    Run LLM three-agent system pipeline
    
    Note: Vector store is always recreated with fresh training data to avoid duplicates.
    
    Parameters:
    -----------
    data_file : str
        Path to data file
    collection_name : str
        Name for Astra DB collection
    top_k : int
        Number of similar cases to retrieve
    config : dict, optional
        Configuration parameters
    
    Returns:
    --------
    results : dict
        Evaluation results
    """
    print("="*80)
    print("LLM THREE-AGENT SYSTEM: COMPLETE PIPELINE")
    print("="*80)
    
    # Initialize system with OpenAI model
    model = config.get('model', 'gpt-4o') if config else 'gpt-4o'
    agent_system = LLMAgentSystem(model=model)
    
    # Ensure data file path is correct (relative to script location or absolute)
    if not os.path.isabs(data_file) and not os.path.exists(data_file):
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        possible_path = os.path.join(parent_dir, data_file)
        if os.path.exists(possible_path):
            data_file = possible_path
    
    # Load data with embeddings
    print(f"\nLoading data from: {data_file}")
    df = load_data(data_file, generate_embeddings=True)
    
    # STEP 1: Create stratified split using train_test_split
    print("\n" + "="*80)
    print("STEP 1: CREATING STRATIFIED TRAIN/VALIDATION SPLIT")
    print("="*80)
    
    # Get all labeled examples (test_set=0)
    df_labeled = df[(df['test_set'] == 0) & 
                    (df['has_cancer'].notna()) & 
                    (df['has_diabetes'].notna())].copy()
    
    # Create a combined label column for stratification
    def create_combined_label(row):
        has_cancer = row['has_cancer'] == 1.0
        has_diabetes = row['has_diabetes'] == 1.0
        
        if has_cancer and has_diabetes:
            return 'Both'
        elif has_cancer:
            return 'Cancer Only'
        elif has_diabetes:
            return 'Diabetes Only'
        else:
            return 'Neither'
    
    df_labeled['combined_label'] = df_labeled.apply(create_combined_label, axis=1)
    
    random_state = config.get('random_state', 42) if config else 42
    
    # Stratified split (80/20)
    df_train, df_val = train_test_split(
        df_labeled,
        test_size=0.2,  # 20% for validation
        stratify=df_labeled['combined_label'],
        random_state=random_state
    )
    
    # Verify stratification worked
    print("\nTraining Set Distribution")
    print("=" * 50)
    print(df_train['combined_label'].value_counts())
    print(f"\nTotal: {len(df_train)}")
    print()
    
    print("Validation Set Distribution")
    print("=" * 50)
    print(df_val['combined_label'].value_counts())
    print(f"\nTotal: {len(df_val)}")
    print()
    
    # STEP 2: Create vector store (always drop existing to avoid duplicates)
    print("\n" + "="*80)
    print("STEP 2: VECTOR STORE SETUP")
    print("="*80)
    
    print("Creating vector store with training data (dropping existing to avoid duplicates)...")
    vector_store = create_vector_store(
        df_train,
        collection_name=collection_name,
        drop_existing=True  # Always drop to prevent duplicate records
    )
    print(f"[OK] Vector store created with {len(df_train)} examples")
    
    # STEP 3: Run on validation set
    print("\n" + "="*80)
    print("STEP 3: RUNNING THREE-AGENT SYSTEM ON VALIDATION SET")
    print("="*80)
    
    val_results = []
    
    for i in range(len(df_val)):
        val_patient = df_val.iloc[i]
        
        print(f"Processing validation case {i+1}/{len(df_val)}...", end='\r')
        
        try:
            # Run three-agent pipeline
            similar_cases, initial_classification, final_decision = agent_system.predict_single(
                val_patient['text'],
                vector_store,
                top_k=top_k,
                collection_name=collection_name
            )
            
            # Evaluation
            correct_cancer = final_decision['has_cancer'] == val_patient['has_cancer']
            correct_diabetes = final_decision['has_diabetes'] == val_patient['has_diabetes']
            correct_overall = correct_cancer and correct_diabetes
            
            val_results.append({
                'patient_id': val_patient['patient_identifier'],
                'text': val_patient['text'],
                'true_label': val_patient['combined_label'],
                'true_has_cancer': val_patient['has_cancer'],
                'true_has_diabetes': val_patient['has_diabetes'],
                'initial_classification': initial_classification['classification'],
                'final_classification': final_decision['final_classification'],
                'pred_has_cancer': final_decision['has_cancer'],
                'pred_has_diabetes': final_decision['has_diabetes'],
                'initial_confidence': initial_classification['confidence'],
                'final_confidence': final_decision['final_confidence'],
                'correct': correct_overall,
                'correct_cancer': correct_cancer,
                'correct_diabetes': correct_diabetes,
                'decision_rationale': final_decision.get('decision_rationale', '')
            })
            
        except Exception as e:
            print(f"\nError processing validation case {i+1}: {e}")
            continue
    
    print("\n")
    
    # STEP 4: Evaluation on validation set
    print("\n" + "="*80)
    print("STEP 4: EVALUATION ON VALIDATION SET")
    print("="*80)
    
    val_results_df = pd.DataFrame(val_results)
    
    # Overall accuracy
    val_accuracy = val_results_df['correct'].mean()
    print(f"\nOverall Accuracy: {val_accuracy:.1%} ({val_results_df['correct'].sum()}/{len(val_results_df)})")
    
    # Combined label report
    y_true_val = val_results_df['true_label'].tolist()
    y_pred_val = val_results_df['final_classification'].tolist()
    
    print("\n" + "="*80)
    print("LLM THREE-AGENT SYSTEM PERFORMANCE (VALIDATION)")
    print("="*80)
    print(classification_report(y_true_val, y_pred_val, zero_division=0))
    
    # Per-class performance
    print("="*80)
    for cls in ['Diabetes Only', 'Cancer Only', 'Both', 'Neither']:
        mask = val_results_df['true_label'] == cls
        if mask.sum() > 0:
            correct = ((val_results_df['true_label'] == cls) & 
                      (val_results_df['final_classification'] == cls)).sum()
            total = mask.sum()
            print(f"[OK] {cls} recall: {correct}/{total} = {correct/total:.0%}")
    
    # Confidence statistics
    print(f"\n{'='*80}")
    print("CONFIDENCE STATISTICS (VALIDATION)")
    print(f"{'='*80}")
    print(f"  Average Initial Confidence: {val_results_df['initial_confidence'].mean():.3f}")
    print(f"  Average Final Confidence:   {val_results_df['final_confidence'].mean():.3f}")
    
    # Cases where classification changed
    changed_cases = (val_results_df['initial_classification'] != 
                     val_results_df['final_classification']).sum()
    print(f"\nCases where final decision differed from initial: {changed_cases} ({changed_cases/len(val_results_df)*100:.1f}%)")
    
    # Confusion matrix
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX (VALIDATION)")
    print(f"{'='*80}")
    
    label_order = ['Neither', 'Cancer Only', 'Diabetes Only', 'Both']
    cm_val = confusion_matrix(y_true_val, y_pred_val, labels=label_order)
    cm_val_df = pd.DataFrame(
        cm_val,
        index=[f'True: {l}' for l in label_order],
        columns=[f'Pred: {l}' for l in label_order]
    )
    print(cm_val_df)
    
    # Save validation results
    save_llm_evaluation(val_results_df)
    save_llm_test_predictions(val_results_df, dataset_name='validation')
    
    # STEP 5: Run on official test set (test_set=1)
    print("\n" + "="*80)
    print("STEP 5: RUNNING ON OFFICIAL TEST SET (test_set=1)")
    print("="*80)
    
    df_test = df[df['test_set'] == 1].copy()
    
    if len(df_test) > 0:
        # Add combined label if not present
        if 'combined_label' not in df_test.columns:
            df_test['combined_label'] = df_test.apply(get_combined_label, axis=1)
        
        print(f"\nOfficial test set: {len(df_test)} examples")
        
        test_results = []
        
        for i in range(len(df_test)):
            test_patient = df_test.iloc[i]
            
            print(f"Processing test case {i+1}/{len(df_test)}...", end='\r')
            
            try:
                # Run three-agent pipeline
                similar_cases, initial_classification, final_decision = agent_system.predict_single(
                    test_patient['text'],
                    vector_store,
                    top_k=top_k,
                    collection_name=collection_name
                )
                
                # Evaluation
                correct_cancer = final_decision['has_cancer'] == test_patient['has_cancer']
                correct_diabetes = final_decision['has_diabetes'] == test_patient['has_diabetes']
                correct_overall = correct_cancer and correct_diabetes
                
                test_results.append({
                    'patient_id': test_patient['patient_identifier'],
                    'text': test_patient['text'],
                    'true_label': test_patient['combined_label'],
                    'true_has_cancer': test_patient['has_cancer'],
                    'true_has_diabetes': test_patient['has_diabetes'],
                    'initial_classification': initial_classification['classification'],
                    'final_classification': final_decision['final_classification'],
                    'pred_has_cancer': final_decision['has_cancer'],
                    'pred_has_diabetes': final_decision['has_diabetes'],
                    'initial_confidence': initial_classification['confidence'],
                    'final_confidence': final_decision['final_confidence'],
                    'correct': correct_overall,
                    'correct_cancer': correct_cancer,
                    'correct_diabetes': correct_diabetes,
                    'decision_rationale': final_decision.get('decision_rationale', '')
                })
                
            except Exception as e:
                print(f"\nError processing test case {i+1}: {e}")
                continue
        
        print("\n")
        
        # Save test results
        if test_results:
            test_results_df = pd.DataFrame(test_results)
            
            # Save test predictions and summary
            save_llm_test_predictions(test_results_df, dataset_name='test')
           
    
    print(f"\n{'='*80}")
    print("[OK] LLM THREE-AGENT PIPELINE FINISHED")
    print(f"{'='*80}")
    
    return val_results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LLM three-agent system')
    parser.add_argument('--data_file', type=str, default='clinical_data.csv',
                        help='Name of data file')
    parser.add_argument('--collection_name', type=str, default='patient_embeddings',
                        help='Astra DB collection name')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of similar cases to retrieve')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='OpenAI model to use (e.g., gpt-4o, gpt-4-turbo)')
    args = parser.parse_args()
    
    config = {
        'random_state': args.random_state,
        'model': args.model
    }
    
    results = run_llm_agents(
        data_file=args.data_file,
        collection_name=args.collection_name,
        top_k=args.top_k,
        config=config
    )