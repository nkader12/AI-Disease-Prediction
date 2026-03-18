# unit_tests/test_preprocessing_with_embeddings.py
"""
Full preprocessing pipeline test with embeddings
Usage: python unit_tests/test_preprocessing_with_embeddings.py --data_file clinical_data.csv
"""

import argparse
import os
import sys
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import (
    load_data,
    create_train_val_test_splits,
    prepare_features_labels,
    validate_data
)

load_dotenv()


def test_full_pipeline(data_file):
    print("="*80)
    print("FULL PREPROCESSING PIPELINE WITH EMBEDDINGS")
    print("="*80)
    
    # Step 1: Load data and generate embeddings if needed
    print("\nSTEP 1: Load data")
    df = load_data(data_file, generate_embeddings=True)
    
    # Step 2: Validate data
    print("\nSTEP 2: Validate data")
    stats = validate_data(df)
    
    # Step 3: Create splits
    print("\nSTEP 3: Create train/val/unlabeled/test splits")
    df_train, df_val, df_unlabeled, df_test = create_train_val_test_splits(
        df, 
        val_size=0.2, 
        random_state=42
    )
    
    # Step 4: Prepare features for labeled data (train + val)
    print("\n" + "="*80)
    print("STEP 4: Prepare features for model training")
    print("="*80)
    
    X_train, y_train, texts_train = prepare_features_labels(df_train, include_text=True)
    X_val, y_val, texts_val = prepare_features_labels(df_val, include_text=True)
    X_test, y_test, texts_test = prepare_features_labels(df_test, include_text=True)
    
    print(f"\nLabeled Training Set:")
    print(f"  Features (X_train): {X_train.shape}")
    print(f"  Labels (y_train): {y_train.shape}")
    print(f"  Texts: {len(texts_train)} texts")
    print(f"  Sample label: {y_train[0]}")
    
    print(f"\nLabeled Validation Set:")
    print(f"  Features (X_val): {X_val.shape}")
    print(f"  Labels (y_val): {y_val.shape}")
    print(f"  Texts: {len(texts_val)} texts")
    
    print(f"\nTest Set:")
    print(f"  Features (X_test): {X_test.shape}")
    print(f"  Labels (y_test): {y_test.shape}")
    print(f"  Texts: {len(texts_test)} texts")
    
    # Step 5: Prepare unlabeled data
    print("\n" + "="*80)
    print("STEP 5: Prepare unlabeled data for LLM approach")
    print("="*80)
    
    if 'embeddings' in df_unlabeled.columns:
        X_unlabeled = np.vstack(df_unlabeled['embeddings'].values)
        texts_unlabeled = df_unlabeled['text'].tolist()
        
        print(f"\nUnlabeled Set:")
        print(f"  Features (X_unlabeled): {X_unlabeled.shape}")
        print(f"  Texts: {len(texts_unlabeled)} texts")
        print(f"  No labels (for prediction)")
    else:
        print("⚠ No embeddings in unlabeled set")
    
    # Step 6: Summary of outputs
    print("\n" + "="*80)
    print("OUTPUT SUMMARY - 3 DATAFRAMES READY")
    print("="*80)
    
    print("\n1. LABELED DATA (for traditional ML training):")
    print(f"   df_train: {len(df_train)} samples")
    print(f"   df_val: {len(df_val)} samples")
    print(f"   ✓ Has embeddings: {'embeddings' in df_train.columns}")
    print(f"   ✓ Has labels: {'combined_label' in df_train.columns}")
    print(f"   ✓ Has text: {'text' in df_train.columns}")
    print(f"   → Ready for: BaselineEnsemble training")
    
    print("\n2. UNLABELED DATA (for LLM approach):")
    print(f"   df_unlabeled: {len(df_unlabeled)} samples")
    print(f"   ✓ Has embeddings: {'embeddings' in df_unlabeled.columns}")
    print(f"   ✓ Has text: {'text' in df_unlabeled.columns}")
    print(f"   ✗ No labels (to be predicted)")
    print(f"   → Ready for: LLM three-agent system")
    
    print("\n3. TEST DATA (for final evaluation):")
    print(f"   df_test: {len(df_test)} samples")
    print(f"   ✓ Has embeddings: {'embeddings' in df_test.columns}")
    print(f"   ✓ Has labels: {'combined_label' in df_test.columns}")
    print(f"   ✓ Has text: {'text' in df_test.columns}")
    print(f"   → Ready for: Final model comparison")
    
    # Validation checks
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    checks = []
    
    # Check 1: No data leakage
    train_ids = set(df_train['patient_identifier'])
    val_ids = set(df_val['patient_identifier'])
    test_ids = set(df_test['patient_identifier'])
    unlabeled_ids = set(df_unlabeled['patient_identifier'])
    
    no_overlap = (
        len(train_ids & test_ids) == 0 and
        len(val_ids & test_ids) == 0 and
        len(train_ids & unlabeled_ids) == 0 and
        len(val_ids & unlabeled_ids) == 0
    )
    checks.append(("No data leakage between sets", no_overlap))
    
    # Check 2: All data accounted for
    total_split = len(df_train) + len(df_val) + len(df_unlabeled) + len(df_test)
    all_accounted = total_split == len(df)
    checks.append(("All data accounted for", all_accounted))
    
    # Check 3: Embeddings present
    has_embeddings = (
        'embeddings' in df_train.columns and
        'embeddings' in df_val.columns and
        'embeddings' in df_test.columns and
        'embeddings' in df_unlabeled.columns
    )
    checks.append(("All sets have embeddings", has_embeddings))
    
    # Check 4: Labels present in labeled sets
    has_labels = (
        'combined_label' in df_train.columns and
        'combined_label' in df_val.columns and
        'combined_label' in df_test.columns
    )
    checks.append(("Labeled sets have combined_label", has_labels))
    
    # Check 5: Correct shapes
    correct_shapes = (
        X_train.shape[0] == len(df_train) and
        X_val.shape[0] == len(df_val) and
        X_test.shape[0] == len(df_test)
    )
    checks.append(("Feature shapes match dataframes", correct_shapes))
    
    # Print results
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        print("\n" + "="*80)
        print("✓ ALL CHECKS PASSED - DATA READY FOR MODELING")
        print("="*80)
        return df_train, df_val, df_unlabeled, df_test
    else:
        print("\n" + "="*80)
        print("✗ SOME CHECKS FAILED")
        print("="*80)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test full preprocessing pipeline')
    parser.add_argument('--data_file', type=str, default='clinical_data.csv',
                        help='Name of data file')
    args = parser.parse_args()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("Embeddings will not be generated")
    
    result = test_full_pipeline(args.data_file)
    
    if result:
        df_train, df_val, df_unlabeled, df_test = result
        print("\n✓ Pipeline complete - dataframes ready for use")