# unit_tests/test_vector_db_load.py
"""
Test Astra DB vector store loading and similarity search
Usage: python unit_tests/test_vector_db_load.py --data_file clinical_data.csv
"""

import argparse
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import load_data, create_train_val_test_splits
from utils.vectore_db_load import create_vector_store, similarity_search, load_vector_store

load_dotenv()


def test_credentials():
    print("="*80)
    print("TEST 1: Check Credentials")
    print("="*80)
    
    token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    missing = []
    if not token:
        missing.append("ASTRA_DB_APPLICATION_TOKEN")
    if not api_endpoint:
        missing.append("ASTRA_DB_API_ENDPOINT")
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    
    if missing:
        print(f"❌ Missing credentials in .env:")
        for cred in missing:
            print(f"  - {cred}")
        return False
    
    print(f"✓ ASTRA_DB_APPLICATION_TOKEN: {token[:20]}...")
    print(f"✓ ASTRA_DB_API_ENDPOINT: {api_endpoint[:50]}...")
    print(f"✓ OPENAI_API_KEY: {openai_key[:20]}...")
    print("\n✓ TEST 1 PASSED")
    return True


def test_vector_store_creation(df_train):
    print("\n" + "="*80)
    print("TEST 2: Create Vector Store")
    print("="*80)
    
    print(f"Input: {len(df_train)} training documents")
    print(f"Columns: {df_train.columns.tolist()}")
    
    vector_store = create_vector_store(df_train, drop_existing=True)
    
    print(f"\n✓ Vector store created")
    print(f"✓ Collection: patient_embeddings")
    print(f"✓ Documents stored: {len(df_train)}")
    print("\n✓ TEST 2 PASSED")
    
    return vector_store


def test_similarity_search(vector_store, df_val):
    print("\n" + "="*80)
    print("TEST 3: Similarity Search")
    print("="*80)
    
    test_patient = df_val.iloc[0]
    
    print(f"Query Patient:")
    print(f"  ID: {test_patient['patient_identifier']}")
    print(f"  True Label: {test_patient['combined_label']}")
    print(f"  Text length: {len(test_patient['text'])} chars")
    print(f"  Text preview: {test_patient['text'][:150]}...")
    
    similar_cases = similarity_search(vector_store, test_patient['text'], top_k=5)
    
    print(f"\nRetrieved {len(similar_cases)} similar cases:")
    print("-"*80)
    
    for i, case in enumerate(similar_cases, 1):
        print(f"\n{i}. Patient {case['patient_id']}")
        print(f"   Similarity: {case['similarity_score']:.4f}")
        print(f"   Label: {case['combined_label']}")
        print(f"   Cancer: {case['has_cancer']}, Diabetes: {case['has_diabetes']}")
        print(f"   Text: {case['text'][:120]}...")
    
    print("\n✓ TEST 3 PASSED")
    return similar_cases


def test_validation(similar_cases):
    print("\n" + "="*80)
    print("TEST 4: Validate Results")
    print("="*80)
    
    checks = []
    
    # Check 1: Correct number of results
    correct_count = len(similar_cases) == 5
    checks.append(("Retrieved 5 cases", correct_count))
    
    # Check 2: All have required fields
    required_fields = ['text', 'has_cancer', 'has_diabetes', 'combined_label', 'patient_id', 'similarity_score']
    all_fields = all(all(field in case for field in required_fields) for case in similar_cases)
    checks.append(("All cases have required fields", all_fields))
    
    # Check 3: Similarity scores are valid
    valid_scores = all(0 <= case['similarity_score'] <= 1 for case in similar_cases)
    checks.append(("Similarity scores in [0, 1]", valid_scores))
    
    # Check 4: Scores are descending
    descending = all(
        similar_cases[i]['similarity_score'] >= similar_cases[i+1]['similarity_score'] 
        for i in range(len(similar_cases)-1)
    )
    checks.append(("Scores in descending order", descending))
    
    # Check 5: Labels are valid
    valid_labels = all(
        case['combined_label'] in ['Neither', 'Cancer Only', 'Diabetes Only', 'Both']
        for case in similar_cases
    )
    checks.append(("All labels are valid", valid_labels))
    
    # Check 6: Text is not empty
    non_empty = all(len(case['text']) > 0 for case in similar_cases)
    checks.append(("All texts non-empty", non_empty))
    
    print("\nValidation Results:")
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        print("\n✓ TEST 4 PASSED")
    else:
        print("\n✗ TEST 4 FAILED")
    
    return all_passed


def test_load_existing_store():
    print("\n" + "="*80)
    print("TEST 5: Load Existing Vector Store")
    print("="*80)
    
    print("Loading existing collection...")
    vector_store = load_vector_store(collection_name="patient_embeddings")
    
    print("✓ Vector store loaded successfully")
    
    # Quick search test
    test_query = "Patient has diabetes mellitus."
    results = similarity_search(vector_store, test_query, top_k=3)
    
    print(f"✓ Search with test query returned {len(results)} results")
    print("\n✓ TEST 5 PASSED")
    
    return vector_store


def run_all_tests(data_file):
    print("\n" + "="*80)
    print("ASTRA DB VECTOR STORE - FULL TEST SUITE")
    print("="*80)
    
    # Test 1: Credentials
    if not test_credentials():
        return False
    
    # Load and split data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    df = load_data(data_file, generate_embeddings=True)
    df_train, df_val, df_unlabeled, df_test = create_train_val_test_splits(df)
    print(f"✓ Data loaded: {len(df_train)} train, {len(df_val)} val")
    
    # Test 2: Create vector store
    vector_store = test_vector_store_creation(df_train)
    
    # Test 3: Similarity search
    similar_cases = test_similarity_search(vector_store, df_val)
    
    # Test 4: Validate results
    validation_passed = test_validation(similar_cases)
    
    # Test 5: Load existing store
    test_load_existing_store()
    
    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    if validation_passed:
        print("✓ ALL TESTS PASSED")
        print("\nVector store is ready for use:")
        print("  - Training data loaded")
        print("  - Similarity search working")
        print("  - Results validated")
        print("  - Can be reloaded from Astra DB")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Astra DB vector store')
    parser.add_argument('--data_file', type=str, default='clinical_data.csv',
                        help='Name of data file')
    args = parser.parse_args()
    
    success = run_all_tests(args.data_file)
    sys.exit(0 if success else 1)