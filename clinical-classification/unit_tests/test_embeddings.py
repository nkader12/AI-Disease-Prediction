# unit_tests/test_embeddings.py
"""
Test embedding generation
Usage: python unit_tests/test_embeddings.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.embeddings import (
    generate_embeddings,
    add_embeddings_to_df,
    save_embeddings,
    load_embeddings
)

load_dotenv()


def test_generate_embeddings():
    print("="*80)
    print("TEST 1: Generate Embeddings")
    print("="*80)
    
    test_texts = [
        "Patient has type 2 diabetes mellitus.",
        "Diagnosed with lung cancer, stage IV.",
        "No significant medical history."
    ]
    
    print(f"Input: {len(test_texts)} texts")
    embeddings = generate_embeddings(test_texts)
    
    print(f"\nOutput validation:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Expected: ({len(test_texts)}, 1536)")
    print(f"  Dimension match: {embeddings.shape[1] == 1536}")
    print(f"  Count match: {embeddings.shape[0] == len(test_texts)}")
    print(f"  No NaN values: {not np.any(np.isnan(embeddings))}")
    print(f"  Sample values: {embeddings[0][:5]}...")
    
    assert embeddings.shape[0] == len(test_texts), "Row count mismatch"
    assert embeddings.shape[1] == 1536, "Embedding dimension should be 1536"
    assert not np.any(np.isnan(embeddings)), "Found NaN values"
    
    print("\n✓ TEST 1 PASSED")
    return embeddings


def test_add_to_dataframe():
    print("\n" + "="*80)
    print("TEST 2: Add Embeddings to DataFrame")
    print("="*80)
    
    test_df = pd.DataFrame({
        'patient_id': [1, 2, 3],
        'text': [
            "Patient has diabetes.",
            "Cancer diagnosis confirmed.",
            "Healthy patient."
        ]
    })
    
    print(f"Input dataframe shape: {test_df.shape}")
    print(f"Columns before: {test_df.columns.tolist()}")
    
    result_df = add_embeddings_to_df(test_df)
    
    print(f"\nOutput validation:")
    print(f"  Dataframe shape: {result_df.shape}")
    print(f"  Columns after: {result_df.columns.tolist()}")
    print(f"  Has 'embeddings' column: {'embeddings' in result_df.columns}")
    print(f"  Row count preserved: {len(result_df) == len(test_df)}")
    print(f"  Embedding type: {type(result_df['embeddings'].iloc[0])}")
    print(f"  Embedding shape: {result_df['embeddings'].iloc[0].shape}")
    
    assert 'embeddings' in result_df.columns, "Missing embeddings column"
    assert len(result_df) == 3, "Row count changed"
    assert isinstance(result_df['embeddings'].iloc[0], np.ndarray), "Wrong type"
    
    print("\n✓ TEST 2 PASSED")
    return result_df


def test_save_load():
    print("\n" + "="*80)
    print("TEST 3: Save and Load Embeddings")
    print("="*80)
    
    test_embeddings = np.random.rand(5, 1536)
    test_path = "data/processed/test_embeddings.npy"
    
    print(f"Creating test embeddings: {test_embeddings.shape}")
    print(f"Saving to: {test_path}")
    
    save_embeddings(test_embeddings, test_path)
    
    print(f"File exists: {os.path.exists(test_path)}")
    print(f"File size: {os.path.getsize(test_path)} bytes")
    
    loaded = load_embeddings(test_path)
    
    print(f"\nLoaded embeddings shape: {loaded.shape}")
    print(f"Arrays match: {np.allclose(test_embeddings, loaded)}")
    
    assert np.allclose(test_embeddings, loaded), "Arrays don't match"
    

    
    print("\n✓ TEST 3 PASSED")


def run_all_tests():
    print("\n" + "="*80)
    print("EMBEDDING UNIT TESTS")
    print("="*80)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ OPENAI_API_KEY not found in .env file")
        print("Please add: OPENAI_API_KEY=your_key_here")
        return
    
    print(f"✓ Found OPENAI_API_KEY: {api_key[:20]}...")
    
    try:
        test_generate_embeddings()
        test_add_to_dataframe()
        test_save_load()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()