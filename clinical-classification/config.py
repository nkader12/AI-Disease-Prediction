"""
Central configuration file for clinical classification system.

This module contains all configuration parameters for data processing,
model training, and evaluation.
"""

import os
from typing import Dict, Any

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_CONFIG: Dict[str, Any] = {
    # Data file paths
    'data_file': 'clinical_data.csv',
    'processed_dir': 'data/processed/',
    
    # Train/validation/test split ratios
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    
    # Random seed for reproducibility
    'random_state': 42,
}

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

EMBEDDING_CONFIG: Dict[str, Any] = {
    # OpenAI embedding model
    'model': 'text-embedding-3-small',
    'dimension': 1536,
    
    # Batch processing
    'batch_size': 100,
}

# =============================================================================
# BASELINE MODEL CONFIGURATION
# =============================================================================

BASELINE_CONFIG: Dict[str, Any] = {
    # TF-IDF parameters
    'tfidf': {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
    },
    
    # Logistic Regression parameters
    'logistic': {
        'max_iter': 1000,
        'C': 1.0,
        'random_state': 42,
    },
    
    # Random Forest parameters
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'random_state': 42,
    },
    
    # Semi-supervised learning parameters
    'semisup': {
        'n_neighbors': 5,
        'max_iter': 10,
    },
    
    # Ensemble weights
    'ensemble_weights': {
        'logistic': 0.4,
        'random_forest': 0.4,
        'semisup': 0.2,
    },
}

# =============================================================================
# LLM AGENT CONFIGURATION
# =============================================================================

LLM_CONFIG: Dict[str, Any] = {
    # OpenAI model
    'model': 'gpt-4o',
    'temperature': 0.0,
    
    # Agent 1: Similarity Search
    'similarity_search': {
        'top_k': 5,
        'collection_name': 'patient_embeddings',
    },
    
    # Agent 2 & 3: LLM Classification
    'classification': {
        'max_tokens': 1000,
        'timeout': 30,
    },
}

# =============================================================================
# VECTOR STORE CONFIGURATION
# =============================================================================

VECTOR_STORE_CONFIG: Dict[str, Any] = {
    # Astra DB configuration (loaded from environment)
    'collection_name': 'patient_embeddings',
    'embedding_dimension': 1536,
    
    # Batch insertion
    'batch_size': 50,
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVALUATION_CONFIG: Dict[str, Any] = {
    # Metrics to compute
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    
    # Classification labels
    'labels': ['Neither', 'Cancer Only', 'Diabetes Only', 'Both'],
    
    # Output paths
    'results_dir': 'results/',
    'figures_dir': 'results/figures/',
    'reports_dir': 'results/reports/',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(config_name: str) -> Dict[str, Any]:
    """
    Get configuration dictionary by name.
    
    Args:
        config_name: Name of configuration (data, embedding, baseline, llm, 
                     vector_store, evaluation)
    
    Returns:
        Configuration dictionary
    
    Raises:
        ValueError: If config_name is not recognized
    """
    configs = {
        'data': DATA_CONFIG,
        'embedding': EMBEDDING_CONFIG,
        'baseline': BASELINE_CONFIG,
        'llm': LLM_CONFIG,
        'vector_store': VECTOR_STORE_CONFIG,
        'evaluation': EVALUATION_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(configs.keys())}"
        )
    
    return configs[config_name]


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_CONFIG['processed_dir'],
        EVALUATION_CONFIG['results_dir'],
        EVALUATION_CONFIG['figures_dir'],
        EVALUATION_CONFIG['reports_dir'],
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
