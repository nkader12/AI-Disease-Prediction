"""
Utility modules for clinical classification system.

This package contains utilities for:
- Data preprocessing and loading
- Embedding generation
- Model evaluation
- Vector database operations
"""

from .preprocessing import (
    load_data,
    get_combined_label,
    create_train_val_test_splits,
    prepare_features_labels,
    validate_data
)

from .embeddings import (
    generate_embeddings,
    add_embeddings_to_df,
    save_embeddings,
    load_embeddings
)

from .evaluation import (
    compute_metrics,
    print_evaluation_report,
    save_evaluation_results,
    create_confusion_matrix_report,
    print_confusion_matrix
)

from .vectore_db_load import (
    create_vector_store,
    load_vector_store,
    similarity_search
)

__all__ = [
    # Preprocessing
    'load_data',
    'get_combined_label',
    'create_train_val_test_splits',
    'prepare_features_labels',
    'validate_data',
    # Embeddings
    'generate_embeddings',
    'add_embeddings_to_df',
    'save_embeddings',
    'load_embeddings',
    # Evaluation
    'compute_metrics',
    'print_evaluation_report',
    'save_evaluation_results',
    'create_confusion_matrix_report',
    'print_confusion_matrix',
    # Vector Store
    'create_vector_store',
    'load_vector_store',
    'similarity_search',
]
