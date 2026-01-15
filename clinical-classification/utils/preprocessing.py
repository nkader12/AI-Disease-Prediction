"""
utils/preprocessing.py

Data preprocessing utilities for clinical note classification.
Handles loading, cleaning, and splitting data into train/validation/test sets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import re
from .embeddings import add_embeddings_to_df


def load_data(filepath: str, generate_embeddings: bool = False) -> pd.DataFrame:
    """
    Load clinical notes data from CSV
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file with columns: patient_identifier, text, has_cancer, 
        has_diabetes, test_set
    generate_embeddings : bool
        If True, generate embeddings for text column (default: False)
    
    Returns:
    --------
    df : pd.DataFrame
        Loaded dataframe with text column as string type
    """
    df = pd.read_csv(filepath)
    
    # Ensure text is string
    df['text'] = df['text'].fillna('').astype(str)
    
    # Convert embeddings from string to numpy array if needed
    if 'embeddings' in df.columns:
        if isinstance(df['embeddings'].iloc[0], str):
            df['embeddings'] = df['embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    elif generate_embeddings:
        print("No embeddings found, generating...")
        df = add_embeddings_to_df(df, text_column='text', embedding_column='embeddings')
    
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def get_combined_label(row: pd.Series) -> str:
    """
    Create combined label from has_cancer and has_diabetes columns
    
    Parameters:
    -----------
    row : pd.Series
        Row with has_cancer and has_diabetes columns (0.0 or 1.0)
    
    Returns:
    --------
    label : str
        One of: 'Neither', 'Cancer Only', 'Diabetes Only', 'Both'
    """
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


def create_train_val_test_splits(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into labeled train, labeled validation, unlabeled, and test sets
    
    Strategy:
    ---------
    1. Test set: Pre-defined in data (test_set == 1)
    2. Labeled pool: test_set == 0 AND has labels
    3. Split labeled pool into train/validation (stratified by combined_label)
    4. Unlabeled: test_set == 0 AND no labels
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    val_size : float
        Proportion of labeled data to hold out for validation (default: 0.2)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    df_train : pd.DataFrame
        Training set (labeled, test_set=0)
    df_validation : pd.DataFrame
        Validation set (labeled, test_set=0, held out)
    df_unlabeled : pd.DataFrame
        Unlabeled set (test_set=0, no labels)
    df_test : pd.DataFrame
        Final test set (test_set=1)
    """
    from sklearn.model_selection import train_test_split
    
    # Test set (pre-defined)
    df_test = df[df['test_set'] == 1].copy()
    
    # Labeled data (for training/validation)
    labeled_mask = (df['test_set'] == 0) & (df['has_cancer'].notna()) & (df['has_diabetes'].notna())
    df_labeled = df[labeled_mask].copy()
    
    # Unlabeled data
    unlabeled_mask = (df['test_set'] == 0) & (df['has_cancer'].isna() | df['has_diabetes'].isna())
    df_unlabeled = df[unlabeled_mask].copy()
    
    # Create combined label for stratification
    df_labeled['combined_label'] = df_labeled.apply(get_combined_label, axis=1)
    
    # Stratified split of labeled data
    df_train, df_validation = train_test_split(
        df_labeled,
        test_size=val_size,
        stratify=df_labeled['combined_label'],
        random_state=random_state
    )
    
    # Add combined_label to test set
    df_test['combined_label'] = df_test.apply(get_combined_label, axis=1)


    print(f"\nTraining distribution:")
    print(df_train['combined_label'].value_counts().to_dict())

    print(f"\nValidation distribution:")
    print(df_validation['combined_label'].value_counts().to_dict())

    print(f"\nTest distribution:")
    print(df_test['combined_label'].value_counts().to_dict())

    return df_train, df_validation, df_unlabeled, df_test


def prepare_features_labels(
    df: pd.DataFrame,
    include_text: bool = True
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract features (embeddings), labels, and texts from dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with embeddings and combined_label columns
    include_text : bool
        Whether to return text data (needed for regex matching)
    
    Returns:
    --------
    X : np.ndarray, shape (n_samples, embedding_dim)
        Feature matrix (embeddings)
    y : np.ndarray, shape (n_samples,)
        Labels
    texts : list of str (optional)
        Original text data
    """
    X = np.vstack(df['embeddings'].values)
    y = df['combined_label'].values
    
    if include_text:
        texts = df['text'].tolist()
        return X, y, texts
    else:
        return X, y, None


def validate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
    
    Returns:
    --------
    stats : dict
        Validation statistics
    """
    stats = {
        'total_records': len(df),
        'missing_text': df['text'].isna().sum(),
        'missing_embeddings': df['embeddings'].isna().sum() if 'embeddings' in df.columns else 0,
        'empty_text': (df['text'].str.len() == 0).sum(),
        'avg_text_length': df['text'].str.len().mean(),
        'labeled_records': ((df['has_cancer'].notna()) & (df['has_diabetes'].notna())).sum(),
        'unlabeled_records': ((df['has_cancer'].isna()) | (df['has_diabetes'].isna())).sum(),
    }
    
    print("="*80)
    print("DATA VALIDATION")
    print("="*80)
    for key, value in stats.items():
        print(f"  {key:20s}: {value}")
    
    return stats

