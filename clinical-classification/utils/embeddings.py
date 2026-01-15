# utils/embeddings.py
"""
Embedding generation utilities.
Handles OpenAI embeddings creation and storage.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional
from tqdm.auto import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def generate_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    api_key: Optional[str] = None
) -> np.ndarray:
    """
    Generate embeddings for texts using OpenAI API
    
    Parameters:
    -----------
    texts : list of str
        List of text strings to embed
    model : str
        OpenAI embedding model name
    batch_size : int
        Number of texts to process per API call
    api_key : str, optional
        OpenAI API key (defaults to OPENAI_API_KEY env variable)
    
    Returns:
    --------
    embeddings : np.ndarray, shape (n_texts, embedding_dim)
        Embedding vectors (1536-dim for text-embedding-3-small)
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    embeddings = []
    
    print(f"Generating embeddings for {len(texts)} texts using {model}...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        
        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            raise
    
    embeddings_array = np.array(embeddings)
    print(f"Generated embeddings shape: {embeddings_array.shape}")
    
    return embeddings_array


def add_embeddings_to_df(
    df: pd.DataFrame,
    text_column: str = 'text',
    embedding_column: str = 'embeddings',
    model: str = "text-embedding-3-small"
) -> pd.DataFrame:
    """
    Generate embeddings and add to dataframe as new column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with text data
    text_column : str
        Name of column containing text
    embedding_column : str
        Name of column to store embeddings
    model : str
        OpenAI embedding model
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with embeddings column added
    """
    texts = df[text_column].tolist()
    embeddings = generate_embeddings(texts, model=model)
    
    df[embedding_column] = list(embeddings)
    
    print(f"Added '{embedding_column}' column to dataframe")
    return df


def save_embeddings(embeddings: np.ndarray, filepath: str):
    """
    Save embeddings array to file
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embedding array to save
    filepath : str
        Path to save embeddings (.npy file)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, embeddings)
    print(f"Saved embeddings to {filepath}")


def load_embeddings(filepath: str) -> np.ndarray:
    """
    Load embeddings from file
    
    Parameters:
    -----------
    filepath : str
        Path to embeddings file (.npy)
    
    Returns:
    --------
    embeddings : np.ndarray
        Loaded embedding array
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")
    
    embeddings = np.load(filepath)
    print(f"Loaded embeddings from {filepath}, shape: {embeddings.shape}")
    return embeddings

