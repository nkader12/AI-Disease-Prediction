# utils/vector_store.py
"""
Astra DB vector store utilities for similarity search.
Handles storage and retrieval of patient embeddings.
"""

import os
import pandas as pd
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def create_vector_store(
    df: pd.DataFrame,
    collection_name: str = "patient_embeddings",
    drop_existing: bool = True
) -> AstraDBVectorStore:
    """
    Create and populate Astra DB vector store with labeled training data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training dataframe with text, labels, and patient identifiers
        Must have columns: text, patient_identifier, has_cancer, has_diabetes, combined_label
    collection_name : str
        Name for the Astra DB collection
    drop_existing : bool
        Whether to drop existing collection before creating new one
    
    Returns:
    --------
    vector_store : AstraDBVectorStore
        Initialized vector store with data loaded
    """
    token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not token or not api_endpoint:
        raise ValueError("Astra DB credentials not found in .env file")
    
    # Drop existing collection if requested
    if drop_existing:
        from astrapy import DataAPIClient
        astra_client = DataAPIClient(token)
        db = astra_client.get_database_by_api_endpoint(api_endpoint)
        
        if collection_name in db.list_collection_names():
            db.drop_collection(collection_name)
            print(f"[OK] Dropped existing collection: {collection_name}")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store with token parameter (confirmed from API signature)
    vector_store = AstraDBVectorStore(
        collection_name=collection_name,
        embedding=embeddings,
        token=token,
        api_endpoint=api_endpoint,
    )
    
    # Prepare data
    texts = df['text'].tolist()
    metadatas = [
        {
            "patient_identifier": str(row['patient_identifier']),
            "has_cancer": float(row['has_cancer']),
            "has_diabetes": float(row['has_diabetes']),
            "combined_label": row['combined_label']
        }
        for _, row in df.iterrows()
    ]
    
    # Add texts to vector store
    print(f"Adding {len(texts)} documents to vector store...")
    document_ids = vector_store.add_texts(texts=texts, metadatas=metadatas)
    print(f"[OK] Added {len(document_ids)} documents to vector store")
    
    return vector_store


def load_vector_store(
    collection_name: str = "patient_embeddings"
) -> AstraDBVectorStore:
    """
    Load existing Astra DB vector store
    
    Parameters:
    -----------
    collection_name : str
        Name of the collection to load
    
    Returns:
    --------
    vector_store : AstraDBVectorStore
        Loaded vector store
    """
    token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not token or not api_endpoint:
        raise ValueError("Astra DB credentials not found in .env file")
    
    # Check if collection exists
    from astrapy import DataAPIClient
    astra_client = DataAPIClient(token)
    db = astra_client.get_database_by_api_endpoint(api_endpoint)
    
    if collection_name not in db.list_collection_names():
        raise ValueError(f"Collection '{collection_name}' does not exist. Create it first with create_vector_store()")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load vector store with token parameter (confirmed from API signature)
    vector_store = AstraDBVectorStore(
        collection_name=collection_name,
        embedding=embeddings,
        token=token,
        api_endpoint=api_endpoint,
    )
    
    return vector_store


def similarity_search(
    vector_store: AstraDBVectorStore,
    query_text: str,
    top_k: int = 5,
    collection_name: str = None
) -> List[Dict]:
    """
    Search for similar patient cases in vector store
    
    Parameters:
    -----------
    vector_store : AstraDBVectorStore
        Initialized vector store (may be None, will be recreated if needed)
    query_text : str
        Patient text to search for similar cases
    top_k : int
        Number of similar cases to retrieve
    collection_name : str, optional
        Collection name (used if vector_store needs to be recreated)
    
    Returns:
    --------
    retrieved_cases : list of dict
        List of similar cases with metadata and similarity scores
    """
    # Get credentials - always get fresh from environment
    # Ensure .env is loaded
    load_dotenv(override=True)
    
    token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not token or not api_endpoint:
        raise ValueError(
            "Astra DB credentials not found in environment variables.\n"
            f"ASTRA_DB_APPLICATION_TOKEN: {'SET' if token else 'NOT SET'}\n"
            f"ASTRA_DB_API_ENDPOINT: {'SET' if api_endpoint else 'NOT SET'}\n"
            "Make sure these are set in your .env file in the clinical-classification directory."
        )
    
    # Get collection name
    if collection_name is None:
        collection_name = 'patient_embeddings'  # default
        if vector_store is not None:
            if hasattr(vector_store, 'collection_name'):
                collection_name = vector_store.collection_name
            elif hasattr(vector_store, '_collection_name'):
                collection_name = vector_store._collection_name
    
    # Always recreate vector store with fresh credentials to avoid authentication issues
    # This matches the notebook approach where vector store is created fresh each time
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store with fresh credentials (matching notebook implementation)
    fresh_vector_store = AstraDBVectorStore(
        collection_name=collection_name,
        embedding=embeddings,
        token=token,
        api_endpoint=api_endpoint,
    )
    
    # Perform the search with fresh vector store
    try:
        results = fresh_vector_store.similarity_search_with_score(
            query=query_text,
            k=top_k
        )
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error message
        if "authentication" in error_msg.lower() or "auth_token" in error_msg.lower() or "api_key" in error_msg.lower():
            raise ValueError(
                f"Astra DB authentication failed. Error: {error_msg}\n"
                f"Please check that ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT are correctly set in your .env file."
            )
        else:
            raise
    
    retrieved_cases = []
    for doc, score in results:
        case = {
            'text': doc.page_content,
            'has_cancer': doc.metadata.get('has_cancer', 0.0),
            'has_diabetes': doc.metadata.get('has_diabetes', 0.0),
            'combined_label': doc.metadata.get('combined_label', 'Neither'),
            'patient_id': doc.metadata.get('patient_identifier', 'unknown'),
            'similarity_score': float(score)
        }
        retrieved_cases.append(case)
    
    return retrieved_cases