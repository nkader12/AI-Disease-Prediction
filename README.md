# Clinical Classification System

A dual-model system for classifying clinical discharge summaries to identify cancer and diabetes diagnoses using both traditional ML ensemble methods and LLM-based multi-agent systems.

## Overview

This project implements two classification approaches:
1. **Baseline Ensemble**: Combines Logistic Regression, Random Forest, and semi-supervised learning
2. **LLM Three-Agent System**: Uses OpenAI GPT-4 with Astra DB vector store for retrieval-augmented classification

## Project Structure

```
clinical-classification/
├── config.py                 # Central configuration
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── data/                    # Data directory
│   └── clinical_data.csv
├── models/                  # Model implementations
│   ├── baseline.py          # Baseline ensemble classifier
│   └── experimental.py      # LLM three-agent system
├── utils/                   # Utility modules
│   ├── preprocessing.py     # Data loading and preprocessing
│   ├── embeddings.py        # Embedding generation
│   ├── evaluation.py        # Metrics and evaluation
│   └── vectore_db_load.py   # Astra DB vector store operations
├── scripts/                 # Executable scripts
│   ├── run_baseline.py      # Train and evaluate baseline
│   ├── run_experimental.py  # Train and evaluate LLM system
│   └── compare_models.py    # Compare model predictions
├── notebooks/               # Jupyter notebooks for exploration
├── unit_tests/              # Unit tests
└── results/                 # Output files and reports
```

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and add your credentials:
```bash
OPENAI_API_KEY=your_openai_api_key
ASTRA_DB_APPLICATION_TOKEN=your_astra_token
ASTRA_DB_API_ENDPOINT=your_astra_endpoint
```

## Usage

### Run Baseline Model
```bash
python scripts/run_baseline.py --data_file clinical_data.csv
```

### Run LLM Agent System
```bash
python scripts/run_experimental.py --data_file clinical_data.csv --recreate_vector_store
```

### Compare Models
```bash
python scripts/compare_models.py
```

## Model Details

### Baseline Ensemble
- **Components**: Logistic Regression, Random Forest, Semi-supervised learning
- **Features**: TF-IDF text features + keyword indicators
- **Output**: Weighted ensemble predictions with confidence scores

### LLM Three-Agent System
- **Agent 1**: Similarity search using Astra DB vector store (OpenAI embeddings)
- **Agent 2**: Initial classification using GPT-4
- **Agent 3**: Final decision combining retrieval context and initial classification
- **Output**: Classification with detailed reasoning and confidence

## Classification Labels

- **Neither**: No cancer or diabetes
- **Cancer Only**: Cancer diagnosis present
- **Diabetes Only**: Diabetes diagnosis present
- **Both**: Both cancer and diabetes present

## Results

Results are saved to the `results/` directory:
- `baseline_test_output.csv`: Baseline predictions
- `llm_agents_test_output.csv`: LLM system predictions
- `model_comparison_summary.json`: Comparison statistics
- `model_disagreements.csv`: Cases where models disagree

## Requirements

- Python 3.12+
- OpenAI API access
- Astra DB account (for vector store)

## Testing

Run unit tests:
```bash
python unit_tests/test_preprocessing.py --data_file clinical_data.csv
python unit_tests/test_embeddings.py
python unit_tests/test_vectore_db.py
```
