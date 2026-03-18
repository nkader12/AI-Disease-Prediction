# Clinical Classification System - Technical Documentation

## Overview

This project implements a dual-model clinical classification system for identifying cancer and diabetes diagnoses from hospital discharge summaries.

## Architecture

### Two-Model Approach

1. **Baseline Ensemble Model** (`models/baseline.py`)
   - Traditional ML approach combining multiple classifiers
   - Components: Regex patterns, Logistic Regression, Random Forest, Semi-supervised learning
   - Fast, interpretable, no API dependencies

2. **LLM Three-Agent System** (`models/experimental.py`)
   - Modern LLM-based approach with retrieval augmentation
   - Agent 1: Similarity search using Astra DB vector store
   - Agent 2: Initial classification using GPT-4
   - Agent 3: Final decision synthesis
   - Higher accuracy, provides detailed reasoning

## Project Structure

```
clinical-classification/
├── config.py                 # Centralized configuration
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore patterns
├── README.md                # User-facing documentation
├── DOCUMENTATION.md         # Technical documentation (this file)
│
├── data/                    # Data directory
│   ├── README.md            # Data documentation
│   ├── clinical_data.csv
│   └── processed/           # Cached processed data
│
├── models/                  # Model implementations
│   ├── __init__.py          # Package initialization
│   ├── baseline.py          # Baseline ensemble model
│   └── experimental.py      # LLM three-agent system
│
├── utils/                   # Utility modules
│   ├── __init__.py          # Package initialization
│   ├── preprocessing.py     # Data loading and preprocessing
│   ├── embeddings.py        # OpenAI embedding generation
│   ├── evaluation.py        # Metrics and evaluation
│   └── vectore_db_load.py   # Astra DB vector store operations
│
├── scripts/                 # Executable scripts
│   ├── run_baseline.py      # Train/evaluate baseline model
│   ├── run_experimental.py  # Train/evaluate LLM system
│   └── compare_models.py    # Compare model predictions
│
├── notebooks/               # Jupyter notebooks
│   └── dev_notebook.ipynb   # Development and exploration
│
├── unit_tests/              # Unit tests
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   └── test_vectore_db.py
│
└── results/                 # Output directory
    ├── figures/             # Plots and visualizations
    └── reports/             # Evaluation reports
```

## Code Organization Standards

### Module Documentation
- Every module has a comprehensive docstring explaining its purpose
- All functions have docstrings with:
  - Description of functionality
  - Parameters with types
  - Return values with types
  - Example usage where appropriate

### Type Hints
- Function signatures use type hints for clarity
- Complex types use `typing` module (Dict, List, Tuple, Optional)

### Configuration Management
- All configuration centralized in `config.py`
- Environment-specific settings in `.env` file
- No hardcoded values in model or utility code

### Error Handling
- Informative error messages with context
- Validation of inputs and environment variables
- Graceful degradation where possible

## Data Flow

### Training Pipeline

1. **Data Loading** (`utils/preprocessing.py`)
   - Load CSV with discharge summaries
   - Validate data quality
   - Split into train/validation/test sets

2. **Feature Preparation**
   - **Baseline**: TF-IDF vectorization + keyword features
   - **LLM System**: OpenAI embeddings (text-embedding-3-small)

3. **Model Training**
   - **Baseline**: Train ensemble components on labeled data
   - **LLM System**: Populate Astra DB vector store with training embeddings

4. **Evaluation**
   - Generate predictions on validation/test sets
   - Compute metrics (accuracy, precision, recall, F1)
   - Save results to JSON and CSV files

### Prediction Pipeline

#### Baseline Ensemble
```
Input Text
    ↓
Regex Pattern Matching → Keyword Features
    ↓
TF-IDF Vectorization
    ↓
Ensemble Prediction (Logistic + RF + Semi-supervised)
    ↓
Final Classification + Confidence
```

#### LLM Three-Agent System
```
Input Text
    ↓
Agent 1: Similarity Search (Astra DB)
    ↓
Agent 2: Initial LLM Classification (GPT-4)
    ↓
Agent 3: Final Decision (GPT-4 + Retrieval Context)
    ↓
Final Classification + Reasoning + Confidence
```

## Configuration

### Environment Variables (`.env`)
```bash
OPENAI_API_KEY=sk-proj-...           # OpenAI API key
ASTRA_DB_APPLICATION_TOKEN=AstraCS:... # Astra DB token
ASTRA_DB_API_ENDPOINT=https://...   # Astra DB endpoint
```

### Model Configuration (`config.py`)
- Data split ratios
- Embedding parameters
- Model hyperparameters
- Evaluation settings

## API Dependencies

### OpenAI API
- **Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **LLM**: `gpt-4o` for classification tasks
- **Cost Consideration**: Embeddings are cached, LLM calls are minimized

### Astra DB
- **Purpose**: Vector store for similarity search
- **Collection**: `patient_embeddings`
- **Embedding Dimension**: 1536
- **Search**: Cosine similarity with top-k retrieval

## Performance Characteristics

### Baseline Ensemble
- **Speed**: Fast (< 1 second per prediction)
- **Cost**: Free (no API calls)
- **Accuracy**: Moderate (33-40% on test set)
- **Interpretability**: High (feature importance, regex matches)

### LLM Three-Agent System
- **Speed**: Slower (2-5 seconds per prediction)
- **Cost**: API usage costs (embeddings + LLM calls)
- **Accuracy**: Higher (validation set performance better)
- **Interpretability**: Provides detailed reasoning

## Testing

### Unit Tests
- `test_preprocessing.py`: Data loading and splitting
- `test_embeddings.py`: Embedding generation
- `test_vectore_db.py`: Vector store operations

### Running Tests
```bash
python unit_tests/test_preprocessing.py --data_file clinical_data.csv
python unit_tests/test_embeddings.py
python unit_tests/test_vectore_db.py
```

## Output Files

### Model Predictions
- `baseline_test_output.csv`: Baseline predictions with confidence
- `llm_agents_test_output.csv`: LLM predictions with reasoning

### Evaluation Metrics
- `baseline_evaluation.json`: Baseline performance metrics
- `llm_agents_evaluation.json`: LLM system performance metrics

### Model Comparison
- `model_comparison_summary.json`: Comparison statistics
- `model_disagreements.csv`: Cases where models disagree
- `high_confidence_disagreements.csv`: High-confidence disagreements

## Best Practices

### Code Style
- Follow PEP 8 style guidelines
- Use descriptive variable names
- Keep functions focused and modular
- Maximum line length: 100 characters

### Documentation
- Update docstrings when modifying functions
- Keep README.md current with usage examples
- Document breaking changes

### Version Control
- Commit logical units of work
- Write clear commit messages
- Don't commit sensitive data (.env files)
- Don't commit large data files or model artifacts

### Security
- Never hardcode API keys
- Use environment variables for credentials
- Add `.env` to `.gitignore`
- Redact PHI from logs and outputs

## Maintenance

### Adding New Features
1. Update relevant module in `models/` or `utils/`
2. Add tests in `unit_tests/`
3. Update documentation
4. Update `config.py` if new parameters needed

### Updating Dependencies
1. Update `requirements.txt`
2. Test with new versions
3. Document any breaking changes

### Troubleshooting
- Check `.env` file for correct credentials
- Verify Python version (3.12+)
- Check API quotas and rate limits
- Review error logs for detailed messages

## Future Enhancements

### Potential Improvements
- Add more sophisticated ensemble methods
- Implement model explainability tools
- Add real-time prediction API
- Implement active learning pipeline
- Add more comprehensive evaluation metrics
- Create web-based UI for predictions

### Scalability Considerations
- Batch processing for large datasets
- Caching strategies for embeddings
- Parallel processing for predictions
- Database optimization for vector store
