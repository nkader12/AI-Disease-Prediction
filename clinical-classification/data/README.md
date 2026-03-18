# Data Directory

This directory contains clinical discharge summary data for classification.

## Data Files

### `data.csv`
Main dataset containing clinical discharge summaries with the following columns:

- `patient_identifier`: Unique patient ID
- `text`: Clinical discharge summary text (de-identified)
- `has_cancer`: Binary label (1.0 = cancer present, 0.0 = no cancer, NaN = unlabeled)
- `has_diabetes`: Binary label (1.0 = diabetes present, 0.0 = no diabetes, NaN = unlabeled)
- `test_set`: Binary flag (1 = test set, 0 = train/validation set)

### `processed/`
Directory for processed data files:
- Embeddings (`.npy` files)
- Cached preprocessed datasets
- Feature matrices

## Data Splits

The data is split into:
1. **Training Set**: Labeled data (test_set=0) used for model training
2. **Validation Set**: Labeled data (test_set=0) held out for hyperparameter tuning
3. **Test Set**: Data with test_set=1 flag (no labels for final evaluation)
4. **Unlabeled Set**: Data with test_set=0 but missing labels (for semi-supervised learning)

## Classification Labels

Combined labels are derived from `has_cancer` and `has_diabetes`:
- **Neither**: No cancer and no diabetes
- **Cancer Only**: Cancer present, no diabetes
- **Diabetes Only**: Diabetes present, no cancer
- **Both**: Both cancer and diabetes present

## Data Privacy

All patient identifiers and PHI (Protected Health Information) have been redacted
and replaced with `[REDACTED]` tokens in the discharge summaries.
