# Breast Cancer Classification · Model Comparison

An educational machine learning project comparing classification pipelines on scikit-learn's Wisconsin Diagnostic Breast Cancer dataset: 569 samples and 30 numerical features.

## Approach

1. Inspect the dataset and class balance.
2. Create an 80/20 stratified train/test split (`random_state=42`).
3. Compare unscaled logistic regression with a scaling pipeline.
4. Evaluate confusion matrices, classification reports, ROC-AUC, and learning curves.
5. Compare logistic regression, K-nearest neighbors, and SVM using five-fold cross-validation.
6. Explore PCA and tune logistic regression's regularization parameter.

## Recorded results

These values come from the notebook's saved outputs; cross-validation results are distinct from the initial holdout evaluation.

| Model / evaluation | Accuracy |
| --- | ---: |
| Unscaled logistic regression · initial holdout | 96.49% |
| Scaled logistic regression · mean 5-fold CV | 98.07% |
| Scaled KNN · mean 5-fold CV | 96.49% |
| Scaled SVM · mean 5-fold CV | 97.36% |

Scaled logistic regression's recorded initial holdout ROC-AUC is **0.9954**, using benign (class 1) as the positive class.

## Interpretation and limitations

This is a learning benchmark, not a clinical diagnostic tool. Later cross-validation and grid search use the full dataset, so the initial holdout is not an untouched final evaluation set for all subsequent model choices. Nested cross-validation or a separate final test set would strengthen the comparison. With benign coded as 1, false positives mean malignant samples predicted as benign; class-specific recall should always be read alongside the label mapping.

## Open the project

- [Analysis notebook](Breast_Cancer_Diagnosis_Model_Comparison%20%20(2).ipynb)

## Run locally

Use Python 3 in an isolated environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebook and run all cells from top to bottom. The dataset is loaded through scikit-learn.

Dependencies are listed without exact version pins; results may vary slightly across library versions.
