# Iris Flower Classifier

Classic multiclass classification on the Iris dataset with rich visualizations and four ML algorithms.

## Features
- EDA: feature distributions per species
- PCA 2D visualization of clusters
- Four classifiers: KNN, Decision Tree, Random Forest, SVM
- Decision boundary visualization (SVM + PCA)
- Cross-validated accuracy comparison

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `iris_eda.png` — feature histograms by species
- `iris_pca.png` — 2D PCA scatter
- `iris_correlation.png` — feature correlation heatmap
- `confusion_matrix.png` — best model confusion matrix
- `decision_boundary.png` — SVM decision regions
- `iris_model.pkl` — saved model
