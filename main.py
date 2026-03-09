"""
Iris Flower Classifier
Classic multiclass classification with extensive visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data():
    iris = load_iris(as_frame=True)
    df   = iris.frame
    df.columns = [c.replace(' (cm)', '').replace(' ', '_') for c in df.columns]
    return df, iris.target_names


def eda(df, class_names, save_path='iris_eda.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Pairplot substitute
    features = [c for c in df.columns if c != 'target']
    colors   = ['#E74C3C', '#27AE60', '#3498DB']
    for i, feat in enumerate(features):
        ax = axes[i // 2, i % 2]
        for cls_id, color in enumerate(colors):
            subset = df[df['target'] == cls_id][feat]
            ax.hist(subset, alpha=0.6, color=color,
                    label=class_names[cls_id], bins=15)
        ax.set_title(feat.replace('_', ' ').title())
        ax.legend(fontsize=7)
    plt.suptitle('Iris Feature Distributions by Species')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"EDA plot saved to {save_path}")


def plot_correlation(df, save_path='iris_correlation.png'):
    plt.figure(figsize=(7, 5))
    features = [c for c in df.columns if c != 'target']
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Iris Feature Correlations')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def pca_visualization(X, y, class_names, save_path='iris_pca.png'):
    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(X)
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    plt.figure(figsize=(8, 6))
    for cls_id, (name, color) in enumerate(zip(class_names, colors)):
        mask = y == cls_id
        plt.scatter(Xp[mask, 0], Xp[mask, 1], c=color, label=name, alpha=0.7, s=60)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Iris Dataset — PCA Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PCA plot saved to {save_path}")


def train_all_models(X_train, X_test, y_train, y_test, class_names):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        'KNN-5':         KNeighborsClassifier(n_neighbors=5),
        'DecisionTree':  DecisionTreeClassifier(max_depth=4, random_state=42),
        'RandomForest':  RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM-RBF':       SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(model, X_tr_s, y_train, cv=5, scoring='accuracy')
        results[name] = {'model': model, 'acc': acc, 'cv_mean': cv.mean(), 'cv_std': cv.std()}
        print(f"{name}: Test Acc={acc:.4f}, CV={cv.mean():.4f}±{cv.std():.4f}")

    return results, scaler


def plot_confusion(model, scaler, X_test, y_test, class_names, save_path='confusion_matrix.png'):
    y_pred = model.predict(scaler.transform(X_test))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix — Iris Classifier')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_decision_boundary(model, scaler, X, y, class_names, save_path='decision_boundary.png'):
    """2D decision boundary using first two PCA components."""
    pca = PCA(n_components=2)
    X_s = scaler.transform(X)
    X_2d = pca.fit_transform(X_s)

    # Re-train model on 2D data
    model_2d = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    model_2d.fit(X_2d, y)

    h = 0.05
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    colors_map = ['#FADBD8', '#D5F5E3', '#D6EAF8']
    colors_pts  = ['#E74C3C', '#27AE60', '#3498DB']
    plt.figure(figsize=(8, 6))
    for cls_id in range(3):
        plt.contourf(xx, yy, Z == cls_id, alpha=0.3, colors=[colors_map[cls_id]])
    for cls_id, color in enumerate(colors_pts):
        mask = y == cls_id
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=class_names[cls_id], s=60, alpha=0.8)
    plt.title('SVM Decision Boundary (PCA 2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Decision boundary plot saved to {save_path}")


def main():
    print("=" * 60)
    print("IRIS FLOWER CLASSIFIER")
    print("=" * 60)

    df, class_names = load_data()
    print(f"Dataset: {len(df)} samples, classes: {list(class_names)}")
    print(df.describe())

    feat_cols = [c for c in df.columns if c != 'target']
    X = df[feat_cols].values
    y = df['target'].values

    eda(df, class_names)
    plot_correlation(df)
    pca_visualization(X, y, class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n--- Training Models ---")
    results, scaler = train_all_models(X_train, X_test, y_train, y_test, class_names)

    best_name = max(results, key=lambda k: results[k]['acc'])
    best_model = results[best_name]['model']
    print(f"\nBest model: {best_name} (acc={results[best_name]['acc']:.4f})")

    plot_confusion(best_model, scaler, X_test, y_test, class_names)
    plot_decision_boundary(best_model, scaler, X, y, class_names)

    print("\nFull Classification Report:")
    y_pred = best_model.predict(scaler.transform(X_test))
    print(classification_report(y_test, y_pred, target_names=class_names))

    joblib.dump({'model': best_model, 'scaler': scaler, 'features': feat_cols,
                 'classes': list(class_names)}, 'iris_model.pkl')
    print("Model saved to iris_model.pkl")
    print("\n✓ Iris Classifier complete!")


if __name__ == '__main__':
    main()
