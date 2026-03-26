import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

def generate_all_visuals():
    print("="*60)
    print("  GENERATING UNIVERSAL HISTORIAN VISUAL ANALYTICS  ")
    print("="*60)
    
    # Paths
    base_dir = "E:/Risk-prediction/universal_historian"
    data_path = f"{base_dir}/data/universal_historian_data.parquet"
    model_path = f"{base_dir}/models/universal_historian_v1.pkl"
    plot_dir = f"{base_dir}/outputs/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"⏳ Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 2. Data Analysis Plots
    print("📊 Generating Data Analysis plots...")
    
    # Plot 1: Class Imbalance
    plt.figure(figsize=(10, 6))
    sns.countplot(x='target', data=df, palette='viridis')
    plt.title('Universal Historian: Class Imbalance (Safe vs Default)')
    plt.savefig(f"{plot_dir}/01_class_imbalance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr = df.drop(columns=['issue_year']).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Feature Correlation Matrix (13 Universal Features)')
    plt.savefig(f"{plot_dir}/02_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Feature Distributions (3x5 Grid)
    features = [c for c in df.columns if c not in ['target', 'issue_year']]
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 18))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
        axes[i].set_title(f'Dist: {col}')
    for i in range(len(features), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/03_feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Model Performance Plots
    print("📈 Generating Model Performance plots...")
    
    # Load Model
    model = joblib.load(model_path)
    
    # Split data to get test set (2017-2018)
    test_df = df[df['issue_year'] >= 2017]
    X_test = test_df[features]
    y_test = test_df['target']
    
    # Get Probabilities
    probs = model.predict_proba(X_test)[:, 1]
    
    # Plot 4: ROC and PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    
    # PR
    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(rec, prec)
    ax2.plot(rec, prec, color='green', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.savefig(f"{plot_dir}/model_roc_pr.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Confusion Matrix (at 0.40 threshold)
    threshold = 0.40
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.savefig(f"{plot_dir}/model_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Feature Importance
    plt.figure(figsize=(12, 8))
    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    fi = fi.sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=fi, palette='magma')
    plt.title('XGBoost Feature Importance (Proprietary-Free)')
    plt.savefig(f"{plot_dir}/model_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 7: Threshold Sensitivity
    thresholds = np.linspace(0.01, 0.99, 100)
    recalls = []
    precisions = []
    f1s = []
    
    for t in thresholds:
        yp = (probs >= t).astype(int)
        tp = np.sum((yp == 1) & (y_test == 1))
        fp = np.sum((yp == 1) & (y_test == 0))
        fn = np.sum((yp == 0) & (y_test == 1))
        
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        recalls.append(r)
        precisions.append(p)
        f1s.append(f1)
        
    plt.figure(figsize=(12, 7))
    plt.plot(thresholds, recalls, label='Recall (Default Catch Rate)', lw=2)
    plt.plot(thresholds, precisions, label='Precision', lw=2)
    plt.plot(thresholds, f1s, label='F1-Score', lw=2, color='black', linestyle='--')
    plt.axvline(x=0.40, color='red', linestyle=':', label='Current Threshold (0.40)')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Sensitivity Analysis')
    plt.legend()
    plt.savefig(f"{plot_dir}/model_threshold_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All plots generated successfully in {plot_dir}!")

if __name__ == '__main__':
    generate_all_visuals()
