import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def train_universal():
    print("="*60)
    print("  TRAINING UNIVERSAL FINANCIAL HISTORIAN MODEL  ")
    print("="*60)
    
    data_path = 'e:/Barclays-ForkIT/lending_club/data/processed/universal_historian_data.parquet'
    print(f"\n⏳ Loading 2.2 Million Row Memory Mapped Parquet: {data_path} ...")
    df = pd.read_parquet(data_path)
    
    print(f"📦 Total Cohort Size: {len(df):,}")
    
    # Temporal Split to prevent look-ahead bias
    print("✂️ Performing Walk-Forward Temporal Split to prevent future leakage...")
    train_mask = df['issue_year'] <= 2016
    test_mask = df['issue_year'] >= 2017
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    print(f"   Train Set (2012-2016): {len(train_df):,} rows")
    print(f"   Test Set (2017-2018) : {len(test_df):,} rows")
    
    features = [c for c in df.columns if c not in ['target', 'issue_year']]
    X_train, y_train = train_df[features], train_df['target']
    X_test, y_test = test_df[features], test_df['target']
    
    # Calculate exact mathematical algorithmic balance
    neg_count = int(sum(y_train == 0))
    pos_count = int(sum(y_train == 1))
    scale_weight = float(neg_count) / max(pos_count, 1)
    
    print(f"⚖️ Algorithmic Imbalance Detection: Negative={neg_count:,}, Positive={pos_count:,}")
    print(f"   Configuring XGBoost Core scale_pos_weight to {scale_weight:.3f}")
    
    print("\n🚀 Initiating Heavy-Duty XGBoost Engine...")
    print("   [Hyper-Params: Trees=200, Depth=6, LR=0.08, Subsample=0.8]")
    
    # Explicitly using 'hist' for super-fast tree construction on millions of rows
    model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='aucpr',
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("\n📈 Model Evaluation on Unseen 2017-2018 Test Data:")
    
    # Custom calibration mapping: standard predict() assumes 0.50 threshold.
    # We historically bias toward aggressive recall protection.
    probs = model.predict_proba(X_test)[:, 1]
    
    # Evaluating heavily risk-averse threshold to maximize Default Catch Rate
    threshold = 0.40
    y_pred = (probs >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"   Evaluation Threshold : {threshold:.2f}")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {p:.4f} (When flagging High Risk, % true defaults)")
    print(f"   Recall    : {r:.4f} (Percentage of True Defaults successfully caught)")
    print(f"   F1 Score  : {f:.4f}")
    
    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n🔍 Explaining the AI (Proprietary-Free Feature Importance):")
    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    fi = fi.sort_values(by='Importance', ascending=False)
    for i, row in fi.iterrows():
        print(f"   - {row['Feature']:25s}: {row['Importance']:.4f}")
        
    out_dir = 'e:/Barclays-ForkIT/lending_club/outputs/models'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'universal_historian_v1.pkl')
    joblib.dump(model, out_path)
    
    # Also save features for inference compatibility
    joblib.dump(features, os.path.join(out_dir, 'universal_features_map.pkl'))
    
    print(f"\n💾 SUCCESS! XGBoost Algorithm Serialized to: {out_path}")

if __name__ == '__main__':
    train_universal()
