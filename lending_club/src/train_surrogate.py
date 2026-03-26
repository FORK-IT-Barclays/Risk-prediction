import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def train_surrogate_model():
    print("="*60)
    print("  PRODUCTION OPEN BANKING SURROGATE MODEL PIPELINE")
    print("="*60)
    
    # 1. Load the pristine preprocessed lending club data
    data_path = 'e:/Barclays-ForkIT/lending_club/lending_club_preprocessed_sample.csv'
    print(f"\\n⏳ Loading Preprocessed Lending Club Sample: {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # 2. Define the STRICT open banking feature space
    # IMPORTANT: We can ONLY use features here that we can reasonably fake/extract from a raw bank ledger!
    # No credit bureau scores (FICO), no repayment histories, etc.
    surrogate_features = [
        'annual_inc',       # Extracted from salary/payroll credits
        'dti',              # Extracted from fixed debit outflows relative to income
        'loan_to_income',   # Assumed targeted loan request vs extracted income
        'delinq_2yrs',      # Extracted from overdraft fees, bounced checks in descriptions
    ]
    target = 'risk_label'
    
    # Validate features exist
    missing = [f for f in surrogate_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing essential mapped features in dataset: {missing}")
    
    print(f"\\n⚙️ Surrogate Features defined: {surrogate_features}")
    
    # 3. Handle any unexpected NaNs
    df = df[surrogate_features + [target]].dropna()
    print(f"✅ Extracted {len(df):,} strictly transaction-compatible profiles.")
    
    X = df[surrogate_features]
    y = df[target]

    print(f"\\n📊 Target Class Distribution (0=Good, 1=Default):")
    print(y.value_counts())

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\\n✂️  Split sizes: Train={len(X_train):,}, Test={len(X_test):,}")

    # 5. Overcome Class Imbalance (SMOTE)
    print("⚖️  Applying SMOTE to balance Defaulters vs Non-Defaulters for optimal edge detection...")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"   Balance achieved! New training size: {len(X_train_sm):,} rows.")

    # 6. Train the Extreme Gradient Boosting System
    print("\\n🚀 Training Surrogate XGBoost Model (Estimators: 150, Depth: 5) ...")
    model = xgb.XGBClassifier(
        n_estimators=150, 
        max_depth=5, 
        learning_rate=0.08, 
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        random_state=42, 
        n_jobs=-1
    )
    
    model.fit(X_train_sm, y_train_sm)
    
    # 7. Evaluate Performance
    print("\\n📈 Model Evaluation on Hold-Out Test Set:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {p:.4f} (When it flags High Risk, how often is it right?)")
    print(f"   Recall    : {r:.4f} (How many Defaulters did it successfully catch?)")
    print(f"   F1 Score  : {f:.4f}")
    
    print("\\n✅ Thorough Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 8. Feature Importance
    print("\\n🔍 Feature Importance Rankings:")
    fi = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    fi = fi.sort_values(by='Importance', ascending=False)
    for i, row in fi.iterrows():
        print(f"   - {row['Feature']:20s}: {row['Importance']:.4f}")

    # 9. Save Production Model
    os.makedirs('e:/Barclays-ForkIT/lending_club/models', exist_ok=True)
    model_path = 'e:/Barclays-ForkIT/lending_club/models/open_banking_surrogate.pkl'
    joblib.dump(model, model_path)
    joblib.dump(surrogate_features, 'e:/Barclays-ForkIT/lending_club/models/surrogate_features.pkl')
    print(f"\\n💾 SUCCESS! Open Banking Model Serialized to: {model_path}")
    print("="*60)

if __name__ == "__main__":
    train_surrogate_model()
