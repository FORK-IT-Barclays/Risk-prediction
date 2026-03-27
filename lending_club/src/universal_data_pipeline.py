import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def run_universal_pipeline():
    print("="*60)
    print("  UNIVERSAL FINANCIAL HISTORIAN: DATA PIPELINE  ")
    print("="*60)
    
    raw_path = 'e:/Barclays-ForkIT/lending_club/accepted_2007_to_2018Q4.csv'
    
    # 1. Load strictly necessary columns to save gigabytes of RAM
    columns_to_load = [
        'issue_d', 'loan_status', 'loan_amnt', 'term', 'installment',
        'annual_inc', 'dti', 'open_acc', 'total_acc', 'revol_bal',
        'revol_util', 'inq_last_6mths', 'delinq_2yrs', 'pub_rec'
    ]
    
    print(f"⏳ Loading raw dataset (selective structural columns only)...")
    try:
        # Load efficiently
        df = pd.read_csv(raw_path, usecols=columns_to_load, low_memory=False)
    except Exception as e:
        print(f"Error loading {raw_path}: {e}")
        return
        
    print(f"📦 Initial raw rows: {len(df):,}")
    
    # 2. Filter temporally (Post-2012 only) and drop missing ground truths
    df = df.dropna(subset=['issue_d', 'loan_status'])
    df['issue_year'] = pd.to_datetime(df['issue_d'], format='%b-%Y').dt.year
    df = df[df['issue_year'] >= 2012]
    
    print(f"🗓️ Post-2012 Filtered rows: {len(df):,}")
    
    # 3. Create Ground Truth Target (Default = 1)
    bad_statuses = ['Charged Off', 'Default', 'Late (31-120 days)']
    good_statuses = ['Fully Paid', 'Current']
    
    # Drop edge-case statuses (Grace Period, etc.)
    df = df[df['loan_status'].isin(bad_statuses + good_statuses)]
    df['target'] = df['loan_status'].apply(lambda x: 1 if x in bad_statuses else 0)
    print(f"🎯 Target class distribution:\n{df['target'].value_counts()}")
    
    # 4. Standardize text strings into floats
    print("🔨 Processing text fields into continuous mathematics...")
    df['term_months'] = df['term'].str.extract('(\d+)').astype(float)
    
    # Convert incoming columns to numeric aggressively
    numeric_base = ['annual_inc', 'dti', 'revol_util', 'delinq_2yrs', 'pub_rec', 'revol_bal', 'open_acc', 'total_acc', 'loan_amnt', 'installment']
    for col in numeric_base:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 5. Pipeline Step 1 & 2: Sentinel Imputation & Absolute Capping
    print("🧹 Executing Pipeline Step 1 & 2: Sentinel & Absolute Values...")
    df.loc[df['annual_inc'] < 100, 'annual_inc'] = df['annual_inc'].median()
    df.loc[df['dti'] > 900, 'dti'] = df['dti'].median()
    
    df.loc[df['revol_util'] > 100, 'revol_util'] = 100.0
    df.loc[df['delinq_2yrs'] < 0, 'delinq_2yrs'] = 0
    df.loc[df['pub_rec'] < 0, 'pub_rec'] = 0
    
    # 6. Pipeline Step 3: Extreme Outlier Compression (Winsorization)
    print("🗜️ Executing Pipeline Step 3: Winsorizing Wealth Outliers (99.5th Pctl)...")
    winsorize_cols = ['annual_inc', 'revol_bal', 'loan_amnt']
    for col in winsorize_cols:
        p_high = df[col].quantile(0.995)
        df[col] = df[col].clip(upper=p_high)
        
    # 7. Pipeline Step 4: Constructed Exposure Ratios
    print("🏗️ Executing Pipeline Step 4: Constructing Universal Exposure Ratios...")
    df['loan_to_income_ratio'] = df['loan_amnt'] / df['annual_inc']
    df['installment_burden'] = df['installment'] / (df['annual_inc'] / 12)
    # Replace any accidental zeroes that created infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 8. Pipeline Step 5: Final Null Sweep
    print("🧼 Executing Pipeline Step 5: Final Null Sweep Imputation...")
    final_features = [
        'annual_inc', 'loan_amnt', 'dti', 'term_months', 'open_acc', 'total_acc',
        'revol_bal', 'revol_util', 'delinq_2yrs', 'pub_rec', 'inq_last_6mths',
        'loan_to_income_ratio', 'installment_burden'
    ]
    
    # Impute medians for any stray NaNs across the matrix
    for col in final_features:
        df[col] = df[col].fillna(df[col].median())
        
    # Isolate final universal architecture set
    df_final = df[final_features + ['target', 'issue_year']]
    
    # Final Matrix Validation
    null_count = df_final.isnull().sum().sum()
    print(f"   Final Matrix Shape: {df_final.shape}")
    print(f"   Final Null Count: {null_count} (Should be 0)")
    
    # 9. Save Artifact
    out_dir = 'e:/Barclays-ForkIT/lending_club/data/processed'
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, 'universal_historian_data.parquet')
    print(f"💾 Saving dense vectorized matrix to {out_file} ...")
    df_final.to_parquet(out_file, engine='pyarrow', index=False)
    print("✅ DATA PIPELINE COMPLETE.")

if __name__ == '__main__':
    run_universal_pipeline()
