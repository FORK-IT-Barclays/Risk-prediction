import pandas as pd
import numpy as np
import os

def transform_transactional_data():
    print("="*60)
    print("  OPEN BANKING TRANSACTION TRANSFORMER (MoneyViz)")
    print("="*60)
    
    data_path = 'e:/Barclays-ForkIT/lending_club/data.csv'
    print(f"Loading raw ledger: {data_path}")
    df = pd.read_csv(data_path)
    
    df['Debit Amount'] = df['Debit Amount'].fillna(0.0)
    df['Credit Amount'] = df['Credit Amount'].fillna(0.0)
    df['Transaction Description'] = df['Transaction Description'].fillna('').str.lower()
    
    results = []
    
    # Iterate through unique accounts
    for account, grp in df.groupby('Account Number'):
        # 1. Income Proxy (annual_inc)
        income_tx = grp[grp['Transaction Description'].str.contains('salary|bacs|payroll|wage|employer', regex=True)]
        if len(income_tx) > 0:
            # Assuming monthly or biweekly, just sum credits and try to annualize
            # To be safe and robust, we'll take top 3 credits * 12
            avg_mth = income_tx['Credit Amount'].nlargest(3).mean()
        else:
            avg_mth = grp[grp['Credit Amount'] > 500]['Credit Amount'].mean() 
            
        if pd.isna(avg_mth):
            avg_mth = 2000
            
        annual_inc = max(avg_mth * 12, 12000)
        
        # 2. DTI Proxy (Debt to Income array)
        fixed_debits = grp[(grp['Debit Amount'] > 0) & 
                           (grp['Transaction Description'].str.contains('direct|standing|loan|mortgage|finance|credit', regex=True))]
        monthly_debt = fixed_debits['Debit Amount'].mean() * 2 # guess
        if pd.isna(monthly_debt) or monthly_debt == 0:
            monthly_debt = grp['Debit Amount'].mean() * 3
            
        dti = min((monthly_debt * 12) / annual_inc * 100, 99.9) 
        if pd.isna(dti): dti = 15.0
        
        # 3. Delinquency Proxy (delinq_2yrs)
        late_fee_events = grp['Transaction Description'].str.contains('overdraft|fee|insufficient|late|penalty|return|unpaid', regex=True).sum()
        delinq_2yrs = min(late_fee_events, 20)
        
        # 4. Loan_to_Income Proxy
        # We assume the user is applying for a $15,000 baseline loan.
        assumed_loan = 15000
        loan_to_income = assumed_loan / annual_inc

        results.append({
            'Account Number': account,
            'annual_inc': annual_inc,
            'dti': dti,
            'delinq_2yrs': delinq_2yrs,
            'loan_to_income': loan_to_income,
            'avg_balance': grp['Balance'].mean()
        })
        
    out_df = pd.DataFrame(results)
    print(f"\\n✅ Engineered {len(out_df)} Accounts from raw transaction footprints.")
    print(out_df.head())
    
    os.makedirs('e:/Barclays-ForkIT/lending_club/data/processed', exist_ok=True)
    out_path = 'e:/Barclays-ForkIT/lending_club/data/processed/data_transformed.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\\n💾 Saved engineered open banking features to: {out_path}")
    print("="*60)

if __name__ == "__main__":
    transform_transactional_data()
