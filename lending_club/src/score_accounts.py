import pandas as pd
import joblib
import os
import sys

def run_open_banking_inference():
    print("="*60)
    print("  OPEN BANKING INFERENCE ENGINE & RISK SCORING")
    print("="*60)
    
    # Paths
    model_path = 'e:/Barclays-ForkIT/lending_club/models/open_banking_surrogate.pkl'
    features_path = 'e:/Barclays-ForkIT/lending_club/models/surrogate_features.pkl'
    data_path = 'e:/Barclays-ForkIT/lending_club/data/processed/data_transformed.csv'
    
    if not os.path.exists(data_path):
        print(f"ERROR: Transformed data not found at {data_path}")
        sys.exit(1)
        
    print(f"Loading Models & Features...")
    model = joblib.load(model_path)
    surrogate_features = joblib.load(features_path)
    
    print(f"Loading Transformed Open Banking Ledgers...")
    df = pd.read_csv(data_path)
    accounts = df['Account Number'].copy() if 'Account Number' in df.columns else df.index.copy()
    
    # Filter to match model features exactly
    X = df[surrogate_features]
    
    print(f"Scoring {len(X)} accounts...")
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    
    # Store results
    df['Risk_Probability'] = probs * 100
    df['Risk_Class'] = ['High Risk / Denial' if p == 1 else 'Low Risk / Approved' for p in preds]
    
    # Create HTML Report
    html_content = f"""
    <html>
    <head>
        <title>Open Banking / MoneyViz Risk Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; }}
            h1 {{ color: #2c3e50; text-align: center; }}
            .container {{ width: 80%; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: center; }}
            th {{ background-color: #34495e; color: white; }}
            .high-risk {{ color: #e74c3c; font-weight: bold; background-color: #fadbd8; }}
            .low-risk {{ color: #27ae60; font-weight: bold; background-color: #d5f5e3; }}
            .summary {{ display: flex; justify-content: space-around; background: #ecf0f1; padding: 15px; border-radius: 6px; margin-bottom: 20px; }}
            .stat-box {{ text-align: center; font-size: 1.2em; }}
            .stat-val {{ font-size: 1.5em; font-weight: bold; color: #e67e22; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>💳 Automated Open Banking Underwriting Report</h1>
            <p>This report scores raw transaction accounts against a Surrogate Machine Learning model trained on classic Lending Club defaults, using strictly Open Banking convertible features.</p>
            
            <div class="summary">
                <div class="stat-box">Total Accounts Scored<br><span class="stat-val">{len(df)}</span></div>
                <div class="stat-box">High Risk Identifications<br><span class="stat-val">{(df['Risk_Probability'] > 50).sum()}</span></div>
                <div class="stat-box">Low Risk Pipeline<br><span class="stat-val">{(df['Risk_Probability'] <= 50).sum()}</span></div>
            </div>
            
            <table>
                <tr>
                    <th>Account ID</th>
                    <th>Est. Annual Income</th>
                    <th>Proxy DTI (%)</th>
                    <th>Delinquency Proxies</th>
                    <th>Risk Model Probability</th>
                    <th>Final Decision</th>
                </tr>
    """
    
    for _, row in df.iterrows():
        risk_class = "high-risk" if row['Risk_Probability'] > 50 else "low-risk"
        html_content += f"""
                <tr>
                    <td>{row['Account Number']}</td>
                    <td>${row['annual_inc']:.2f}</td>
                    <td>{row['dti']:.2f}%</td>
                    <td>{row['delinq_2yrs']}</td>
                    <td><strong>{row['Risk_Probability']:.1f}% Default Risk</strong></td>
                    <td class="{risk_class}">{row['Risk_Class']}</td>
                </tr>
        """
        
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    report_path = 'e:/Barclays-ForkIT/lending_club/reports/Risk_Report.html'
    os.makedirs('e:/Barclays-ForkIT/lending_club/reports', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"\\n✅ Execution Complete! Full visual report saved to:")
    print(f"   {report_path}")
    print("="*60)

if __name__ == "__main__":
    run_open_banking_inference()
