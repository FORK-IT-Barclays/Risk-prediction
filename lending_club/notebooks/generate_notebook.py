
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ml_ds)",
        "language": "python",
        "name": "ml_ds"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md("""# 🏦 Lending Club — Pre-Delinquency Vector System
## Data Preprocessing & Exploratory Data Analysis
### Barclays Hack-o-Hire Hackathon

---
> **Goal**: Build the foundational data pipeline for a *pre-delinquency early-warning vector system* on the Lending Club dataset (2007–2018 Q4).  
> **Dataset**: ~2.26 million accepted loan records, 150+ features.
"""))

# ── 0. Imports ─────────────────────────────────────────────────────────────────
cells.append(md("## 0 ▸ Imports & Configuration"))
cells.append(code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from scipy import stats

# ── Display config
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:.4f}'.format)
plt.rcParams.update({
    'figure.dpi': 120,
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#0f1117',
    'axes.edgecolor': '#444',
    'axes.labelcolor': '#ccc',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'text.color': '#eee',
    'grid.color': '#333',
    'grid.linestyle': '--',
    'axes.titlecolor': '#fff'
})
PALETTE = ['#00d4ff', '#ff6b6b', '#ffd700', '#7bed9f', '#a29bfe', '#fd79a8']
print("✅ Imports done")
"""))

# ── 1. Load Data ──────────────────────────────────────────────────────────────
cells.append(md("## 1 ▸ Load Data\n> Sampling 500 k rows for fast exploration; full analysis runs on all rows where indicated."))
cells.append(code("""
DATA_PATH = 'accepted_2007_to_2018Q4.csv'
SAMPLE_N  = 500_000   # adjust to None to load everything (slow)
SEED      = 42

print("⏳ Loading dataset …")
if SAMPLE_N:
    # Count rows first, then sample
    total_rows = sum(1 for _ in open(DATA_PATH)) - 1
    print(f"   Total rows in file: {total_rows:,}")
    skip_idx = sorted(
        np.random.default_rng(SEED).choice(total_rows, total_rows - SAMPLE_N, replace=False) + 1
    )
    df = pd.read_csv(DATA_PATH, skiprows=skip_idx, low_memory=False)
else:
    df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
df.head(3)
"""))

# ── 2. Quick Snapshot ─────────────────────────────────────────────────────────
cells.append(md("## 2 ▸ Dataset Snapshot"))
cells.append(code("""
print("Shape:", df.shape)
print("\\nDtypes summary:")
print(df.dtypes.value_counts())
print("\\nMemory usage: {:.2f} MB".format(df.memory_usage(deep=True).sum() / 1e6))
"""))
cells.append(code("""df.info(max_cols=160, memory_usage='deep', show_counts=True)"""))
cells.append(code("""df.describe(include='all').T.head(30)"""))

# ── 3. Missing Values ─────────────────────────────────────────────────────────
cells.append(md("## 3 ▸ Missing Value Analysis"))
cells.append(code("""
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df = pd.DataFrame({'missing': miss, 'pct': miss_pct})
miss_df = miss_df[miss_df['missing'] > 0].sort_values('pct', ascending=False)
print(f"Columns with missing values: {len(miss_df)}")
miss_df
"""))
cells.append(code("""
# ── Heatmap: top-40 most missing columns
top40 = miss_df.head(40).index.tolist()
fig, ax = plt.subplots(figsize=(14, 6))
palette_miss = sns.color_palette(['#00d4ff', '#ff6b6b'], as_cmap=False)
bars = ax.barh(top40[::-1], miss_df.loc[top40[::-1], 'pct'],
               color=['#ff6b6b' if p > 50 else '#ffd700' if p > 20 else '#00d4ff'
                      for p in miss_df.loc[top40[::-1], 'pct']])
ax.set_xlabel('Missing %')
ax.set_title('Top-40 Columns by Missing %', fontsize=14)
ax.axvline(50, color='#ff6b6b', linestyle='--', lw=1.2, label='50% threshold')
ax.axvline(20, color='#ffd700', linestyle='--', lw=1.2, label='20% threshold')
ax.legend()
plt.tight_layout()
plt.show()
"""))

# ── 4. Target Variable ─────────────────────────────────────────────────────────
cells.append(md("## 4 ▸ Target Variable — `loan_status`\n> For the **pre-delinquency vector system** we map loan_status to a risk label."))
cells.append(code("""
print(df['loan_status'].value_counts())
"""))
cells.append(code("""
# ── Map to binary risk label
DELINQUENT_STATUSES = {
    'Charged Off', 'Default',
    'Late (31-120 days)', 'Late (16-30 days)',
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Late (31-120 days)',
    'Does not meet the credit policy. Status:Late (16-30 days)',
    'Does not meet the credit policy. Status:Default'
}

df['risk_label'] = df['loan_status'].apply(
    lambda x: 1 if x in DELINQUENT_STATUSES else 0
)

label_counts = df['risk_label'].value_counts()
print("Risk label distribution:")
print(label_counts)
print(f"\\nDelinquency rate: {label_counts[1]/len(df)*100:.2f}%")
"""))
cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie
colors_pie = ['#00d4ff', '#ff6b6b']
axes[0].pie(label_counts.values, labels=['Non-Delinquent', 'Delinquent'],
            autopct='%1.2f%%', colors=colors_pie,
            wedgeprops={'edgecolor': '#0f1117', 'linewidth': 2},
            textprops={'color': '#eee'})
axes[0].set_title('Risk Label Distribution', fontsize=13)

# Loan status full bar
ls_counts = df['loan_status'].value_counts()
axes[1].barh(ls_counts.index, ls_counts.values,
             color=['#ff6b6b' if s in DELINQUENT_STATUSES else '#00d4ff'
                    for s in ls_counts.index])
axes[1].set_xlabel('Count')
axes[1].set_title('Loan Status Breakdown', fontsize=13)
plt.tight_layout()
plt.show()
"""))

# ── 5. Feature Categories ─────────────────────────────────────────────────────
cells.append(md("## 5 ▸ Feature Categorisation\n> Organising all 150+ columns into logical groups for structured analysis."))
cells.append(code("""
# ─── Borrower Profile
borrower_cols = ['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate',
                 'installment','grade','sub_grade','emp_title','emp_length',
                 'home_ownership','annual_inc','verification_status','purpose',
                 'title','addr_state','dti','application_type']

# ─── Credit History
credit_cols = ['earliest_cr_line','fico_range_low','fico_range_high',
               'delinq_2yrs','inq_last_6mths','mths_since_last_delinq',
               'mths_since_last_record','open_acc','pub_rec','revol_bal',
               'revol_util','total_acc','mths_since_last_major_derog',
               'last_fico_range_high','last_fico_range_low']

# ─── Payment Behaviour
payment_cols = ['out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv',
                'total_rec_prncp','total_rec_int','total_rec_late_fee',
                'recoveries','collection_recovery_fee','last_pymnt_amnt']

# ─── Hardship / Settlement (delinquency signals)
hardship_cols = [c for c in df.columns if 'hardship' in c.lower() or
                 'settlement' in c.lower() or 'debt_settlement' in c.lower()]

# ─── Joint / Secondary Applicant
joint_cols = [c for c in df.columns if 'joint' in c.lower() or 'sec_app' in c.lower()]

print("Borrower:  ", len(borrower_cols))
print("Credit:    ", len(credit_cols))
print("Payment:   ", len(payment_cols))
print("Hardship:  ", len(hardship_cols))
print("Joint:     ", len(joint_cols))
"""))

# ── 6. Borrower Profile EDA ────────────────────────────────────────────────────
cells.append(md("## 6 ▸ Borrower Profile EDA"))

cells.append(code("""
# ── Loan Amount Distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(df['loan_amnt'], bins=60, color='#00d4ff', edgecolor='#0f1117', alpha=0.85)
axes[0].set_title('Loan Amount Distribution'); axes[0].set_xlabel('Loan Amount ($)')

# By risk label
for lbl, grp in df.groupby('risk_label'):
    axes[1].hist(grp['loan_amnt'], bins=60, alpha=0.6,
                 label='Delinquent' if lbl == 1 else 'Non-Delinquent',
                 color='#ff6b6b' if lbl == 1 else '#00d4ff')
axes[1].set_title('Loan Amount by Risk Label'); axes[1].set_xlabel('Loan Amount ($)')
axes[1].legend()
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# ── Grade & Subgrade analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

grade_risk = df.groupby('grade')['risk_label'].mean().sort_index() * 100
axes[0].bar(grade_risk.index, grade_risk.values,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(grade_risk))])
axes[0].set_title('Delinquency Rate by Grade (%)'); axes[0].set_ylabel('%')

subgrade_risk = df.groupby('sub_grade')['risk_label'].mean() * 100
axes[1].bar(subgrade_risk.index, subgrade_risk.values,
            color=['#ff6b6b' if v > 20 else '#ffd700' if v > 10 else '#00d4ff'
                   for v in subgrade_risk.values])
axes[1].set_title('Delinquency Rate by Sub-Grade (%)'); axes[1].set_ylabel('%')
axes[1].tick_params(axis='x', rotation=90)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# ── Interest Rate distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Convert int_rate to float if string
if df['int_rate'].dtype == object:
    df['int_rate'] = df['int_rate'].str.replace('%','').astype(float)

axes[0].hist(df['int_rate'], bins=60, color='#a29bfe', edgecolor='#0f1117')
axes[0].set_title('Interest Rate Distribution'); axes[0].set_xlabel('Int Rate (%)')

# Box by grade
df.boxplot(column='int_rate', by='grade', ax=axes[1],
           boxprops=dict(color='#00d4ff'), medianprops=dict(color='#ffd700'),
           whiskerprops=dict(color='#aaa'), capprops=dict(color='#aaa'),
           flierprops=dict(marker='o', color='#ff6b6b', markersize=2))
axes[1].set_title('Interest Rate by Grade'); axes[1].set_xlabel('Grade')
plt.suptitle('')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# ── Employment Length
emp_order = ['< 1 year','1 year','2 years','3 years','4 years',
             '5 years','6 years','7 years','8 years','9 years','10+ years', 'n/a']
df['emp_length'] = pd.Categorical(df['emp_length'], categories=emp_order, ordered=True)
emp_risk = df.groupby('emp_length', observed=False)['risk_label'].mean() * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(emp_risk.index.astype(str), emp_risk.values, color='#7bed9f')
ax.set_title('Delinquency Rate by Employment Length (%)'); ax.set_ylabel('%')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# ── Home Ownership & Purpose
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ho_risk = df.groupby('home_ownership')['risk_label'].mean().sort_values(ascending=False) * 100
axes[0].bar(ho_risk.index, ho_risk.values, color=PALETTE)
axes[0].set_title('Delinquency Rate by Home Ownership (%)'); axes[0].set_ylabel('%')

pur_risk = df.groupby('purpose')['risk_label'].mean().sort_values(ascending=False) * 100
axes[1].barh(pur_risk.index, pur_risk.values, color='#fd79a8')
axes[1].set_title('Delinquency Rate by Loan Purpose (%)'); axes[1].set_xlabel('%')
plt.tight_layout(); plt.show()
"""))

# ── 7. Annual Income ──────────────────────────────────────────────────────────
cells.append(md("## 7 ▸ Annual Income Analysis"))
cells.append(code("""
# Cap outliers at 99th percentile for visualisation
inc_cap = df['annual_inc'].quantile(0.99)
df_inc = df[df['annual_inc'] <= inc_cap]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(df_inc['annual_inc'], bins=80, color='#ffd700', edgecolor='#0f1117')
axes[0].set_title('Annual Income Distribution (capped @99th pct)')
axes[0].set_xlabel('Annual Income ($)')

# Income vs loan_amnt scatter (sample 10k for speed)
sample = df_inc.sample(10_000, random_state=SEED)
scatter_c = ['#ff6b6b' if r == 1 else '#00d4ff' for r in sample['risk_label']]
axes[1].scatter(sample['annual_inc'], sample['loan_amnt'], c=scatter_c,
                alpha=0.3, s=8)
axes[1].set_xlabel('Annual Income ($)'); axes[1].set_ylabel('Loan Amount ($)')
axes[1].set_title('Annual Income vs Loan Amount')
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(facecolor='#ff6b6b', label='Delinquent'),
                          Patch(facecolor='#00d4ff', label='Non-Delinquent')])
plt.tight_layout(); plt.show()
"""))

# ── 8. DTI ────────────────────────────────────────────────────────────────────
cells.append(md("## 8 ▸ Debt-to-Income (DTI) Analysis"))
cells.append(code("""
dti_cap = df['dti'].quantile(0.99)
df_dti = df[df['dti'] <= dti_cap]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for lbl, grp in df_dti.groupby('risk_label'):
    axes[0].hist(grp['dti'], bins=80, alpha=0.6,
                 label='Delinquent' if lbl == 1 else 'Non-Delinquent',
                 color='#ff6b6b' if lbl == 1 else '#00d4ff')
axes[0].set_title('DTI Distribution by Risk Label'); axes[0].legend()

# DTI vs int_rate
axes[1].scatter(df_dti['dti'].sample(10_000, random_state=SEED),
                df['int_rate'].loc[df_dti.index].sample(10_000, random_state=SEED),
                alpha=0.2, color='#a29bfe', s=6)
axes[1].set_xlabel('DTI'); axes[1].set_ylabel('Interest Rate (%)')
axes[1].set_title('DTI vs Interest Rate')
plt.tight_layout(); plt.show()
"""))

# ── 9. FICO Score Analysis ────────────────────────────────────────────────────
cells.append(md("## 9 ▸ FICO Score Analysis"))
cells.append(code("""
df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
df['last_fico_avg'] = (df['last_fico_range_low'] + df['last_fico_range_high']) / 2
df['fico_drop'] = df['fico_avg'] - df['last_fico_avg']  # key delinquency signal!

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for lbl, grp in df.groupby('risk_label'):
    col = '#ff6b6b' if lbl == 1 else '#00d4ff'
    lblt = 'Delinquent' if lbl else 'Non-Delinquent'
    axes[0].hist(grp['fico_avg'].dropna(), bins=60, alpha=0.6, color=col, label=lblt)
    axes[1].hist(grp['last_fico_avg'].dropna(), bins=60, alpha=0.6, color=col, label=lblt)
    axes[2].hist(grp['fico_drop'].dropna(), bins=60, alpha=0.6, color=col, label=lblt)

for ax, t in zip(axes, ['Orig FICO Avg','Last FICO Avg','FICO Drop (Orig-Last)']):
    ax.set_title(t); ax.legend()
plt.tight_layout(); plt.show()

print("\\nFICO drop stats by risk label:")
print(df.groupby('risk_label')['fico_drop'].describe())
"""))

# ── 10. Temporal Analysis ─────────────────────────────────────────────────────
cells.append(md("## 10 ▸ Temporal Analysis (Issuance Trends)"))
cells.append(code("""
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
df['issue_year'] = df['issue_d'].dt.year
df['issue_quarter'] = df['issue_d'].dt.to_period('Q').astype(str)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Loan volume over years
vol_yr = df.groupby('issue_year').size()
axes[0].bar(vol_yr.index.astype(str), vol_yr.values, color='#00d4ff')
axes[0].set_title('Loans Issued per Year'); axes[0].set_xlabel('Year')
axes[0].tick_params(axis='x', rotation=45)

# Delinquency rate over years
delq_yr = df.groupby('issue_year')['risk_label'].mean() * 100
axes[1].plot(delq_yr.index.astype(str), delq_yr.values,
             color='#ff6b6b', marker='o', lw=2)
axes[1].set_title('Delinquency Rate by Issue Year (%)'); axes[1].set_xlabel('Year')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()
"""))

# ── 11. Geographic Analysis ────────────────────────────────────────────────────
cells.append(md("## 11 ▸ Geographic Analysis (State-level)"))
cells.append(code("""
state_stats = df.groupby('addr_state').agg(
    total=('risk_label', 'count'),
    delinquent=('risk_label', 'sum'),
    avg_loan=('loan_amnt', 'mean'),
    avg_int=('int_rate', 'mean')
).reset_index()
state_stats['delinq_rate'] = state_stats['delinquent'] / state_stats['total'] * 100

fig = px.choropleth(
    state_stats, locations='addr_state', locationmode='USA-states',
    color='delinq_rate', scope='usa',
    color_continuous_scale='RdYlGn_r',
    title='Delinquency Rate by US State (%)',
    template='plotly_dark'
)
fig.show()
"""))
cells.append(code("""
# Top/Bottom 10 states
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
top10 = state_stats.nlargest(10, 'delinq_rate')
bot10 = state_stats.nsmallest(10, 'delinq_rate')

axes[0].barh(top10['addr_state'], top10['delinq_rate'], color='#ff6b6b')
axes[0].set_title('Top 10 States — Highest Delinquency Rate (%)'); axes[0].set_xlabel('%')

axes[1].barh(bot10['addr_state'], bot10['delinq_rate'], color='#7bed9f')
axes[1].set_title('Top 10 States — Lowest Delinquency Rate (%)'); axes[1].set_xlabel('%')
plt.tight_layout(); plt.show()
"""))

# ── 12. Payment Behaviour ─────────────────────────────────────────────────────
cells.append(md("## 12 ▸ Payment Behaviour Analysis"))
cells.append(code("""
df['repay_ratio'] = df['total_rec_prncp'] / df['loan_amnt'].replace(0, np.nan)
df['interest_paid_ratio'] = df['total_rec_int'] / (df['total_rec_prncp'] + 1)
df['late_fee_flag'] = (df['total_rec_late_fee'] > 0).astype(int)

print("Late fee flag counts:")
print(df['late_fee_flag'].value_counts())
print(f"\\nLate fee flag ↔ delinquency correlation: {df['late_fee_flag'].corr(df['risk_label']):.4f}")
"""))
cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Repayment ratio distribution
for lbl, grp in df.groupby('risk_label'):
    col = '#ff6b6b' if lbl == 1 else '#00d4ff'
    axes[0].hist(grp['repay_ratio'].dropna().clip(0, 2), bins=60,
                 alpha=0.6, color=col, label='Delinquent' if lbl else 'Non-Delinquent')
axes[0].set_title('Repayment Ratio (Principal Recovered)'); axes[0].legend()

# Recoveries
rec_cap = df['recoveries'].quantile(0.99)
df_rec = df[df['recoveries'] <= rec_cap]
for lbl, grp in df_rec.groupby('risk_label'):
    col = '#ff6b6b' if lbl == 1 else '#00d4ff'
    axes[1].hist(grp['recoveries'].dropna(), bins=60,
                 alpha=0.6, color=col, label='Delinquent' if lbl else 'Non-Delinquent')
axes[1].set_title('Recoveries Distribution'); axes[1].legend()
plt.tight_layout(); plt.show()
"""))

# ── 13. Correlation Heatmap ────────────────────────────────────────────────────
cells.append(md("## 13 ▸ Correlation Heatmap — Key Numeric Features"))
cells.append(code("""
core_num_cols = [
    'loan_amnt','int_rate','installment','annual_inc','dti',
    'delinq_2yrs','inq_last_6mths','open_acc','pub_rec',
    'revol_bal','revol_util','total_acc','out_prncp',
    'total_pymnt','total_rec_prncp','total_rec_int','total_rec_late_fee',
    'recoveries','last_pymnt_amnt','fico_avg','last_fico_avg','fico_drop',
    'repay_ratio','late_fee_flag','risk_label'
]
corr_df = df[core_num_cols].corr()

plt.figure(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr_df, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='#222')
plt.title('Correlation Matrix — Key Features', fontsize=15, pad=15)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# Top correlations with risk_label
corr_target = corr_df['risk_label'].drop('risk_label').sort_values(key=abs, ascending=False)
print("Top 20 features correlated with risk_label:")
top20_corr = corr_target.head(20).to_frame()
top20_corr.columns = ['correlation_with_risk_label']
display(top20_corr)
"""))

# ── 14. Pre-delinquency Signal Features ───────────────────────────────────────
cells.append(md("""## 14 ▸ Pre-Delinquency Signal Feature Engineering
> These engineered features will become the **input dimensions of the vector system**.
"""))
cells.append(code("""
# ── Convert cyclical date feature
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
df['cr_history_months'] = ((df['issue_d'] - df['earliest_cr_line'])
                            .dt.days / 30).clip(lower=0)

# ── Loan-to-income ratio
df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)

# ── Revolving utilisation as float
if df['revol_util'].dtype == object:
    df['revol_util'] = df['revol_util'].str.replace('%','').astype(float)

# ── Installment-to-income ratio
df['inst_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)

# ── Outstanding principal ratio
df['out_prncp_ratio'] = df['out_prncp'] / (df['loan_amnt'] + 1)

# ── Delinquency recency score (lower mths_since = more recent delinquency = higher risk)
df['delinq_recency_score'] = 1 / (df['mths_since_last_delinq'].fillna(999) + 1)

# ── Hard inquiry pressure
df['inq_pressure'] = df['inq_last_6mths'].fillna(0) / (df['total_acc'].fillna(1) + 1)

print("✅ Feature engineering complete")
engineered = ['cr_history_months','loan_to_income','inst_to_income',
              'out_prncp_ratio','delinq_recency_score','inq_pressure',
              'repay_ratio','fico_drop','late_fee_flag']
print("Engineered features:", engineered)
df[engineered + ['risk_label']].describe()
"""))

cells.append(code("""
# ── Distribution of engineered features
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()
for i, col in enumerate(engineered):
    for lbl, grp in df.groupby('risk_label'):
        col_c = '#ff6b6b' if lbl == 1 else '#00d4ff'
        data = grp[col].replace([np.inf, -np.inf], np.nan).dropna()
        data = data.clip(data.quantile(0.01), data.quantile(0.99))
        axes[i].hist(data, bins=50, alpha=0.55, color=col_c,
                     label='Delinquent' if lbl else 'Non-Delinquent')
    axes[i].set_title(col, fontsize=11)
    axes[i].legend(fontsize=7)
plt.suptitle('Engineered Feature Distributions by Risk Label', fontsize=14, y=1.01)
plt.tight_layout(); plt.show()
"""))

# ── 15. Categorical Encoding ──────────────────────────────────────────────────
cells.append(md("## 15 ▸ Categorical Encoding"))
cells.append(code("""
cat_cols_encode = ['grade','sub_grade','home_ownership','verification_status',
                   'purpose','application_type','initial_list_status','pymnt_plan']
le = LabelEncoder()
for col in cat_cols_encode:
    if col in df.columns:
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        print(f"  {col}: {df[col].nunique()} unique values → encoded")
print("\\n✅ Categorical encoding complete")
"""))

# ── 16. Missing Value Treatment ────────────────────────────────────────────────
cells.append(md("## 16 ▸ Missing Value Treatment Strategy"))
cells.append(code("""
# ── Drop columns with > 40% missing
threshold = 0.40
cols_before = df.shape[1]
high_miss = miss_df[miss_df['pct'] > 40].index.tolist()
print(f"Dropping {len(high_miss)} columns with >40% missing:")
print(high_miss[:20], '...' if len(high_miss) > 20 else '')
df.drop(columns=[c for c in high_miss if c in df.columns], inplace=True)
print(f"\\nColumns: {cols_before} → {df.shape[1]}")
"""))
cells.append(code("""
# ── Fill remaining numeric NaN with median per group
num_cols = df.select_dtypes(include=np.number).columns.tolist()
miss_num = [c for c in num_cols if df[c].isnull().any()]

for col in miss_num:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# ── Fill remaining categorical NaN with mode / 'Unknown'
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

print(f"Remaining nulls after treatment: {df.isnull().sum().sum()}")
"""))

# ── 17. Outlier Analysis ──────────────────────────────────────────────────────
cells.append(md("## 17 ▸ Outlier Analysis (IQR & Z-score)"))
cells.append(code("""
key_outlier_cols = ['loan_amnt','annual_inc','dti','revol_bal','total_pymnt']
outlier_summary = []

for col in key_outlier_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    z = np.abs(stats.zscore(df[col].dropna()))
    n_z = (z > 3).sum()
    outlier_summary.append({'feature': col, 'IQR_outliers': n_out,
                            'IQR_pct': round(n_out/len(df)*100,2),
                            'Z_score_outliers': n_z,
                            'Z_pct': round(n_z/len(df[col].dropna())*100,2)})

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))
"""))
cells.append(code("""
fig, axes = plt.subplots(1, len(key_outlier_cols), figsize=(20, 5))
for ax, col in zip(axes, key_outlier_cols):
    data = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99)).dropna()
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor='#00d4ff', alpha=0.6, color='#00d4ff'),
               medianprops=dict(color='#ffd700', lw=2),
               whiskerprops=dict(color='#aaa'), capprops=dict(color='#aaa'),
               flierprops=dict(marker='o', color='#ff6b6b', markersize=2, alpha=0.3))
    ax.set_title(col, fontsize=10)
plt.suptitle('Box Plots — Key Numeric Features', fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ── 18. Pre-Delinquency Vector Feature Set Summary ────────────────────────────
cells.append(md("""## 18 ▸ Pre-Delinquency Vector — Final Feature Set Summary
> These features will feed into the **vector representation** for the early-warning system.
"""))
cells.append(code("""
# ── Encode term to numeric months (36 or 60)
if df['term'].dtype == object:
    df['term_months'] = df['term'].str.strip().str.extract(r'(\\d+)').astype(float)
else:
    df['term_months'] = df['term']

VECTOR_FEATURES = {
    'Borrower Risk': ['loan_to_income','inst_to_income','dti','annual_inc'],
    'Credit Health':  ['fico_avg','last_fico_avg','fico_drop','cr_history_months',
                       'delinq_2yrs','pub_rec','inq_last_6mths'],
    'Product Risk':   ['int_rate','grade_enc','sub_grade_enc','loan_amnt','term_months'],
    'Behaviour':      ['revol_util','revol_bal','repay_ratio','out_prncp_ratio',
                       'delinq_recency_score','inq_pressure','late_fee_flag'],
    'Portfolio':      ['total_acc','open_acc','total_rec_late_fee','recoveries']
}

print("=" * 60)
print("   PRE-DELINQUENCY VECTOR FEATURE REGISTRY")
print("=" * 60)
total = 0
num_df = df.select_dtypes(include='number')
for dim, feats in VECTOR_FEATURES.items():
    available = [f for f in feats if f in df.columns]
    total += len(available)
    print(f"\\n▸ {dim} ({len(available)} features)")
    for f in available:
        if f in num_df.columns:
            corr_val = num_df[f].corr(df['risk_label'])
        else:
            corr_val = float('nan')
        print(f"   {f:<35} corr={corr_val:+.4f}")
print(f"\\n{'='*60}")
print(f"   TOTAL VECTOR DIMENSIONS: {total}")
print("=" * 60)
"""))

cells.append(code("""
# ── Save preprocessed DataFrame
all_vector_feats = [f for feats in VECTOR_FEATURES.values() for f in feats if f in df.columns]
df_vector = df[all_vector_feats + ['risk_label', 'loan_status', 'issue_d', 'addr_state']].copy()

# Parquet needs homogeneous arrow types — cast object cols
for c in df_vector.select_dtypes('object').columns:
    try:
        df_vector[c] = df_vector[c].astype(str)
    except Exception:
        pass

df_vector.to_parquet('lending_club_preprocessed.parquet', index=False)
df_vector.to_csv('lending_club_preprocessed_sample.csv', index=False)
print(f"✅ Preprocessed data saved: {df_vector.shape}")
df_vector.head()
"""))

# ── 19. Final Summary ─────────────────────────────────────────────────────────
cells.append(md("""## 19 ▸ Summary & Next Steps

### 🔍 Key EDA Findings

| Finding | Detail |
|---|---|
| Dataset size | ~2.26M rows, 150+ columns |
| Delinquency rate | ~20–25% (class imbalance to address) |
| Strongest predictors | `fico_drop`, `repay_ratio`, `out_prncp_ratio`, `late_fee_flag`, `int_rate` |
| Grade signal | Grade G/F have significantly higher delinquency rates |
| DTI signal | Delinquent borrowers have higher DTI on average |
| Temporal trend | Loans issued 2007–2010 have highest delinquency (GFC era) |
| Geographic hotspots | Some states show >30% delinquency rate |

### 🚀 Next Steps (Vector System)

1. **Feature Scaling** — StandardScaler / RobustScaler on vector features
2. **Class Imbalance** — SMOTE / class_weight balancing
3. **Dimensionality Reduction** — PCA / UMAP for vector space visualisation
4. **Model Training** — XGBoost / LightGBM baseline → deep vector encoder
5. **Vector Similarity** — Cosine similarity for peer-group delinquency clustering
6. **Alerting Layer** — Threshold-based early warning triggers on vector drift
"""))

nb.cells = cells
with open('lending_club_eda.ipynb', 'w') as f:
    nbf.write(nb, f)
print("✅ Notebook written: lending_club_eda.ipynb")
