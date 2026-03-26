# %% [markdown]
# # Exploratory Data Analysis: PKDD'99 Berka Dataset
# This script uses standard `# %%` cell markers so you can run it interactively 
# in the VS Code Interactive Window, exactly like a Jupyter Notebook (`.ipynb`).

# %% 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Try importing seaborn for better visuals, but fallback to matplotlib if not installed
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    print("Seaborn not installed, using standard matplotlib.")

# Configure pandas display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

print("Loading Data...")
loan = pd.read_csv(os.path.join(data_dir, "loan.csv"), sep=";")
account = pd.read_csv(os.path.join(data_dir, "account.csv"), sep=";")
client = pd.read_csv(os.path.join(data_dir, "client.csv"), sep=";")
disp = pd.read_csv(os.path.join(data_dir, "disp.csv"), sep=";")
trans = pd.read_csv(os.path.join(data_dir, "trans.csv"), sep=";", low_memory=False)

print("Data Loaded Successfully!")

# %% [markdown]
# ## 1. Check for Missing Values & Anomalies in Core Tables
# Are there any unexpected Nulls in out primary entity tables?

# %%
print("--- Missing Values in LOAN ---")
print(loan.isnull().sum())

print("\n--- Missing Values in DISP ---")
print(disp.isnull().sum())

print("\n--- Missing Values in Transactions ---")

a = trans.isna().sum()
print(a[a>0])
print(trans.columns)

# Ensure primary keys are completely unique
print("\n--- Uniqueness Check ---")
print(f"Total Loans: {len(loan)}, Unique Loan IDs: {loan['loan_id'].nunique()}, Unique Account IDs: {loan['account_id'].nunique()}")

# %% [markdown]
# ## 2. Target Variable Analysis (Loan Statuses)
# Let's see the exact distribution of what we are trying to predict.
# A = finished, paid  | C = running, OK
# B = finished, unpaid | D = running, in debt

# %%
status_counts = loan['status'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
status_counts.plot(kind='bar', color=['forestgreen', 'firebrick', 'lightgreen', 'salmon'])
plt.title('Distribution of Loan Statuses')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
for i, v in enumerate(status_counts):
    plt.text(i, v + 5, str(v), ha='center')
plt.show()

# Binary mapping for Machine Learning:
# Good = 0 (A, C) | Bad = 1 (B, D)
loan['default'] = loan['status'].map({'A': 0, 'C': 0, 'B': 1, 'D': 1})
print("\n--- Binary Default Rate ---")
print(loan['default'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))

# %% [markdown]
# ## 3. Transaction Data (The Behavioral Goldmine)
# Let's filter the 1M+ transactions down to just the 191k belonging to loan accounts
# to check for anomalies like extreme negative balances or crazy transaction amounts.

# %%
loan_account_ids = loan['account_id'].unique()
loan_trans = trans[trans['account_id'].isin(loan_account_ids)].copy()

print(f"Total Transactions for Loan Accounts: {len(loan_trans)}")

# Check for Nulls in the subset of transactions
print("\n--- Missing Values in Transactions (Loan subset) ---")
print(loan_trans.isnull().sum())

# The 'bank' and 'account' fields correctly have nulls for CASH transactions.
# Let's fill them with 'CASH' for better readability later
loan_trans['bank'] = loan_trans['bank'].fillna('CASH')
loan_trans['account'] = loan_trans['account'].fillna('CASH')
loan_trans['k_symbol'] = loan_trans['k_symbol'].fillna('UNKNOWN')

# %% [markdown]
# ## 4. Anomalies in Transaction Values
# Let's plot the distribution of transaction amounts to see if there are 
# impossible values (e.g., negative withdrawals or billions of dollars).

# %%
plt.figure(figsize=(10, 5))
plt.hist(loan_trans['amount'], bins=100, color='royalblue', edgecolor='gray')
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.yscale('log') # Log scale to see outliers properly
plt.show()

# Show the top 5 highest transactions
print("\n--- Top 5 Highest Transactions ---")
print(loan_trans[['amount', 'type', 'operation', 'bank']].sort_values(by='amount', ascending=False).head())

print("\n--- Are there any Negative balances? ---")
neg_balances = loan_trans[loan_trans['balance'] < 0]
print(f"Count of negative balances: {len(neg_balances)}")
if len(neg_balances) > 0:
    print(neg_balances[['account_id', 'date', 'type', 'amount', 'balance']].head())

# %% [markdown]
# ## Conclusion of Initial Pure EDA
# 1. Outstanding Cleanliness: The core tables essentially have 0 missing values for critical IDs.
# 2. Nulls logically explained: `bank` and `account` in `trans` are null mostly because the transaction was purely an ATM/Cash op.
# 3. Class Imbalance Mapping: Creating a binary 0/1 variable gives us the strict 89% vs 11% ratio.
# 4. Outliers: Using the log-scale histogram helps visualize severe outliers.

# %% [markdown]
# ## 5. Loan Amount & Duration Breakdown
# Does the size of the loan or its duration correlate with defaulting?

# %%
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='default', y='amount', data=loan, palette='Set2')
plt.title('Loan Amount vs Default (0=Good, 1=Bad)')
plt.ylabel('Loan Amount')

plt.subplot(1, 2, 2)
sns.boxplot(x='default', y='duration', data=loan, palette='Set2')
plt.title('Loan Duration vs Default')
plt.ylabel('Duration (Months)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Demographic EDA (Age & Gender)
# Let's extract Age and Gender from the `client` table to see who is defaulting.
# In the Czech system, `birth_number` is YYMMDD, but females have +50 added to the month.

# %%
# Get the primary OWNER for each account
owners = disp[disp['type'] == 'OWNER']
loan_clients = pd.merge(loan, owners, on='account_id')
loan_demographics = pd.merge(loan_clients, client, on='client_id')

def extract_gender(b_num):
    return 'F' if int(str(b_num)[2:4]) > 50 else 'M'

def extract_age(b_num):
    year_part = int(str(b_num)[0:2])
    birth_year = 1900 + year_part  # All clients born in the 1900s
    return 1999 - birth_year  # Dataset ends in 1999

loan_demographics['gender'] = loan_demographics['birth_number'].apply(extract_gender)
loan_demographics['age'] = loan_demographics['birth_number'].apply(extract_age)

# Plot Age
plt.figure(figsize=(10, 5))
sns.histplot(data=loan_demographics, x='age', hue='default', multiple='stack', bins=20, palette='Set1')
plt.title('Age Distribution of Loan Holders by Default Status')
plt.xlabel('Age (in 1999)')
plt.show()

# Plot Gender Default Rates
gender_defaults = loan_demographics.groupby('gender')['default'].mean() * 100
plt.figure(figsize=(6, 4))
gender_defaults.plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Default Rate by Gender (%)')
plt.ylabel('Default Rate (%)')
plt.xticks(rotation=0)
for i, v in enumerate(gender_defaults):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
plt.show()

# %% [markdown]
# ## 7. Regional / District Analysis
# Does the district play a role in defaults? Let's look at average salary vs default.

# %%
district = pd.read_csv(os.path.join(data_dir, "district.csv"), sep=";")
# district table has 'A1' as district_id, 'A11' as average salary
district_info = district.rename(columns={'A1': 'district_id', 'A11': 'average_salary'})
# district_id collision: account has district_id, client has district_id. 
# We'll use the client's home district for demographics.
loan_dist = pd.merge(loan_demographics, district_info[['district_id', 'average_salary']], on='district_id')

plt.figure(figsize=(8, 5))
sns.boxplot(x='default', y='average_salary', data=loan_dist, palette='Set3')
plt.title('District Average Salary vs Loan Default')
plt.ylabel("Average Salary of Customer\'s District")
plt.show()

# %% [markdown]
# ## 8. Temporal EDA: Transactions Over Time
# Let's map how transactions are distributed across the years.

# %%
# Convert trans date (YYMMDD) to datetime
# Only grab a 10% sample of loan_trans to speed up plotting
sample_trans = loan_trans.sample(frac=0.1, random_state=42).copy()
sample_trans['date_parsed'] = pd.to_datetime(sample_trans['date'].astype(str), format='%y%m%d')

plt.figure(figsize=(12, 5))
sample_trans.groupby(sample_trans['date_parsed'].dt.to_period('M')).size().plot(color='darkorange')
plt.title('Transaction Volume Over Time (Loan Accounts 10% Sample)')
plt.xlabel('Month-Year')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.show()

# %%
print("\n--- Full EDA Complete! ---")
