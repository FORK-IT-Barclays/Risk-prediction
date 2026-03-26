
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = 'accepted_2007_to_2018Q4.csv'
SAMPLE_N   = 5_000   # rows to sample for heatmap (more = slower)
SEED       = 42

# ── Load sample ──────────────────────────────────────────────────────────────
print("⏳ Loading sample …")
df = pd.read_csv(DATA_PATH, nrows=50_000, low_memory=False)
df = df.sample(SAMPLE_N, random_state=SEED).reset_index(drop=True)
print(f"✅ Loaded {df.shape}")

# ── Keep only columns that actually have nulls ───────────────────────────────
miss_pct = df.isnull().mean() * 100
null_cols = miss_pct[miss_pct > 0].sort_values(ascending=False)
print(f"Columns with nulls: {len(null_cols)}")

df_null = df[null_cols.index]   # only null-containing columns

# ── Plot 1: Null Heatmap (sample rows × columns) ───────────────────────────
fig, ax = plt.subplots(figsize=(22, 8))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

cmap = mcolors.ListedColormap(['#1e2130', '#ff6b6b'])  # present=dark, null=red

sns.heatmap(
    df_null.isnull(),
    cmap=cmap,
    cbar=True,
    yticklabels=False,
    xticklabels=True,
    ax=ax,
    linewidths=0,
)
ax.set_title(f'Null Value Heatmap  |  {SAMPLE_N:,} sampled rows × {len(null_cols)} columns with nulls',
             fontsize=14, color='white', pad=14)
ax.tick_params(axis='x', colors='#aaa', labelsize=6, rotation=90)
cbar = ax.collections[0].colorbar
cbar.set_ticks([0.25, 0.75])
cbar.set_ticklabels(['Present', 'Missing'])
cbar.ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig('null_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("✅ Saved: null_heatmap.png")

# ── Plot 2: Missing % bar chart (sorted) ────────────────────────────────────
# Use FULL dataset miss % for accuracy
print("⏳ Computing full-dataset missing % …")
df_full = pd.read_csv(DATA_PATH, nrows=200_000, low_memory=False)
full_miss = df_full.isnull().mean() * 100
full_miss = full_miss[full_miss > 0].sort_values(ascending=False)

fig2, axes = plt.subplots(1, 2, figsize=(22, 9))
fig2.patch.set_facecolor('#0f1117')
for ax in axes:
    ax.set_facecolor('#0f1117')
    ax.tick_params(colors='#aaa')
    ax.spines[:].set_color('#444')

# All null cols bar
colors_bar = ['#ff6b6b' if p > 50 else '#ffd700' if p > 20 else '#00d4ff'
              for p in full_miss.values]
axes[0].barh(full_miss.index[::-1], full_miss.values[::-1], color=colors_bar[::-1], height=0.7)
axes[0].set_xlabel('Missing %', color='#ccc')
axes[0].set_title('All Columns — Missing % (sorted)', color='white', fontsize=12)
axes[0].axvline(50, color='#ff6b6b', lw=1.2, linestyle='--', label='>50% (drop)')
axes[0].axvline(20, color='#ffd700', lw=1.2, linestyle='--', label='>20% (impute w/ caution)')
axes[0].legend(facecolor='#1e2130', labelcolor='white', fontsize=9)
axes[0].tick_params(axis='y', labelsize=6)

# Pie: how many cols fall in each bucket
buckets = pd.cut(full_miss, bins=[-1, 1, 20, 50, 100],
                 labels=['<1%', '1–20%', '20–50%', '>50%'])
bucket_counts = buckets.value_counts().sort_index()
axes[1].pie(bucket_counts.values,
            labels=bucket_counts.index,
            autopct='%1.0f%%',
            colors=['#00d4ff', '#7bed9f', '#ffd700', '#ff6b6b'],
            wedgeprops={'edgecolor': '#0f1117', 'linewidth': 2},
            textprops={'color': 'white', 'fontsize': 11})
axes[1].set_title('Column Bucketing by Missing %', color='white', fontsize=12)

plt.suptitle('Lending Club — Null Value Analysis', color='white', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('null_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("✅ Saved: null_analysis.png")
plt.show()
print("\nDone!")
