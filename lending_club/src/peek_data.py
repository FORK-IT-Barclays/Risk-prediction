import pandas as pd
import os
import glob

# Try to read the first 0 rows just to get columns
for f in sorted(glob.glob('e:/Barclays-ForkIT/lending_club/*.csv')):
    name = os.path.basename(f)
    print(f"\n=== {name} ===")
    try:
        size_gb = os.path.getsize(f) / (1024**3)
        print(f"Size: {size_gb:.2f} GB")
        df = pd.read_csv(f, nrows=0, low_memory=False)
        cols = list(df.columns)
        print(f"Total Columns: {len(cols)}")
        print(f"Sample Columns: {cols[:15]}")
    except Exception as e:
        print(f"Error: {e}")
