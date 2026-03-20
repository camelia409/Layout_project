import sqlite3, numpy as np, pandas as pd, sys
sys.path.insert(0, '.')

# Patch to count overlaps by error_type
df = pd.read_parquet('training_data/floor_plan_samples.parquet')

print("=== OVERLAP RATE BY ERROR TYPE ===")
for etype in df['error_type'].unique():
    subset = df[df['error_type'] == etype]
    overlap_rate = subset['viol_overlap'].mean() * 100
    count = len(subset)
    print(f"  {etype:<25} n={count:>6}  overlap={overlap_rate:.1f}%")

print()
print("=== VALID RATE BY ERROR TYPE ===")
for etype in df['error_type'].unique():
    subset = df[df['error_type'] == etype]
    valid_rate = subset['is_valid'].mean() * 100
    print(f"  {etype:<25} valid={valid_rate:.1f}%")

print()
print("=== NBC AREA VIOLATION BY BHK ===")
for bhk in sorted(df['bhk'].unique()):
    subset = df[df['bhk'] == bhk]
    nbc_rate = subset['viol_nbc_area'].mean() * 100
    overlap_rate = subset['viol_overlap'].mean() * 100
    valid_rate = subset['is_valid'].mean() * 100
    print(f"  BHK={bhk}  n={len(subset):>6}  "
          f"nbc_area={nbc_rate:.1f}%  "
          f"overlap={overlap_rate:.1f}%  "
          f"valid={valid_rate:.1f}%")

print()
print("=== VALID RATE BY PLOT SIZE ===")
df['plot_area'] = df['plot_w'] * df['plot_d']
bins = [0, 60, 100, 150, 250, 9999]
labels = ['<60', '60-100', '100-150', '150-250', '>250']
df['size_band'] = pd.cut(df['plot_area'], bins=bins, labels=labels)
for band in labels:
    subset = df[df['size_band'] == band]
    if len(subset) == 0:
        continue
    valid_rate = subset['is_valid'].mean() * 100
    overlap_rate = subset['viol_overlap'].mean() * 100
    print(f"  {band:<12} n={len(subset):>6}  "
          f"overlap={overlap_rate:.1f}%  valid={valid_rate:.1f}%")
