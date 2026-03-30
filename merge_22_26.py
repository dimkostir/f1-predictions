import pandas as pd

print("Loading Phase 2 master tables (2022-2026, current date 31/03/2026)...")

# Load all Phase 2 datasets
master_22_23 = pd.read_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/master_tables/master_table_phase2_22_23.parquet')
master_24_25 = pd.read_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/master_tables/master_table_phase2_24_25.parquet')
master_26 = pd.read_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/master_tables/master_table_phase2_26.parquet')

print(f"  2022-2023: {master_22_23.shape}")
print(f"  2024-2025: {master_24_25.shape}")
print(f"  2026:      {master_26.shape}")

# Combine all years
master_all = pd.concat([master_22_23, master_24_25, master_26], ignore_index=True)

print(f"\n{'='*60}")
print(f"COMBINED DATASET")
print(f"{'='*60}")
print(f"Shape: {master_all.shape}")
print(f"Years: {sorted(master_all['Year'].unique())}")
print(f"\nRaces per year:")
print(master_all.groupby('Year')['Location'].nunique())

# Drop Phase 1 race-based features + dnf (data leakage)
print(f"\nDropping data leakage columns...")
master_clean = master_all.drop(['Median_lap_time', 'team_dif', 'Deg_Rate_Weighted', 'dnf'], axis=1)

print(f"\n{'='*60}")
print(f"FINAL CLEAN DATASET")
print(f"{'='*60}")
print(f"Shape: {master_clean.shape}")
print(f"Columns: {list(master_clean.columns)}")
print(f"\nNull counts in FP2 features:")
print(master_clean[['fp2_median_pace', 'fp2_team_dif', 'fp2_deg_rate']].isnull().sum())
print(f"\nNull counts in qualifying features:")
print(master_clean[['delta_to_pole', 'Qual_Position', 'GridPosition']].isnull().sum())

# Save
output_parquet = '/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/final_phase_01/final_phase_01.parquet'
output_csv = '/Users/dimkostir/Desktop/Projects/f1-predictions/data/csv/final_phase_01/final_phase_01.csv'

master_clean.to_parquet(output_parquet)
master_clean.to_csv(output_csv)

print(f"\n✅ FINAL master table saved!")
print(f"   Parquet: {output_parquet}")
print(f"   CSV:     {output_csv}")
print(f"\n🚀 Ready for training!")
print(f"\nFeatures available for model:")
print(f"  - GridPosition")
print(f"  - delta_to_pole")
print(f"  - Qual_Position")
print(f"  - fp2_median_pace")
print(f"  - fp2_team_dif")
print(f"  - fp2_deg_rate")
print(f"\nTarget: Finish_Position")