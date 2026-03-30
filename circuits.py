import pandas as pd
import numpy as np

print("="*60)
print("BUILDING CIRCUITS.CSV FROM YOUR DATA")
print("="*60)

# ============================================================
# Step 1: Load your Phase 1 data to calculate REAL degradation
# ============================================================
print("\nStep 1: Loading your master tables...")

try:
    # Try 2024-2025 first (cleanest data)
    master = pd.read_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/master_tables/master_table_24_25.parquet')
    print(f"✓ Loaded master_table_24_25.parquet: {master.shape}")
    has_real_data = True
except:
    try:
        # Fallback to final merged table
        master = pd.read_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/final_phase_01/final_phase_01.parquet')
        print(f"✓ Loaded final_phase_01.parquet: {master.shape}")
        has_real_data = True
    except:
        print("✗ Could not load master tables")
        print("Using manual degradation estimates...")
        has_real_data = False
        master = None

# ============================================================
# Step 2: Calculate degradation severity from YOUR data
# ============================================================
if has_real_data and 'Deg_Rate_Weighted' in master.columns:
    print("\nStep 2: Calculating degradation from YOUR race data...")
    
    # Filter out nulls and outliers
    deg_data = master[master['Deg_Rate_Weighted'].notna()].copy()
    
    # Calculate per circuit
    circuit_deg = deg_data.groupby('Location').agg({
        'Deg_Rate_Weighted': ['mean', 'std', 'count']
    }).reset_index()
    
    circuit_deg.columns = ['Location', 'avg_deg_rate', 'std_deg_rate', 'n_races']
    
    print(f"\nDegradation calculated for {len(circuit_deg)} circuits:")
    print(circuit_deg.sort_values('avg_deg_rate', ascending=False)[['Location', 'avg_deg_rate', 'n_races']].head(10))
    
    # Normalize to 0-1
    min_deg = circuit_deg['avg_deg_rate'].min()
    max_deg = circuit_deg['avg_deg_rate'].max()
    circuit_deg['deg_severity'] = (circuit_deg['avg_deg_rate'] - min_deg) / (max_deg - min_deg)
    
else:
    print("\nStep 2: Using manual degradation estimates...")
    circuit_deg = None

# ============================================================
# Step 3: Build circuit features from formula-timer.com data
# ============================================================
print("\nStep 3: Building circuit features...")

circuits_data = {
    'Location': [
        'Sakhir', 'Jeddah', 'Melbourne', 'Shanghai', 'Suzuka',
        'Miami', 'Imola', 'Monaco', 'Barcelona', 'Montreal',
        'Spielberg', 'Silverstone', 'Spa', 'Budapest', 'Zandvoort',
        'Monza', 'Baku', 'Singapore', 'Austin', 'Mexico City',
        'Sao Paolo', 'Las Vegas', 'Lusail', 'Yas Marina'
    ],
    'circuit_length': [
        5412, 6174, 5278, 5451, 5807,
        5412, 4909, 3337, 4675, 4361,
        4318, 5891, 7004, 4381, 4259,
        5793, 6003, 4940, 5513, 4304,
        4309, 6120, 5380, 5281
    ],
    'num_corners': [
        15, 27, 14, 16, 18,
        19, 19, 19, 16, 14,
        10, 18, 19, 14, 14,
        11, 20, 23, 20, 17,
        15, 17, 16, 21
    ],
    'num_drs_zones': [
        3, 3, 3, 2, 2,
        3, 2, 1, 2, 3,
        3, 2, 2, 1, 2,
        2, 2, 3, 2, 3,
        2, 2, 2, 2
    ],
    'avg_speed_kmh': [
        205, 252, 223, 205, 230,
        223, 198, 160, 195, 220,
        237, 240, 235, 195, 210,
        263, 215, 172, 215, 217,
        212, 240, 240, 195
    ],
    'safety_car_rate': [
        0.08, 0.25, 0.15, 0.10, 0.12,
        0.18, 0.20, 0.35, 0.10, 0.22,
        0.15, 0.12, 0.18, 0.15, 0.20,
        0.10, 0.40, 0.50, 0.15, 0.18,
        0.25, 0.20, 0.15, 0.12
    ],
    'track_type': [
        'permanent', 'street', 'permanent', 'permanent', 'permanent',
        'street', 'permanent', 'street', 'permanent', 'semi-permanent',
        'permanent', 'permanent', 'permanent', 'permanent', 'permanent',
        'permanent', 'street', 'street', 'permanent', 'permanent',
        'permanent', 'street', 'permanent', 'permanent'
    ]
}

circuits = pd.DataFrame(circuits_data)

# ============================================================
# Step 4: Calculate overtaking difficulty
# ============================================================
print("\nStep 4: Calculating overtaking difficulty...")

# Formula: (DRS zones × circuit_length / avg_speed) / num_corners
circuits['overtaking_score'] = (
    circuits['num_drs_zones'] * (circuits['circuit_length'] / circuits['avg_speed_kmh']) / 
    circuits['num_corners']
)

# Normalize to 0-1
min_score = circuits['overtaking_score'].min()
max_score = circuits['overtaking_score'].max()
circuits['overtaking_difficulty'] = (circuits['overtaking_score'] - min_score) / (max_score - min_score)

# Manual adjustments for known extremes
circuits.loc[circuits['Location'] == 'Monaco', 'overtaking_difficulty'] = 0.05
circuits.loc[circuits['Location'] == 'Monza', 'overtaking_difficulty'] = 0.85
circuits.loc[circuits['Location'] == 'Spa', 'overtaking_difficulty'] = 0.75
circuits.loc[circuits['Location'] == 'Imola', 'overtaking_difficulty'] = 0.10
circuits.loc[circuits['Location'] == 'Budapest', 'overtaking_difficulty'] = 0.20

circuits = circuits.drop('overtaking_score', axis=1)

# ============================================================
# Step 5: Add degradation severity
# ============================================================
print("\nStep 5: Adding degradation severity...")

if circuit_deg is not None:
    # Use REAL data from your races
    circuits = pd.merge(circuits, circuit_deg[['Location', 'deg_severity']], on='Location', how='left')
    
    # Fill missing with median
    median_deg = circuits['deg_severity'].median()
    circuits['deg_severity'] = circuits['deg_severity'].fillna(median_deg)
    print(f"✓ Using YOUR race data for degradation (filled {circuits['deg_severity'].isna().sum()} missing with median)")
    
else:
    # Fallback: manual estimates
    deg_severity_manual = {
        'Sakhir': 0.8, 'Jeddah': 0.5, 'Melbourne': 0.6, 'Shanghai': 0.5, 'Suzuka': 0.7,
        'Miami': 0.6, 'Imola': 0.4, 'Monaco': 0.3, 'Barcelona': 0.7, 'Montreal': 0.5,
        'Spielberg': 0.6, 'Silverstone': 0.8, 'Spa': 0.7, 'Budapest': 0.8, 'Zandvoort': 0.6,
        'Monza': 0.4, 'Baku': 0.5, 'Singapore': 0.4, 'Austin': 0.7, 'Mexico City': 0.6,
        'Sao Paolo': 0.6, 'Las Vegas': 0.3, 'Lusail': 0.6, 'Yas Marina': 0.5
    }
    circuits['deg_severity'] = circuits['Location'].map(deg_severity_manual)
    print("✓ Using manual degradation estimates")

# ============================================================
# Step 6: Final cleanup and save
# ============================================================
print("\nStep 6: Finalizing...")

# Reorder columns for readability
circuits = circuits[[
    'Location', 'circuit_length', 'num_corners', 'num_drs_zones',
    'avg_speed_kmh', 'track_type', 'overtaking_difficulty',
    'safety_car_rate', 'deg_severity'
]]

print("\n" + "="*60)
print("FINAL CIRCUITS.CSV")
print("="*60)
print(circuits)

print("\nFeature ranges:")
print(f"  circuit_length:         {circuits['circuit_length'].min()}-{circuits['circuit_length'].max()}m")
print(f"  avg_speed_kmh:          {circuits['avg_speed_kmh'].min()}-{circuits['avg_speed_kmh'].max()} km/h")
print(f"  overtaking_difficulty:  {circuits['overtaking_difficulty'].min():.2f}-{circuits['overtaking_difficulty'].max():.2f}")
print(f"  safety_car_rate:        {circuits['safety_car_rate'].min():.2f}-{circuits['safety_car_rate'].max():.2f}")
print(f"  deg_severity:           {circuits['deg_severity'].min():.2f}-{circuits['deg_severity'].max():.2f}")

# Save
output_path = '/Users/dimkostir/Desktop/Projects/f1-predictions/data/csv/circuits_csv/circuits.csv'
circuits.to_csv(output_path, index=False)
circuits.to_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/circuits_parquet/circuits.parquet')

print("\nSaved!")

master