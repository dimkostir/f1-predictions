import pandas as pd
import numpy as np

# Circuit data based on formula-timer.com
# Order matches Location list exactly

circuits_data = {
    'Location': ["Melbourne", "Shanghai", "Suzuka", "Sakhir", "Jeddah",
             "Miami", "Imola", "Monaco", "Barcelona", "Montreal", 
             "Spielberg", "Silverstone", "Spa", "Budapest", "Zandvoort", 
             "Monza", "Baku", "Singapore", "Austin", "Mexico City", "Sao Paolo", 
             "Las Vegas", "Lusail", "Yas Marina"],

    'circuit_length': [
        5278, 5451, 5807, 5412, 6174,  # Melbourne, Shanghai, Suzuka, Sakhir, Jeddah
        5412, 4909, 3337, 4675, 4361,  # Miami, Imola, Monaco, Barcelona, Montreal
        4318, 5891, 7004, 4381, 4259,  # Spielberg, Silverstone, Spa, Budapest, Zandvoort
        5793, 6003, 4940, 5513, 4304,  # Monza, Baku, Singapore, Austin, Mexico City
        4309, 6120, 5380, 5281         # Sao Paolo, Las Vegas, Lusail, Yas Marina
    ],
    'num_corners': [
        14, 16, 18, 15, 27,  # Melbourne, Shanghai, Suzuka, Sakhir, Jeddah
        19, 19, 19, 16, 14,  # Miami, Imola, Monaco, Barcelona, Montreal
        10, 18, 19, 14, 14,  # Spielberg, Silverstone, Spa, Budapest, Zandvoort
        11, 20, 23, 20, 17,  # Monza, Baku, Singapore, Austin, Mexico City
        15, 17, 16, 21       # Sao Paolo, Las Vegas, Lusail, Yas Marina
    ],
    'num_drs_zones': [
        3, 2, 2, 3, 3,  # Melbourne, Shanghai, Suzuka, Sakhir, Jeddah
        3, 2, 1, 2, 3,  # Miami, Imola, Monaco, Barcelona, Montreal
        3, 2, 2, 1, 2,  # Spielberg, Silverstone, Spa, Budapest, Zandvoort
        2, 2, 3, 2, 3,  # Monza, Baku, Singapore, Austin, Mexico City
        2, 2, 2, 2      # Sao Paolo, Las Vegas, Lusail, Yas Marina
    ],
    'avg_speed_kmh': [
        223, 205, 230, 205, 252,  # Melbourne, Shanghai, Suzuka, Sakhir, Jeddah
        223, 198, 160, 195, 220,  # Miami, Imola, Monaco, Barcelona, Montreal
        237, 240, 235, 195, 210,  # Spielberg, Silverstone, Spa, Budapest, Zandvoort
        263, 215, 172, 215, 217,  # Monza, Baku, Singapore, Austin, Mexico City
        212, 240, 240, 195        # Sao Paolo, Las Vegas, Lusail, Yas Marina
    ],
    'safety_car_rate': [
        0.15, 0.10, 0.12, 0.08, 0.25,  # Melbourne, Shanghai, Suzuka, Sakhir, Jeddah
        0.18, 0.20, 0.35, 0.10, 0.22,  # Miami, Imola, Monaco, Barcelona, Montreal
        0.15, 0.12, 0.18, 0.15, 0.20,  # Spielberg, Silverstone, Spa, Budapest, Zandvoort
        0.10, 0.40, 0.50, 0.15, 0.18,  # Monza, Baku, Singapore, Austin, Mexico City
        0.25, 0.20, 0.15, 0.12         # Sao Paolo, Las Vegas, Lusail, Yas Marina
    ],
    'track_type': [
        'permanent', 'permanent', 'permanent', 'permanent', 'street',  # Melbourne, Shanghai, Suzuka, Sakhir, Jeddah
        'street', 'permanent', 'street', 'permanent', 'semi-permanent',  # Miami, Imola, Monaco, Barcelona, Montreal
        'permanent', 'permanent', 'permanent', 'permanent', 'permanent',  # Spielberg, Silverstone, Spa, Budapest, Zandvoort
        'permanent', 'street', 'street', 'permanent', 'permanent',  # Monza, Baku, Singapore, Austin, Mexico City
        'permanent', 'street', 'permanent', 'permanent'  # Sao Paolo, Las Vegas, Lusail, Yas Marina
    ]
}

# Create DataFrame
circuits = pd.DataFrame(circuits_data)


# Manual adjustments (based on F1 knowledge - https://formula-timer.com/circuit)
# Scale: 0 = impossible, 1 = very easy
circuits.loc[circuits['Location'] == 'Monaco', 'overtaking_ability'] = 0.05      # Nearly impossible
circuits.loc[circuits['Location'] == 'Imola', 'overtaking_ability'] = 0.10       # Very hard (narrow)
circuits.loc[circuits['Location'] == 'Singapore', 'overtaking_ability'] = 0.15   # Hard (narrow street)
circuits.loc[circuits['Location'] == 'Budapest', 'overtaking_ability'] = 0.20    # Hard (twisty)
circuits.loc[circuits['Location'] == 'Zandvoort', 'overtaking_ability'] = 0.25   # Hard (narrow, banked)
circuits.loc[circuits['Location'] == 'Barcelona', 'overtaking_ability'] = 0.35   # Medium-hard
circuits.loc[circuits['Location'] == 'Yas Marina', 'overtaking_ability'] = 0.40  # Medium
circuits.loc[circuits['Location'] == 'Suzuka', 'overtaking_ability'] = 0.40      # Medium (fast, technical)
circuits.loc[circuits['Location'] == 'Baku', 'overtaking_ability'] = 0.45        # Medium (long straight but narrow)
circuits.loc[circuits['Location'] == 'Melbourne', 'overtaking_ability'] = 0.45   # Medium
circuits.loc[circuits['Location'] == 'Las Vegas', 'overtaking_ability'] = 0.50   # Medium
circuits.loc[circuits['Location'] == 'Austin', 'overtaking_ability'] = 0.55      # Medium-good
circuits.loc[circuits['Location'] == 'Silverstone', 'overtaking_ability'] = 0.55 # Medium-good (fast corners)
circuits.loc[circuits['Location'] == 'Shanghai', 'overtaking_ability'] = 0.58    # Good (longest DRS straight)
circuits.loc[circuits['Location'] == 'Spielberg', 'overtaking_ability'] = 0.60   # Good (short track, 3 DRS)
circuits.loc[circuits['Location'] == 'Lusail', 'overtaking_ability'] = 0.60      # Good
circuits.loc[circuits['Location'] == 'Jeddah', 'overtaking_ability'] = 0.65      # Good (fast street circuit)
circuits.loc[circuits['Location'] == 'Montreal', 'overtaking_ability'] = 0.65    # Good (Wall of Champions)
circuits.loc[circuits['Location'] == 'Sakhir', 'overtaking_ability'] = 0.70      # Very good (3 DRS zones)
circuits.loc[circuits['Location'] == 'Miami', 'overtaking_ability'] = 0.70       # Very good (wide, 3 DRS)
circuits.loc[circuits['Location'] == 'Mexico City', 'overtaking_ability'] = 0.70 # Very good (altitude, long straight)
circuits.loc[circuits['Location'] == 'Sao Paolo', 'overtaking_ability'] = 0.72   # Very good (sprint races)
circuits.loc[circuits['Location'] == 'Spa', 'overtaking_ability'] = 0.75         # Easy (Kemmel straight)
circuits.loc[circuits['Location'] == 'Monza', 'overtaking_ability'] = 0.85       # Very easy (temple of speed)


# Verify alignment
print("="*60)
print("CIRCUIT DATA WITH OVERTAKING ABILITY")
print("="*60)
print(circuits[['Location', 'circuit_length', 'num_corners', 'avg_speed_kmh', 'overtaking_ability']])

print("\nOvertaking ability range:")
print(f"  Easiest: {circuits.loc[circuits['overtaking_ability'].idxmax(), 'Location']} ({circuits['overtaking_ability'].max():.2f})")
print(f"  Hardest: {circuits.loc[circuits['overtaking_ability'].idxmin(), 'Location']} ({circuits['overtaking_ability'].min():.2f})")

# Save to CSV/Parquet
circuits.to_csv('/Users/dimkostir/Desktop/Projects/f1-predictions/data/circuits_base.csv', index=False)
circuits.to_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/circuits_parquet/circuits_base.parquet', index= False)

new_master = pd.read_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/final_phase_01/final_phase_01.parquet')
new_master = pd.merge(new_master, circuits, on = 'Location', how = 'left')

new_master.to_csv('/Users/dimkostir/Desktop/Projects/f1-predictions/data/csv/final_phase_01/final_phase_01_circuits_added.csv')
new_master.to_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/final_phase_01/final_01_circuits.parquet', index= False)

print(new_master)

print("\n✅ circuits_base.csv created with overtaking_difficulty!")