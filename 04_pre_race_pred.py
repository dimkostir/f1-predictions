import fastf1
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import shap
import matplotlib.pyplot as plt
import time

# Load master table from Phase 1
master = pd.read_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/master_tables/master_26_p1.parquet')

def median_pace(year, r_laps, gp):
    pace = r_laps.groupby('Driver')['LapTime'].median()
    pace = pd.DataFrame(pace)
    pace = pace.reset_index()
    return pace


years = [2026]

locations_list = ["Melbourne", "Shanghai", "Suzuka", "Sakhir", "Jeddah",
             "Miami", "Imola", "Monaco", "Barcelona", "Montreal", "Spielberg",
             "Silverstone", "Spa", "Budapest", "Zandvoort", "Monza", "Baku",
             "Singapore", "Austin", "Mexico City", "Sao Paolo", "Las Vegas", "Lusail", "Yas Marina"]

def get_fp2_laps(year, gp):
    fp2 = fastf1.get_session(year, gp, 'FP2')
    fp2.load()
    fp2 = fp2.laps

    fp2 = fp2[
        (fp2["TrackStatus"] == "1") &
        (fp2["LapTime"].notna()) &
        (fp2["PitInTime"].isna()) &
        (fp2["PitOutTime"].isna())].copy()

    # Timedelta conversions
    for col in ['Time', 'LapTime', 'LapStartTime',
                'Sector1Time', 'Sector2Time', 'Sector3Time',
                'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime']:
        fp2[col] = fp2[col].dt.total_seconds()
    
    # Drop columns AFTER the loop - not inside it
    fp2 = fp2.drop(["PitInTime", "PitOutTime", "LapStartDate"], axis=1)

    return fp2


def team_dif(year, fp2, gp):
    team_df = fp2[['Driver', 'Team']].drop_duplicates()
    pace = median_pace(year, fp2, gp)
    
    teams_dif = pd.merge(team_df, pace, on='Driver', how='inner')
    teams_dif = pd.merge(teams_dif, teams_dif, on='Team', how='inner')

    teams_dif = teams_dif[teams_dif["Driver_x"] != teams_dif["Driver_y"]].copy()
    teams_dif["team_dif"] = teams_dif["LapTime_x"] - teams_dif["LapTime_y"]
    teams_dif = teams_dif.drop(["Driver_y", "LapTime_y", "LapTime_x"], axis=1)
    teams_dif = teams_dif.rename(columns={"Driver_x": "Driver"})

    return teams_dif

def calc_deg_rate(stint_df):
    if len(stint_df) < 4:
        return None
    
    try:
        slope = np.polyfit(stint_df["TyreLife"], stint_df["LapTime"], 1)[0]
        return slope
    except (np.linalg.LinAlgError, ValueError, np.RankWarning):
        # SVD convergence failure - return NaN
        return None


def deg_rate(fp2_laps, year, gp):
    fp2_stint2 = fp2_laps
    degradation_rate = fp2_stint2.groupby(["Driver", "Stint"]).apply(calc_deg_rate, include_groups=False)
    degradation_rate = pd.DataFrame(degradation_rate).reset_index()
    degradation_rate = degradation_rate.rename(columns={0: "Deg_Rate"})
    return degradation_rate

def d_rate_final(fp2, year, gp):
    stint = fp2.groupby(["Driver", "Stint"]).size().reset_index()
    stint = stint.rename(columns={0: "Laps"})
    deg = deg_rate(fp2, year, gp)
    stint_df = pd.merge(stint, deg, on=['Driver', 'Stint'], how='inner')
    merged_df_clean = stint_df[stint_df["Deg_Rate"].notna()]
    result = merged_df_clean.groupby("Driver").apply(
        lambda x: (x["Deg_Rate"] * x["Laps"]).sum() / x["Laps"].sum(), include_groups=False
    ).reset_index()
    result = result.rename(columns={0: "Deg_Rate_Weighted"})
    return result

def circuit_location(year, gp):
    location = fastf1.get_event(year, gp)["Location"]
    return location

def get_fp2_features(year, gp):
    fp2_laps = get_fp2_laps(year, gp)
    fp2_pace = median_pace(year, fp2_laps, gp)
    fp2_dif = team_dif(year, fp2_laps, gp)
    fp2_deg = d_rate_final(fp2_laps, year, gp)

    fp2_features = pd.merge(fp2_pace, fp2_dif, on='Driver', how='outer')
    fp2_features = pd.merge(fp2_features, fp2_deg, on='Driver', how='outer')
    
    fp2_features = fp2_features.drop('Team', axis=1) 
    fp2_features["Year"] = year
    fp2_features["Location"] = circuit_location(year, gp)

    fp2_features = fp2_features.rename(columns={
        'LapTime': 'fp2_median_pace',
        'team_dif': 'fp2_team_dif',
        'Deg_Rate_Weighted': 'fp2_deg_rate'
    })
    return fp2_features

# Main loop to collect all races with error handling
all_races = []
success_count = 0
fail_count = 0

for year in years:
    schedule = fastf1.get_event_schedule(year)
    total_rounds = len(schedule[schedule['EventFormat'] != 'testing'])

    for race in range(1, 4):
        try:
            master_table2 = get_fp2_features(year, race)
            all_races.append(master_table2)
            success_count += 1
            print(f"✓ {year} round {race} — Success ({success_count} total)")
            time.sleep(8)  # 8 seconds delay between races
            
        except Exception as e:
            error_type = type(e).__name__
            fail_count += 1
            
            if "RateLimitExceededError" in error_type:
                print(f"\n⏰ Rate limit at {year} round {race}")
                print("Sleeping 1 hour...")
                time.sleep(3600)
                
                # Retry
                print(f"Retrying {year} round {race}...")
                try:
                    master_table2 = get_fp2_features(year, race)
                    all_races.append(master_table2)
                    success_count += 1
                    print(f"✓ {year} round {race} (retry OK)")
                except Exception as retry_error:
                    print(f"✗ {year} round {race} failed after retry: {retry_error}")
            else:
                print(f"✗ {year} round {race}: {e}")

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY: {success_count} succeeded, {fail_count} failed")
print(f"Total races in all_races: {len(all_races)}")
print(f"{'='*60}\n")


if len(all_races) == 0:
    print("ERROR: No races loaded. Check errors above.")
else:
    # Concatenate all races and merge with Phase 1 master table
    fp2_all = pd.concat(all_races, ignore_index=True)
    print(f"FP2 features shape: {fp2_all.shape}")
    
    # Save FP2 features before merge
    fp2_all.to_parquet('data/processed/fp2_features_2022_2023.parquet')
    print("✅ FP2 features saved!")
    
    master2 = pd.merge(
        master,
        fp2_all,
        on=['Driver', 'Year', 'Location'],
        how='left'
    )
    
    print(f"Master2 shape: {master2.shape}")
    print(f"\nNull counts in FP2 features:")
    print(master2[['fp2_median_pace', 'fp2_team_dif', 'fp2_deg_rate']].isnull().sum())
    
    # Save the result
    master2.to_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/master_tables/master_table_phase2_26.parquet')
    print("\n✅ Phase 2 master table saved! (Parquet)")
    master2.to_csv('/Users/dimkostir/Desktop/Projects/f1-predictions/data/csv/master_table_phase2_26.csv')
    print("\n✅ Phase 2 master table saved! (CSV)")
