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

# Load master table from Phase 1
master = pd.read_parquet('/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/master_tables/master_table_24_25.parquet')

def median_pace(year, r_laps, gp):
    pace = r_laps.groupby('Driver')['LapTime'].median()
    pace = pd.DataFrame(pace)
    pace = pace.reset_index()
    return pace


years = [2022, 2023, 2024, 2025, 2026]
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
        fp2 = fp2.drop(["PitInTime", "PitOutTime", "LapStartDate"], axis=1)

    return fp2


def team_dif(year, fp2, gp):
    team_df = fp2[['Driver', 'Team']].drop_duplicates()
    pace = median_pace(year, fp2, gp)
    
    teams_dif = pd.merge(team_df, pace, on='Driver', how='inner')
    teams_dif = pd.merge(teams_dif, teams_dif, on='Team', how='inner')

    teams_dif = teams_dif[teams_dif["Driver_x"] != teams_dif["Driver_y"]]
    teams_dif["team_dif"] = teams_dif["LapTime_x"] - teams_dif["LapTime_y"]
    teams_dif = teams_dif.drop(["Driver_y", "LapTime_y", "LapTime_x"], axis=1)
    teams_dif = teams_dif.rename(columns={"Driver_x": "Driver"})

    return teams_dif

def calc_deg_rate(stint_df):
    if len(stint_df) < 4:
        return None
    slope = np.polyfit(stint_df["TyreLife"], stint_df["LapTime"], 1)[0]
    return slope


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

# Main loop to collect all races
all_races = []

for year in years:
    schedule = fastf1.get_event_schedule(year)
    total_rounds = len(schedule[schedule['EventFormat'] != 'testing'])

    for race in range(1, total_rounds + 1):
        try:
            master_table2 = get_fp2_features(year, race)
            all_races.append(master_table2)
            print(f"\n\n{year} round {race}")
        except Exception as e:
            print(f"\n\n{year} round {race}: {e} SOS!")

# Concatenate all races and merge with Phase 1 master table
fp2_all = pd.concat(all_races, ignore_index=True)

master_clean = master.drop(['Median_lap_time', 'team_dif', 'Deg_Rate_Weighted'], axis=1)

master2 = pd.merge(
    master_clean,
    fp2_all,
    on=['Driver', 'Year', 'Location'],
    how='left'
)

print(master2)
