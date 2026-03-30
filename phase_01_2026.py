import fastf1
import pandas as pd
import numpy as np
import pyarrow
import pyarrow.parquet as pq

#######################################
#Loads all 2026 races up to 31/03/2026#
######################################

locations_list = ["Melbourne", "Shanghai", "Suzuka", "Sakhir", "Jeddah",
             "Miami", "Imola", "Monaco", "Barcelona", "Montreal", "Spielberg",
             "Silverstone", "Spa", "Budapest", "Zandvoort", "Monza", "Baku",
             "Singapore", "Austin", "Mexico City", "Sao Paolo", "Las Vegas", "Lusail", "Yas Marina"]

years = [2026]

all_races = []

#######################
#Get race results data#
#######################

def get_race_results(year, gp):
    r_results = fastf1.get_session(year, gp, 'R')
    r_results.load()
    r_results = r_results.results
    r_results['Time'] = r_results['Time'].dt.total_seconds()
    r_results = r_results.drop(["DriverNumber", "DriverId", "HeadshotUrl", "TeamColor", "FirstName", "LastName", "BroadcastName", "Q1", "Q2", "Q3"], axis= 1)
    r_results.loc[r_results['Position'] == 1, 'Time'] = 0
    
    #r_results.to_parquet(f'/Users/dimkostir/Desktop/Projects/F1/data/processed/races/{year}_{gp}_results.parquet')

    return r_results

#######################
#Get qual results data#
#######################

def get_qual_results(year, gp):
    q_results = fastf1.get_session(year,gp,'Q')
    q_results.load()
    q_results = q_results.results
    q_results = q_results.drop(["HeadshotUrl", "Points", "Laps", "GridPosition", "Status", "Points", "Laps","TeamColor", "DriverNumber"], axis= 1)
    q_results["Q1"] = q_results["Q1"].dt.total_seconds()
    q_results["Q2"] = q_results["Q2"].dt.total_seconds()
    q_results["Q3"] = q_results["Q3"].dt.total_seconds()
    q_results["Best_Q"] = q_results["Q3"].fillna(q_results["Q2"]).fillna(q_results["Q1"])
   
    best_q = q_results["Best_Q"].min()
    q_results["delta_to_pole"] = q_results["Best_Q"] - best_q

    #q_results.to_parquet(f'/Users/dimkostir/Desktop/Projects/F1/data/processed/qual/qual_{year}_{gp}_results.parquet')

    return q_results

#################
#Lap Median Pace#
#################

def median_pace(year,r_laps, gp):
  pace = r_laps.groupby('Driver')['LapTime'].median()
  pace = pd.DataFrame(pace)
  #pace.to_parquet(f'/Users/dimkostir/Desktop/Projects/F1/data/processed/median_pace/median_pace_{gp}_{year}.parquet')
  pace = pace.reset_index()
  return pace


####################
#Get race laps data#
####################

def get_r_laps(year,gp):
    r_laps = fastf1.get_session(year, gp, 'R')
    r_laps.load()
    r_laps = r_laps.laps

    r_laps = r_laps[
     (r_laps["TrackStatus"] == "1") &
     (r_laps["LapTime"].notna()) &
     (r_laps["PitInTime"].isna()) &
     (r_laps["PitOutTime"].isna())].copy()
    
    r_laps['Time'] = r_laps['Time'].dt.total_seconds()
    r_laps['LapTime'] = r_laps['LapTime'].dt.total_seconds()
    r_laps['LapStartTime'] = r_laps['LapStartTime'].dt.total_seconds()
    r_laps['Sector1Time'] = r_laps['Sector1Time'].dt.total_seconds()
    r_laps['Sector2Time'] = r_laps['Sector2Time'].dt.total_seconds()
    r_laps['Sector3Time'] = r_laps['Sector3Time'].dt.total_seconds()
    r_laps['Sector1SessionTime'] = r_laps['Sector1SessionTime'].dt.total_seconds()
    r_laps['Sector2SessionTime'] = r_laps['Sector2SessionTime'].dt.total_seconds()
    r_laps['Sector3SessionTime'] = r_laps['Sector3SessionTime'].dt.total_seconds()
    r_laps = r_laps.drop(["PitInTime", "PitOutTime"], axis= 1)
    
    median_pace(year, r_laps, gp)
    
    #r_laps.to_parquet(f'/Users/dimkostir/Desktop/Projects/F1/data/processed/r_laps/r_laps_{gp}_{year}.parquet')
    
    return r_laps


######################################
#Def for teams difference calculation#
######################################

def team_dif(year, r_laps, gp):
    team_df = r_laps[['Driver', 'Team']].drop_duplicates()
    pace = median_pace(year, r_laps, gp)
    
    teams_dif = pd.merge(team_df, pace, on='Driver', how='inner')
    teams_dif = pd.merge(teams_dif, teams_dif, on='Team', how='inner')

    teams_dif = teams_dif[teams_dif["Driver_x"] != teams_dif["Driver_y"]]
    teams_dif["team_dif"] = teams_dif["LapTime_x"] - teams_dif["LapTime_y"]
    teams_dif = teams_dif.drop(["Driver_y", "LapTime_y", "LapTime_x"], axis=1)
    teams_dif = teams_dif.rename(columns={"Driver_x": "Driver"})

    #teams_dif.to_parquet(f'/Users/dimkostir/Desktop/Projects/F1/data/processed/teams_dif/teams_dif_{gp}_{year}.parquet')

    return teams_dif

######################################
#Def for degradation rate calculation#
######################################

def calc_deg_rate(stint_df):
    stint_df = stint_df[stint_df["TyreLife"] > 1]
    if len(stint_df) < 4:
        return None
    slope = np.polyfit(stint_df["TyreLife"], stint_df["LapTime"], 1)[0]
    return slope


def deg_rate(r_laps, year, gp):
    r_laps_stint2 = r_laps[r_laps["Stint"] > 1]
    degradation_rate = r_laps_stint2.groupby(["Driver", "Stint"]).apply(calc_deg_rate, include_groups=False)
    degradation_rate = pd.DataFrame(degradation_rate).reset_index()
    degradation_rate = degradation_rate.rename(columns={0: "Deg_Rate"})
    #degradation_rate.to_parquet(f'/Users/dimkostir/Desktop/Projects/F1/data/processed/deg_rate/deg_rate_{year}_{gp}.parquet')
    return degradation_rate

def d_rate_final(r_laps, year, gp):
    stint = r_laps.groupby(["Driver", "Stint"]).size().reset_index()
    stint = stint.rename(columns={0: "Laps"})
    deg = deg_rate(r_laps, year, gp)
    stint_df = pd.merge(stint, deg, on=['Driver', 'Stint'], how='inner')
    merged_df_clean = stint_df[stint_df["Deg_Rate"].notna()]
    result = merged_df_clean.groupby("Driver").apply(
        lambda x: (x["Deg_Rate"] * x["Laps"]).sum() / x["Laps"].sum(), include_groups=False
    ).reset_index()
    result = result.rename(columns={0: "Deg_Rate_Weighted"})
    return result

#############################   
#Get qualification laps data#
#############################

def get_q_laps(year, gp):
    q_laps = fastf1.get_session(year, gp, 'Q')
    q_laps.load(telemetry=False, weather=False, messages=False)
    q_laps = q_laps.laps.copy()

    q_laps = q_laps[q_laps["IsAccurate"] == True]

    # Timedelta conversions
    for col in ['Time', 'LapTime', 'PitOutTime', 'PitInTime', 'LapStartTime',
                'Sector1Time', 'Sector2Time', 'Sector3Time',
                'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime']:
        q_laps[col] = q_laps[col].dt.total_seconds()

    q_laps = q_laps.drop(["LapStartDate"], axis=1)
    #q_laps.to_parquet(f'/Users/dimkostir/Desktop/Projects/F1/data/processed/q_laps/q_laps_{year}_{gp}.parquet')
    
    return q_laps

def circuit_location(year,gp):
   location = fastf1.get_event(year, gp)["Location"]
   return location

##############
#Master Table#
##############

def master_race(year, gp):
   qual_results = get_qual_results(year, gp)
   race_results = get_race_results(year,gp)
   r_laps = get_r_laps(year, gp)

   pace = median_pace(year,r_laps, gp)
   dif = team_dif(year, r_laps,gp)
   deg =  d_rate_final(r_laps, year, gp)
   
   race_results["Driver"] = race_results["Abbreviation"]

   master = race_results
   master = pd.merge(master, pace, on="Driver", how="left")
   master = pd.merge(master, dif, on="Driver", how="left")
   master = pd.merge(master, deg, on="Driver", how="left")
   master = pd.merge(master, qual_results, left_on ="Driver", right_on = "Abbreviation",how="left")

   master["Year"] = year
   master["Location"] = circuit_location(year,gp)

   master = master.rename(columns={"Position_x": "Finish_Position"})
   master = master.rename(columns={"Position_y": "Qual_Position"})
   master = master.rename(columns={"LapTime": "Median_lap_time"})
   master = master.drop(["Abbreviation", "CountryCode", "TeamId", "Time", "Points"],axis =1, errors = 'ignore')

   return master

for year in years:
  
  schedule = fastf1.get_event_schedule(year)
  total_rounds = len(schedule[schedule['EventFormat'] != 'testing'])

  for race in range(1,4):
    try:
         master_table = master_race(year, race)
         all_races.append(master_table)
         print(f"\n\n{year} round {race}")
    except Exception as e:
         print(f"\n\n{year} round {race}: {e} SOS!")

if all_races:
    master_table = pd.concat(all_races, ignore_index=True)
    keep_columns = ["Driver", "TeamName_x", "GridPosition",
    "Finish_Position", "Status", "Median_lap_time",
    "team_dif", "Deg_Rate_Weighted",
    "delta_to_pole", "Qual_Position",
    "Year", "Location"]

    master_table = master_table[keep_columns]
    master_table["dnf"] = master_table["Status"].isin(["Retired", "Did not start", "W", "D", "E", "N", "F"])
    master_table.to_csv(f'/Users/dimkostir/Desktop/Projects/f1-predictions/data/csv/master_26_p1.csv')
    master_table.to_parquet(f'/Users/dimkostir/Desktop/Projects/f1-predictions/data/processed/master_tables/master_26_p1.parquet')
else:
    print("\n\nNo races loaded!!!")

print("\n\nCode Executed. Check the data.\n\n")