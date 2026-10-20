# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:03:32 2026

@author: harry
"""

"""
Inference pipeline:
    - Pulls live data from FPL API once per week,
    updates dataset and makes predictions on upcoming games.

Retrieve new data from API including:
    Bootstrap-static:
        elements (player list)
        teams (team identifiers)
        events (gameweek information)
    Element-summary: (retrieve by player_id)
        history (gw data by week)
        history-past (previous season data) - only need once
    Fixtures:
        fixture-data (all 380 fixtures)

Convert into pandas dataframe objects

Save raw copies to disc (week-by-week data) - pandas or json

Feature engineering:
    Add fixture data to elements (player_df)
    Add historic statistics to elements
    Add current season statistics
    Calculate new features
    Align features
    
Validate dataset:

Import model
Predict probabilites
Output top n probabilities

"""

import requests
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

""" 
- Data engineering / cleaning
    Add historic performance if applicable (prev season):
        Aggregate last season + per90 across all
        merge datasets groupby player_id left join(?)
    Add fixture data:
        home/away, opponent
    Calculate prev1/3 and current season stats (offensive)
    Add team_id to match_logs by using player_id as key:
        aggregate max (defensive stats)
    
    Potential problem where players transfer out of league
    after match logs - creates extra processing or invalid
    values after merging with players df.

"""  

def get_json(url, timeout=30):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()
    
def save_json(data, name, current_time):
    filepath = os.path.join("data", "json", f"{name}_{current_time}.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
def save_csv(df, name, current_time):
    filepath = os.path.join("data", "csv", name, f"{name}_{current_time}.csv")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath}")
    
# Function for last game and last3 games data

def previous_rolling_sum(x, interval):
    """Returns a rolling sum of previous rows based on interval"""
    return x.shift(1).rolling(interval, min_periods=1).sum()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")


fixtures = get_json("https://fantasy.premierleague.com/api/fixtures/")
save_json(fixtures, "fixtures", current_time)

bootstrap = get_json("https://fantasy.premierleague.com/api/bootstrap-static")
save_json(bootstrap, "bootstrap", current_time)

# Convert to pandas dataframe
events = pd.DataFrame(bootstrap['events'])
teams = pd.DataFrame(bootstrap['teams'])
players = pd.DataFrame(bootstrap['elements'])
fixtures = pd.DataFrame(fixtures)

# Retrieve season histories for all players
all_histories = []
match_logs = []

for pid in players['id']:
    url = f"https://fantasy.premierleague.com/api/element-summary/{pid}/"
    data = get_json(url)
    
    df = pd.DataFrame(data['history'])
    match_logs.append(df)
    
    if 'history_past' in data:
        df = pd.DataFrame(data['history_past'])
        df['player_id'] = pid
        all_histories.append(df)

season_histories = pd.concat(all_histories, ignore_index=True)
match_logs = pd.concat(match_logs, ignore_index=True)

# Save csv files
dfs = {
    "events" : events,
    "teams" : teams,
    "players": players,
    "season_histories": season_histories,
    "match_logs": match_logs
    }
    
for name, df in dfs.items():
    save_csv(df, name, current_time)
    
# Change dtypes for operations (xg is listed as string)
season_histories['expected_goals'] = season_histories['expected_goals'].astype(float)
# Ensure rows sorted correctly
season_histories = season_histories.sort_values(by=['player_id', 'season_name'])

# Add player histories if exist
# total variables are the sum of the players stats across all seasons up to and including the given season (24/25)
# last1 variables are the sum of the players stats from the most recent season in their dataset (mostly 24/25)

aggregated_histories = season_histories.groupby('player_id').agg(
    total_goals = ('goals_scored', 'sum'),
    total_xg = ('expected_goals', 'sum'),
    total_mins = ('minutes', 'sum'),
    last_ssn_goals = ('goals_scored', 'last'),
    last_ssn_xg = ('expected_goals', 'last'),
    last_ssn_mins = ('minutes', 'last')
).reset_index()

# Create per90 statistics from historic statistics

aggregated_histories['last_ssn_xgper90'] = (
    aggregated_histories['last_ssn_xg'] / aggregated_histories['last_ssn_mins'].replace(0, np.nan) * 90
)

aggregated_histories['last_ssn_glsper90'] = (
    aggregated_histories['last_ssn_goals'] / aggregated_histories['last_ssn_mins'].replace(0, np.nan) * 90
)

aggregated_histories['career_xgper90'] = (
    aggregated_histories['total_xg'] / aggregated_histories['total_mins'].replace(0, np.nan) * 90
)

aggregated_histories['career_glsper90'] = (
    aggregated_histories['total_goals'] / aggregated_histories['total_mins'].replace(0, np.nan) * 90
)

# Merge histories onto player dataset

main_df = pd.merge(
    players,
    aggregated_histories,
    left_on = ['id'],
    right_on = ['player_id']
)

# Add fixture data
# Events - is_next use variable to establish gameweek

match_state = events.query("is_next == True")
upcoming_gw = match_state['id'].item()

# Add fixture data

upcoming_fixtures = fixtures.query(f"event == {upcoming_gw}")

# Create a home dataframe
home_fixtures = upcoming_fixtures[['event', 'team_h', 'team_a', 'team_h_difficulty']].copy()
home_fixtures['is_home'] = True
home_fixtures = home_fixtures.rename(columns={
    'team_h' : 'team',
    'team_a' : 'opponent',
    'team_h_difficulty' : 'fdr'
})

# Create an away dataframe
away_fixtures = upcoming_fixtures[['event', 'team_a', 'team_h', 'team_a_difficulty']].copy()
away_fixtures['is_home'] = False
away_fixtures = away_fixtures.rename(columns={
    'team_a' : 'team',
    'team_h' : 'opponent',
    'team_a_difficulty' : 'fdr'
})

clean_fixtures = pd.concat([home_fixtures, away_fixtures], ignore_index=True)

main_df = pd.merge(
    main_df,
    clean_fixtures,
    left_on = ['team'],
    right_on = ['team']
)

# Add match logs

# Check for duplicates
print(f"Exact duplicates: {match_logs.duplicated().sum():,}")

match_logs['kickoff_time'] = pd.to_datetime(match_logs['kickoff_time'])

# Sort values to ensure integrity of fixture order (dgw and kickoff_time)
match_logs = match_logs.sort_values(by=['element', 'round', 'kickoff_time'])

# Check dtypes
print(match_logs['element'].dtype)
print(players['id'].dtype)

print(match_logs['element'].isin(players['id']).sum())

match_logs_team = pd.merge(
    match_logs,
    players[['team', 'id']],
    left_on='element',
    right_on='id',
    how='left')

# Testing for NaN values of players who left the league
# match_logs likely drops previous logs of players who left from database
p_31 = players.loc[players['id'] == 31]

match_logs_team['team'].isna().sum()

test = match_logs_team.loc[
    match_logs_team['team'].isna(),
    'id'
].unique()


defence_stats = match_logs_team.groupby(['team', 'round'], as_index=False).agg({
    'goals_conceded': 'max',
    'expected_goals_conceded': 'max'})

defence_stats['gc_last1'] = (
    defence_stats.groupby('team')['goals_conceded']
    .transform(lambda x: previous_rolling_sum(x, 1))
)

defence_stats['gc_last3'] = (
    defence_stats.groupby('team')['goals_conceded']
    .transform(lambda x: previous_rolling_sum(x, 3))
)

defence_stats['xgc_last1'] = (
    defence_stats.groupby('team')['expected_goals_conceded']
    .transform(lambda x: previous_rolling_sum(x, 1))
)

defence_stats['xgc_last3'] = (
    defence_stats.groupby('team')['expected_goals_conceded']
    .transform(lambda x: previous_rolling_sum(x, 3))
)

# Check dtype
defence_stats['expected_goals_conceded'] = defence_stats['expected_goals_conceded'].astype(float)
print(defence_stats.dtypes)


defence_stats['season_xgc'] = (
    defence_stats.groupby('team')['expected_goals_conceded']
        .transform(lambda x: x.shift(1).cumsum())
)

defence_stats['season_gc'] = (
    defence_stats.groupby('team')['goals_conceded']
        .transform(lambda x: x.shift(1).cumsum())
)

# Only select latest round
def_stats_last = defence_stats.groupby(
    'team').tail(1)

# Merge main dataframe and opponent dataframe with keys next_opp and gameweek

main_df = main_df.merge(
    def_stats_last[['team','round','gc_last1', 'gc_last3', 'xgc_last1', 'xgc_last3', 'season_xgc', 'season_gc']],
    how='left',
    left_on=['opponent'],
    right_on=['team'],
    suffixes=('','_opp')
)
# If id not in players: drop from match_logs(?)
# Maybe groupby player_id, team (last) first per player to give latest team
# But then what if a player moves from one pl to another pl after playing?
# Doesn't matter ^! since this is just for calc defensive stats of that team
# Calculate off/def stats

# Calculate offensive stats

# Change dtype
match_logs['expected_goals'] = match_logs['expected_goals'].astype(float)


# Create last game features for minutes, goals, expected goals, and threat
# Use groupby to ensure shift is locked within player_id

match_logs['goals_last1'] = (
    match_logs.groupby('element')['goals_scored']
        .transform(lambda x: previous_rolling_sum(x, 1))
)

match_logs['mins_last1'] = (
    match_logs.groupby('element')['minutes']
        .transform(lambda x: previous_rolling_sum(x, 1))
)

match_logs['xg_last1'] = (
    match_logs.groupby('element')['expected_goals']
        .transform(lambda x: previous_rolling_sum(x, 1))
)

match_logs['threat_last1'] = (
    match_logs.groupby('element')['threat']
        .transform(lambda x: previous_rolling_sum(x, 1))
)

# Create rolling previous 3 game statistics

match_logs['goals_last3'] = (
    match_logs.groupby('element')['goals_scored']
        .transform(lambda x: previous_rolling_sum(x, 3))
)

match_logs['mins_last3'] = (
    match_logs.groupby('element')['minutes']
        .transform(lambda x: previous_rolling_sum(x, 3))
)

match_logs['xg_last3'] = (
    match_logs.groupby('element')['expected_goals']
        .transform(lambda x: previous_rolling_sum(x, 3))
)

match_logs['threat_last3'] = (
    match_logs.groupby('element')['threat']
        .transform(lambda x: previous_rolling_sum(x, 3))
)

# Test cumulative stats - not including given gw data (shift(1))
match_logs['season_mins'] = (
    match_logs.groupby('element')['minutes']
        .transform(lambda x: x.shift(1).cumsum())
)

match_logs['season_xg'] = (
    match_logs.groupby('element')['expected_goals']
        .transform(lambda x: x.shift(1).cumsum())
)

match_logs['season_goals'] = (
    match_logs.groupby('element')['goals_scored']
        .transform(lambda x: x.shift(1).cumsum())
)

# Calculate xg_per90 feature - what about goals_per90?
# Avoid divide-by-zero
match_logs['season_xg_per90'] = (
    match_logs['season_xg'] / match_logs['season_mins'].replace(0, np.nan) * 90
)

# Join to main_df

# Only select latest round
match_logs_last = match_logs.groupby(
    'element').tail(1)

main_df = main_df.merge(
    match_logs_last[['goals_last1', 'xg_last1', 'mins_last1',
                     'threat_last1', 'goals_last3', 'mins_last3',
                     'xg_last3', 'threat_last3', 'season_mins',
                     'season_xg', 'season_goals', 'season_xg_per90',
                     'element']],
    how='left',
    left_on=['id'],
    right_on=['element'],
    suffixes=('','_opp')
    )

# Ensure variables equal to model variables

# Rename variables
main_df = main_df.rename(columns={
    'element_type' : 'position',
    'is_home': 'was_home',
    'opponent': 'opponent_team',
    'event': 'gameweek'
})

feature_list = ['position',
               'gameweek',
               'was_home',
               'opponent_team',
               'gc_last3',
               'xgc_last3',
               'season_gc',
               'season_xgc',
               'mins_last1',
               'goals_last1',
               'xg_last1',
               'xg_last3',
                'goals_last3',
                'mins_last3',
                'season_mins',
                'season_xg',
                'season_goals',
                'season_xg_per90',
               'fdr',
               'threat_last1',
               'threat_last3',
               'last_ssn_goals',
               'last_ssn_xg',
               'last_ssn_mins',
               'last_ssn_xgper90',
               'last_ssn_glsper90',
               'career_xgper90',
               'career_glsper90',
               'gc_last1',
               'xgc_last1'
               ]

target_variable = 'goal_scored'

for var in feature_list:
    if var not in main_df.columns:
        main_df[var] = np.nan
        
X_inference = main_df[feature_list]

assert list(X_inference.columns) == feature_list

# Load model
model = joblib.load("models/fpl_rf_model.joblib")

# Make predictions
preds = model.predict_proba(X_inference)[:, 1]

metadata_cols = ['web_name',
                'gameweek',
                'id']

prediction_df = (main_df[metadata_cols].copy())

prediction_df['P(goal)'] = model.predict_proba(main_df[feature_list])[:, 1]

prediction_df = prediction_df.sort_values(by=['P(goal)'], ascending=False)

# Save predictions
save_csv(prediction_df, "predicts", current_time)