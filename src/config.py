from pathlib import Path
import pandas as pd


CURRENT_SEASON = 2026

#Constants from prior seasons, caluclated in attached srs_prior_calc.xlsx

TAU = 4.01              # Standard devation of prior results using preseason srs to predict wins
SLOPE = 1.79            # Slope of plotting SRS results against preseason wins
PRIOR_HFA = 2.189285065 # Homefield advantage from 2025
N_PRIOR = 256           # Games used to calculate prior HFA

#Number of simulations to run
N_SIMS = 10000

TEAM_RENAME_DICT = rename_dict = {'STL': 'LA', 'OAK': 'LV', 'SD':'LAC'}

# Conference and division lookups
DIVISIONS = {
    'AFC East': ['BUF', 'MIA', 'NE', 'NYJ',],
    'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
    'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
    'AFC West': ['DEN', 'KC', 'LAC', 'LV'],
    'NFC East': ['NYG', 'WAS', 'DAL', 'PHI'],
    'NFC North': ['MIN', 'GB', 'DET', 'CHI'],
    'NFC South': ['NO', 'ATL', 'CAR', 'TB'],
    'NFC West': ['SF', 'LA', 'SEA', 'ARI']
}

conferences = {}
for conference in ['AFC', 'NFC']:
    conferences[conference] = [team for key, teams in DIVISIONS.items() if conference in key for team in teams]

# Derived lookups
div_lookup = {t: d for d, teams in DIVISIONS.items() for t in teams}
conf_lookup = {t: d.split()[0] for t, d in div_lookup.items()}

# Directories

DATA_DIR = "../data/"

SRS_DIR = DATA_DIR +"srs/"
SCHEDULE_DIR = DATA_DIR+'schedule/'
CACHE = Path(DATA_DIR+'cache/')
ASSET_DIR = DATA_DIR + 'assets/'
RAW_DIR = DATA_DIR + 'raw/'
PREDICTION_DIR = DATA_DIR + 'predictions/'
PREDICTION_ARCHIVE_DIR = PREDICTION_DIR + 'archive/'
PRESEASON_OVERUNDERS = RAW_DIR + '2026_preseason_ou.csv'

# file names
SCHEDULE_FILE = SCHEDULE_DIR + 'schedule.csv'
PAST_FILE = SCHEDULE_DIR + 'past.csv'
FUTURE_FILE = SCHEDULE_DIR + 'future.csv'
RECORDS_FILE = SCHEDULE_DIR + 'records.csv'
MARGIN_PREDICTIONS_FILE = PREDICTION_DIR + 'predicted_margins.csv'

weekly_srs_file = SRS_DIR + f'{CURRENT_SEASON}_weekly_srs.csv'
weekly_hfa_file = SRS_DIR + f'{CURRENT_SEASON}_weekly_hfa.csv'

current_week = pd.read_csv(PAST_FILE)['week'].max()
SRS_DATA_FILE = SRS_DIR + f'week{current_week}_team_srs_data.csv'
