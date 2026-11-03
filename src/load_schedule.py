import nflreadpy as nfl
import pandas as pd
import numpy as np
from config import *

# Load current season nfl schedule from readpy
cols = ['season','week','gameday','home_team','away_team','home_score', 'away_score', 'game_type', 'result', 'location']
df = nfl.load_schedules([CURRENT_SEASON]).to_pandas()[cols]

# Add flags for playoff and neutral
df['playoff'] = np.where(df['game_type']!= 'REG', 1, 0)
df['neutral'] = np.where(df['location']== 'Neutral', 1, 0)

# Convert gameday to a datetime
df['gameday'] = pd.to_datetime(df['gameday'] )

# Get a count of games played for each team
played = df['home_score'].notna() & (df['game_type'] == 'REG')
games_played = pd.concat([df.loc[played, 'home_team'],
                          df.loc[played, 'away_team']]).value_counts()

# Split the games into past (games that have been played) and future games
past = df[(~df['home_score'].isnull()) & (~df['away_score'].isnull())].copy()
future = df[(df['home_score'].isnull()) & (df['away_score'].isnull())].copy()

# Add details to the past results
past = past.rename(columns = {'result': 'actual_margin'})
past['winner'] = np.where(past['actual_margin'] > 0, past['home_team'], past['away_team'])
past['winner'] = np.where(past['actual_margin'] == 0, 'Tie', past['winner'])

# Get each teams, W/L/Tie record and create a dataframe along with a str
teams = games_played.index
wins = past['winner'].value_counts()
wins = wins.reindex(teams, fill_value=0).astype(int)
tie_games = past[past['winner'] == 'Tie']
ties = (pd.concat([tie_games['home_team'], tie_games['away_team']])
          .value_counts()
          .reindex(teams, fill_value=0)
          .astype(int))
losses = games_played - wins - ties

records = pd.concat([wins, losses, ties], axis=1, keys=["wins", "losses", "ties"])
records['record'] = records['wins'].astype(str) + '-' + records['losses'].astype(str)
records.loc[records['ties'] > 0, 'record'] += '-' + records['ties'].astype(str)
records.index.name = 'team'

# Write all to CSVs
df.to_csv(SCHEDULE_FILE, index= False)
past.to_csv(PAST_FILE, index= False)
future.to_csv(FUTURE_FILE, index= False)
records.to_csv(SCHEDULE_DIR + 'records.csv', index=True)




