import numpy as np
import pandas as pd
from statistics import NormalDist
from datetime import datetime

from config import *

# A function that takes a df with future predictions lined up, prints the
def predict_week(future, week, print_output = False, view ='post'):

    # Get the specified weeks data
    week_df = future[future['week'] == week].copy()
    week_df['pred_winner'] = np.where(week_df[f'{view}_home_win_prob']>0.5, week_df['home_team'], week_df['away_team'])
    week_df['pred_loser'] = np.where(week_df[f'{view}_home_win_prob']<=0.5, week_df['home_team'], week_df['away_team'])
    week_df['pred_win_prob'] = np.where(week_df[f'{view}_home_win_prob']>0.5, week_df[f'{view}_home_win_prob'], 1 - week_df[f'{view}_home_win_prob'])
    week_df['pred_margin'] = np.where(week_df[f'{view}_home_win_prob']>0.5, week_df[f'{view}_pred_margin'], -week_df[f'{view}_pred_margin'])
    week_df = week_df.sort_values(by = 'pred_win_prob', ascending = False)

   # Print the text
    text = f"*** Week {week} Predictions: ***\n"
    for idx, row in week_df.iterrows():
        text+=f'{row['away_team']} @ {row['home_team']}: {row['pred_winner']} is {row['pred_win_prob']*100:.1f}% likely to beat {row['pred_loser']}. Expected Margin is {row['pred_margin']:.1f} points.\n'
    if print_output:
        print(text)
    # If the week is complete (IE no games in past df), write the prediction as a dataframe
    if week not in list(past['week']):
        week_df['view'] = view
        week_df = week_df[['season', 'week', 'view','home_team', 'away_team', 'pred_winner', 'pred_margin', 'pred_win_prob']]
        week_df.to_csv(PREDICTION_DIR+ f'week{week}_prediction_{view}.csv', index = False)

# Get the home win probability from the mean expected margin and the standard deviation
def get_home_win_prob(mean, std):
    d = NormalDist(mu=mean, sigma=std)
    return 1-d.cdf(0)

# Load future games, past games, and srs data
future = pd.read_csv(FUTURE_FILE)
past = pd.read_csv(PAST_FILE)
srs = pd.read_csv(SRS_DATA_FILE)

# Observed total games and prior games
games_obs = len(past)
games_prior = 256

# Home field advantage observed, prior, and posterior
hfa_df  = pd.read_csv(weekly_hfa_file)
hfa_obs = hfa_df.loc[hfa_df['week'] == current_week, 'hfa'].iloc[0]
hfa_prior = PRIOR_HFA           # Prior hfa is 2025 data
hfa_post =(games_obs*hfa_obs+games_prior*hfa_prior)/(games_obs+games_prior)

sigma_obs = srs['sigma_obs'].iloc[0]
views = ['prior', 'obs', 'post']

# Map games played for eahc home and away team
games_played = pd.concat([past['home_team'], past['away_team']]).value_counts()
future['home_games_played'] = future['home_team'].map(games_played).fillna(0).astype(int)
future['away_games_played'] = future['away_team'].map(games_played).fillna(0).astype(int)

# Merge srs data into the future games for home and away teams
for team in ['home', 'away']:
    future = future.merge(
        srs[['team', 'srs_prior', 'srs_obs', 'srs_post']], left_on = f'{team}_team', right_on = 'team', how = 'left'
    ).rename(columns = {
        'srs_prior':f'{team}_srs_prior',
        'srs_obs':f'{team}_srs_obs',
        'srs_post':f'{team}_srs_post'}
    ).drop(columns = ['team'])
    # Posterior standard deviation, will depend on games played for each team
    future[f'{team}_std_post'] = future.apply(
        lambda row: np.sqrt(1/(row[f'{team}_games_played']/sigma_obs**2 + 1/TAU**2)),
        axis = 1
    )


# Calculate the prior, observed, and posterior expected margin based on SRS formula
# margin = srs_home - srs_away + homefield_advantage (if not played on a neutral field)
for view in views:
    future[f'{view}_pred_margin'] = (future[f'home_srs_{view}'] - future[f'away_srs_{view}']
                                     + (~future['neutral'].astype(bool)) * hfa_post)

# Calculate home win probability for each view

# Posterior prediction is uses the posterior margin, and the added variance of the observation with the posterior variance of each team
future['post_home_win_prob'] = future.apply(
    lambda row: get_home_win_prob(row['post_pred_margin'], np.sqrt(sigma_obs**2+row['home_std_post']**2+row['away_std_post']**2)), axis=1
)

# Prior just uses the prior margin and the prior std dev
future['prior_home_win_prob'] = future.apply(
    lambda row: get_home_win_prob(row['prior_pred_margin'], TAU),
    axis = 1
)

# Observed just uses the observed margin and the observed std (sigma)
future['obs_home_win_prob'] = future.apply(
    lambda row: get_home_win_prob(row['obs_pred_margin'], sigma_obs),
    axis = 1
)

# Predict the next week
predict_week(future, current_week+1, print_output = True, view ='post')

# Write future df to csv twice, once for the current predictions and once to the archives labeled by date
future.to_csv(MARGIN_PREDICTIONS_FILE, index = False)
today = datetime.today().strftime('%Y%m%d')
future.to_csv(PREDICTION_ARCHIVE_DIR+f'{today}_predicted_margins.csv', index = False)