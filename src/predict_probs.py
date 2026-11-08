import numpy as np
import pandas as pd
from statistics import NormalDist
from datetime import datetime

from config import *

def predict_week(future, week, print_output = False, view ='post'):
    """
    Summarize predicted winners, win probabilities, and margins for a single week.

    Args:
        future: DataFrame of upcoming games with '{view}_home_win_prob' and
            '{view}_pred_margin' columns already computed.
        week: Week number to predict.
        print_output: If True, print a readable summary of each game.
        view: Which estimate to use: 'prior', 'obs', or 'post'.

    If the week has not been played yet, the predictions are also written to
    PREDICTION_DIR as a csv.
    """

    # Restate each game from the perspective of the predicted winner
    week_df = future[future['week'] == week].copy()
    week_df['pred_winner'] = np.where(week_df[f'{view}_home_win_prob']>0.5, week_df['home_team'], week_df['away_team'])
    week_df['pred_loser'] = np.where(week_df[f'{view}_home_win_prob']<=0.5, week_df['home_team'], week_df['away_team'])
    week_df['pred_win_prob'] = np.where(week_df[f'{view}_home_win_prob']>0.5, week_df[f'{view}_home_win_prob'], 1 - week_df[f'{view}_home_win_prob'])
    week_df['pred_margin'] = np.where(week_df[f'{view}_home_win_prob']>0.5, week_df[f'{view}_pred_margin'], -week_df[f'{view}_pred_margin'])
    week_df = week_df.sort_values(by = 'pred_win_prob', ascending = False)

    # Build a readable summary, most confident picks first
    text = f"*** Week {week} Predictions: ***\n"
    for idx, row in week_df.iterrows():
        text+=f'{row['away_team']} @ {row['home_team']}: {row['pred_winner']} is {row['pred_win_prob']*100:.1f}% likely to beat {row['pred_loser']}. Expected Margin is {row['pred_margin']:.1f} points.\n'
    if print_output:
        print(text)
    # Only save predictions for weeks that haven't been played yet (no games in past),
    # so a true pre-game prediction is never overwritten after the fact
    if week not in list(past['week']):
        week_df['view'] = view
        week_df = week_df[['season', 'week', 'view','home_team', 'away_team', 'pred_winner', 'pred_margin', 'pred_win_prob']]
        week_df.to_csv(PREDICTION_DIR+ f'week{week}_prediction_{view}.csv', index = False)

def get_home_win_prob(mean, std):
    """Return P(home margin > 0), treating the margin as Normal(mean, std)."""
    d = NormalDist(mu=mean, sigma=std)
    return 1-d.cdf(0)

# Load upcoming games, completed games, and team SRS ratings
future = pd.read_csv(FUTURE_FILE)
past = pd.read_csv(PAST_FILE)
srs = pd.read_csv(SRS_DATA_FILE)

# Number of games observed this season, and the effective number of games
# the prior is worth when blending the two
games_obs = len(past)
games_prior = 256

# Home field advantage: blend this season's observed HFA with the prior,
# weighted by the number of games behind each
hfa_df  = pd.read_csv(weekly_hfa_file)
future['hfa_obs'] = hfa_df.loc[hfa_df['week'] == current_week, 'hfa'].iloc[0]
future['hfa_prior'] = PRIOR_HFA           # Prior HFA is from 2025 data
future['hfa_post'] =(games_obs*future['hfa_obs']+games_prior*future['hfa_prior'] )/(games_obs+games_prior)

# Game-level noise: std dev of actual margins around SRS-predicted margins
sigma_obs = srs['sigma_obs'].iloc[0]
views = ['prior', 'obs', 'post']



# Count games played so far by each home and away team
games_played = pd.concat([past['home_team'], past['away_team']]).value_counts()
future['home_games_played'] = future['home_team'].map(games_played).fillna(0).astype(int)
future['away_games_played'] = future['away_team'].map(games_played).fillna(0).astype(int)

# Attach prior, observed, and posterior SRS for both teams in each game
for team in ['home', 'away']:
    future = future.merge(
        srs[['team', 'srs_prior', 'srs_obs', 'srs_post']], left_on = f'{team}_team', right_on = 'team', how = 'left'
    ).rename(columns = {
        'srs_prior':f'{team}_srs_prior',
        'srs_obs':f'{team}_srs_obs',
        'srs_post':f'{team}_srs_post'}
    ).drop(columns = ['team'])
    # Posterior std dev of the team's rating (normal-normal update): shrinks
    # from TAU toward 0 as the team plays more games
    future[f'{team}_std_post'] = future.apply(
        lambda row: np.sqrt(1/(row[f'{team}_games_played']/sigma_obs**2 + 1/TAU**2)),
        axis = 1
    )
    future[f'{team}_std_prior'] = TAU
    future[f'{team}_std_obs'] = sigma_obs


# Expected home margin for each view:
#   margin = srs_home - srs_away + HFA (HFA omitted for neutral-site games)
for view in views:
    future[f'{view}_pred_margin'] = (future[f'home_srs_{view}'] - future[f'away_srs_{view}']
                                     + (~future['neutral'].astype(bool)) * future[f'hfa_{view}'])

# Home win probability for each view

# Posterior: total variance combines game-level noise with the rating
# uncertainty of each team
future['post_home_win_prob'] = future.apply(
    lambda row: get_home_win_prob(row['post_pred_margin'], np.sqrt(sigma_obs**2+row['home_std_post']**2+row['away_std_post']**2)), axis=1
)

# Prior: prior margin with the prior std dev
future['prior_home_win_prob'] = future.apply(
    lambda row: get_home_win_prob(row['prior_pred_margin'], TAU),
    axis = 1
)

# Observed: observed margin with game-level noise only
future['obs_home_win_prob'] = future.apply(
    lambda row: get_home_win_prob(row['obs_pred_margin'], sigma_obs),
    axis = 1
)

# Predict the upcoming week
predict_week(future, current_week+1, print_output = True, view ='post')

# Save all margin predictions: once as the current file, once as a dated archive copy
future.to_csv(MARGIN_PREDICTIONS_FILE, index = False)
today = datetime.today().strftime('%Y%m%d')
future.to_csv(PREDICTION_ARCHIVE_DIR+f'{today}_predicted_margins.csv', index = False)