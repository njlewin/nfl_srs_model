from datetime import datetime
import pandas as pd
import numpy as np
import random
from config import *
pd.set_option('display.max_rows', None)


# Predict a winner for a given game from the future dataframe once home win probabilities are calculated
def predict_winner(row, view = 'post'):
    if random.random()<=row[f'{view}_home_win_prob']:
        return row['home_team']
    else:
        return row['away_team']

# Once the season is predicted use the NFL's seeding method to determine the playoff seeds in the NFC and AFC
def add_seeds(df, tiebreak = None):
    df = df.copy()
    sort_cols = ['wins'] +([tiebreak] if tiebreak else [])
    ranked = df.sort_values(sort_cols, ascending = False, kind = 'stable')
    ranked['div_winner'] = ~ranked.duplicated('division')
    df["seed"] = pd.Series(np.nan, index=df.index, dtype="Int64")

    for conf, g in ranked.groupby('conference', sort = False):
        winners = g.index[g['div_winner']][:4]
        wildcards = g.index[~g['div_winner']][:3]
        df.loc[winners, "seed"] = range(1, len(winners) + 1)
        df.loc[wildcards, "seed"] = range(5, 5 + len(wildcards))
    return df

# Simulate a season, using the future df--which has all future games with prior, observed, and posterior win probabilities
# calculated in predict_probs.py, along with the win_history df, srs ratings and the desired view.
def simulate_reg_season(future, win_history, srs, view = 'post'):
    # Get the SRS of the given view between prior assumptions, oberved data, and bayesian posterior:
    view_srs = srs[f'srs_{view}'].rename('srs')
    view_srs.index = srs['team']

    # Get the total predicted wins by predicting a winner each game in the rest of the season and add it to current wins
    wins = win_history.add(
        future.apply(predict_winner, axis=1, view=view).value_counts(),
        fill_value=0
    )

    # Build a forecast dataframe with the results.
    forecast = (
        wins.rename('wins').to_frame()
        .join(view_srs, how='inner')
        .rename_axis('team')
    )
    # add conferenc and division data
    forecast['division'] = forecast.index.map(div_lookup)
    forecast['conference'] = forecast.index.map(conf_lookup)
    forecast['div_place'] = (
            forecast.sort_values(['wins', 'srs'], ascending=False)
            .groupby('division')
            .cumcount() + 1
    ).astype(int)

    # Add playoff seeding
    forecast = add_seeds(forecast, 'srs')
    forecast['made_playoffs'] = forecast['seed'].notna().astype(int)
    forecast['won_division'] = (forecast['div_place'] == 1).astype(int)
    forecast['top_seed'] = (~(forecast['seed'].isna()) & (forecast['seed'] == 1)).astype(int)
    return forecast

# Run n regular season simulations and return the average results as a dataframe.
def simulate_reg_seasons(n, future, win_history, srs, records, view = 'post', write_ouput = True):
    # Run n forecasts and add them to a single dataframe
    forecasts = []
    for i in range(n):
        forecast = simulate_reg_season(future, win_history, srs, view)
        forecasts.append(forecast)
    all_sims = pd.concat(forecasts)

    # Take the average results across all simulations
    summary = all_sims.groupby(level='team').agg(
        conference=('conference', 'first'),
        division=('division', 'first'),
        srs=('srs', 'first'),
        avg_pred_wins=('wins', 'mean'),
        avg_div_place=('div_place', 'mean'),
        playoff_odds=('made_playoffs', 'mean'),
        division_odds=('won_division', 'mean'),
        top_seed_odds=('top_seed', 'mean'),
    )
    summary['current_record'] = records.reindex(summary.index).fillna('0-0')

    odds_cols = ['playoff_odds', 'division_odds', 'top_seed_odds']
    summary = summary[['conference', 'division', 'current_record', 'srs',
                       'avg_pred_wins', 'avg_div_place'] + odds_cols]
    summary[odds_cols] *= 100
    summary = summary.round(1).sort_values(['conference', 'playoff_odds'], ascending=False)
    today = datetime.today().strftime('%Y%m%d')

    summary.to_csv(PREDICTION_DIR + f'reg_season_predictions_{view}.csv')
    summary.to_csv(PREDICTION_ARCHIVE_DIR + f'{today}_reg_season_predictions_{view}.csv')
    return summary

# Load the team records, both as win totals and current record string
team_records = pd.read_csv(RECORDS_FILE).set_index(['team'])
wins = (team_records['wins'] + team_records['ties'] * 0.5)
records = team_records['record']

# Load the predicted margins and SRS data
margins = pd.read_csv(MARGIN_PREDICTIONS_FILE)
srs = pd.read_csv(SRS_DATA_FILE)

# Run the simulation
summary  = simulate_reg_seasons(N_SIMS, margins, wins, srs,records, 'post')
print(summary)