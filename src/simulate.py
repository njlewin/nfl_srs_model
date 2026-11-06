from datetime import datetime
import pandas as pd
import numpy as np
import random
from config import *
pd.set_option('display.max_rows', None)


def predict_winner(row, view = 'post'):
    """
    Randomly draw a winner for one game, using the home win probability
    for the given view ('prior', 'obs', or 'post') from predict_probs.py.
    """
    if random.random()<=row[f'{view}_home_win_prob']:
        return row['home_team']
    else:
        return row['away_team']

def add_seeds(df, tiebreak = None):
    """
    Assign playoff seeds in each conference using the NFL's format:
    the 4 division winners get seeds 1-4, and the 3 best remaining teams
    get wildcard seeds 5-7. Teams that miss the playoffs get a null seed.

    Teams are ranked by wins, with ties broken by the `tiebreak` column
    (e.g. SRS). This is a simplification of the NFL's actual tiebreakers
    (head-to-head, division record, etc.).
    """
    df = df.copy()
    sort_cols = ['wins'] +([tiebreak] if tiebreak else [])
    ranked = df.sort_values(sort_cols, ascending = False, kind = 'stable')
    # After ranking, the first team listed from each division is its winner
    ranked['div_winner'] = ~ranked.duplicated('division')
    df["seed"] = pd.Series(np.nan, index=df.index, dtype="Int64")

    # Groups keep the ranked order, so taking the first 4 division winners
    # and first 3 non-winners gives seeds in the right order
    for conf, g in ranked.groupby('conference', sort = False):
        winners = g.index[g['div_winner']][:4]
        wildcards = g.index[~g['div_winner']][:3]
        df.loc[winners, "seed"] = range(1, len(winners) + 1)
        df.loc[wildcards, "seed"] = range(5, 5 + len(wildcards))
    return df

def simulate_reg_season(future, win_history, srs, view = 'post'):
    """
    Simulate the rest of the regular season once and return final standings.

    Args:
        future: Remaining games with home win probabilities for each view,
            as calculated in predict_probs.py.
        win_history: Current win totals per team (ties count as half a win).
        srs: Team SRS ratings; used to break ties in standings and seeding.
        view: Which estimate to simulate with: 'prior', 'obs', or 'post'.

    Returns:
        DataFrame indexed by team with final wins, SRS, division place,
        playoff seed, and 0/1 flags for making the playoffs, winning the
        division, and earning the top seed.
    """
    # SRS ratings for the chosen view (prior, observed, or Bayesian posterior)
    view_srs = srs[f'srs_{view}'].rename('srs')
    view_srs.index = srs['team']

    # Draw a winner for every remaining game, count the wins per team, and add
    # them to current wins. fill_value=0 covers teams with no simulated wins.
    wins = win_history.add(
        future.apply(predict_winner, axis=1, view=view).value_counts(),
        fill_value=0
    )

    # Combine final win totals with SRS into one row per team
    forecast = (
        wins.rename('wins').to_frame()
        .join(view_srs, how='inner')
        .rename_axis('team')
    )
    # Add conference and division, then rank teams within each division
    # by wins (SRS breaks ties)
    forecast['division'] = forecast.index.map(div_lookup)
    forecast['conference'] = forecast.index.map(conf_lookup)
    forecast['div_place'] = (
            forecast.sort_values(['wins', 'srs'], ascending=False)
            .groupby('division')
            .cumcount() + 1
    ).astype(int)

    # Add playoff seeds and 0/1 outcome flags, which become probabilities
    # when averaged across simulations
    forecast = add_seeds(forecast, 'srs')
    forecast['made_playoffs'] = forecast['seed'].notna().astype(int)
    forecast['won_division'] = (forecast['div_place'] == 1).astype(int)
    forecast['top_seed'] = (~(forecast['seed'].isna()) & (forecast['seed'] == 1)).astype(int)
    return forecast

def simulate_reg_seasons(n, future, win_history, srs, records, view = 'post'):
    """
    Run n regular season simulations and summarize the results per team.

    Args:
        n: Number of seasons to simulate.
        future, win_history, srs, view: Passed to simulate_reg_season.
        records: Current record strings per team (e.g. '5-3'), for display.

    Returns:
        DataFrame with each team's average wins and division place, and
        its playoff, division, and top seed odds (as percentages). The
        summary is also written to PREDICTION_DIR and a dated archive copy.
    """
    # Simulate n seasons and stack the results into one dataframe
    forecasts = []
    for i in range(n):
        forecast = simulate_reg_season(future, win_history, srs, view)
        forecasts.append(forecast)
    all_sims = pd.concat(forecasts)

    # Average across simulations. The mean of each 0/1 flag is the share of
    # simulations where it happened, i.e. that outcome's probability.
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

    # Order columns, convert odds to percentages, and sort by playoff odds
    # within each conference
    odds_cols = ['playoff_odds', 'division_odds', 'top_seed_odds']
    summary = summary[['conference', 'division', 'current_record', 'srs',
                       'avg_pred_wins', 'avg_div_place'] + odds_cols]
    summary[odds_cols] *= 100
    summary = summary.round(1).sort_values(['conference', 'playoff_odds'], ascending=False)
    today = datetime.today().strftime('%Y%m%d')

    # Save the current predictions, plus a dated copy to the archive
    summary.to_csv(PREDICTION_DIR + f'reg_season_predictions_{view}.csv')
    summary.to_csv(PREDICTION_ARCHIVE_DIR + f'{today}_reg_season_predictions_{view}.csv')
    return summary

# Load current team records, both as win totals (ties count as half a win)
# and as display strings
team_records = pd.read_csv(RECORDS_FILE).set_index(['team'])
wins = (team_records['wins'] + team_records['ties'] * 0.5)
records = team_records['record']

# Load remaining games with win probabilities (from predict_probs.py) and SRS ratings
margins = pd.read_csv(MARGIN_PREDICTIONS_FILE)
srs = pd.read_csv(SRS_DATA_FILE)

# Simulate the rest of the season N_SIMS times using the posterior view
summary  = simulate_reg_seasons(N_SIMS, margins, wins, srs,records, 'post')
print(summary)