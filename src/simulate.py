from datetime import datetime
import pandas as pd
import numpy as np
import random
from config import *
from predict_probs import add_win_probabilities

pd.set_option('display.max_rows', None)
past = pd.read_csv(PAST_FILE)
srs = pd.read_csv(SRS_DATA_FILE)
hfa_df = pd.read_csv(weekly_hfa_file)

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

def simulate_playoff(forecast, srs, past, hfa_df, view = 'post'):
    """
    Simulate the playoffs once from a simulated regular season's seeds.

    Bracket follows the NFL format: seeds 2-7 play in the wild card round,
    the 1 seed hosts the lowest remaining seed in the divisional round
    (reseeding), the higher seed hosts each remaining conference game, and
    the Super Bowl is played at a neutral site.

    Args:
        forecast: Output of simulate_reg_season (indexed by team, with
            'conference' and 'seed' columns).
        srs, past, hfa_df: Passed to add_win_probabilities.
        view: Which estimate to simulate with: 'prior', 'obs', or 'post'.

    Returns:
        forecast with boolean columns marking where each team's season
        ended: wild_card_loss, divisional_loss, conference_championship_loss,
        superbowl_loss, superbowl_win. Non-playoff teams are False in all.
    """
    seed = forecast['seed']
    conference = forecast['conference']
    playoff_teams = forecast[seed.notna()].reset_index().astype({'seed': int})
    seed_lookup = playoff_teams.set_index(['conference', 'seed'])['team']

    def play_round(games):
        """Draw a winner for each (home_team, away_team, neutral) game; return winners and losers."""
        games = pd.DataFrame(games, columns = ['home_team', 'away_team', 'neutral'])
        games = add_win_probabilities(games, past, srs, hfa_df)
        winners = games.apply(predict_winner, axis = 1, view = view)
        losers = np.where(winners == games['home_team'], games['away_team'], games['home_team'])
        return list(winners), list(losers)

    def by_seed(teams, conf):
        """Teams from one conference, best (lowest) seed first."""
        return sorted([t for t in teams if conference[t] == conf], key = lambda t: seed[t])

    # Wild card round: matchups by seed from the bracket file
    wc = WILD_CARD_GAMES
    wc_games = [(seed_lookup[(c, h)], seed_lookup[(c, a)], 0)
                for c, h, a in zip(wc['conference'], wc['home_team_seed'], wc['away_team_seed'])]
    wc_winners, wc_losers = play_round(wc_games)

    # Divisional round: the 1 seed (bye) plus the wild card winners, reseeded so
    # the best seed hosts the worst remaining seed
    div_games = []
    for conf in ['AFC', 'NFC']:
        alive = by_seed([seed_lookup[(conf, 1)]] + wc_winners, conf)
        div_games += [(alive[0], alive[3], 0), (alive[1], alive[2], 0)]
    div_winners, div_losers = play_round(div_games)

    # Conference championships: higher remaining seed hosts
    cc_games = []
    for conf in ['AFC', 'NFC']:
        alive = by_seed(div_winners, conf)
        cc_games.append((alive[0], alive[1], 0))
    cc_winners, cc_losers = play_round(cc_games)

    # Super Bowl: neutral site, so home/away labels don't matter
    sb_winners, sb_losers = play_round([(cc_winners[0], cc_winners[1], 1)])

    # Flag where each team's season ended
    forecast = forecast.copy()
    outcomes = {
        'wild_card_loss': wc_losers,
        'divisional_loss': div_losers,
        'conference_championship_loss': cc_losers,
        'superbowl_loss': sb_losers,
        'superbowl_win': sb_winners,
    }
    for col, teams in outcomes.items():
        forecast[col] = forecast.index.isin(teams)
    return forecast

def simulate_seasons(n, future, win_history, srs, records, view = 'post'):
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
        forecast = simulate_playoff(forecast, srs, past, hfa_df, view)
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
        wild_card_loss = ('wild_card_loss', 'mean'),
        divisional_loss = ('divisional_loss', 'mean'),
        conference_championship_loss = ('conference_championship_loss', 'mean'),
        superbowl_loss = ('superbowl_loss', 'mean'),
        superbowl_win = ('superbowl_win', 'mean'),
    )
    summary['current_record'] = records.reindex(summary.index).fillna('0-0')

    # Order columns, convert odds to percentages, and sort by playoff odds
    # within each conference
    odds_cols = ['playoff_odds', 'division_odds', 'top_seed_odds']
    playoff_cols = ['wild_card_loss', 'divisional_loss', 'conference_championship_loss','superbowl_loss', 'superbowl_win']
    summary = summary[['conference', 'division', 'current_record', 'srs',
                       'avg_pred_wins', 'avg_div_place'] + odds_cols + playoff_cols]
    summary[odds_cols+playoff_cols] *= 100
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

# Simulate the rest of the season N_SIMS times using the posterior view
for view in ['post']:
    summary  = simulate_seasons(N_SIMS, margins, wins, srs, records, view)
