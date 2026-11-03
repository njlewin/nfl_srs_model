import numpy as np
import pandas as pd
import nflreadpy as nfl
from pandas import DataFrame

from config import *


def od_srs_rankings(df, season):
    """Fit offensive and defensive Simple Rating System (SRS) ratings by least squares.

    Each team's score in a game is modeled as:
        points = league avg + own offense - opponent defense +/- hfa/2

    Args:
        df: Game results with columns home_team, away_team, home_score,
            away_score and neutral (1 if played at a neutral site).
        season: Season label for the output. Not used in the fit.

    Returns:
        A tuple of (result_df, hfa). result_df has columns season, team, osrs
        and dsrs, sorted from best to worst total SRS (osrs + dsrs). Higher is
        better for both ratings. hfa is the home-field advantage in margin points.
    """
    teams = sorted(set(df['home_team']) | set(df['away_team']))
    col = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    DEF_OFFSET = n_teams          # defense columns come after offense columns
    AVG_COL = 2 * n_teams         # league scoring average
    HFA_COL = 2 * n_teams + 1     # home-field advantage (in margin points)

    # Define the X and y matrices: two rows per game, one per team's points
    X = np.zeros((2 * len(df), 2 * n_teams + 2))
    y = np.zeros(2 * len(df))
    for i, g in enumerate(df.itertuples()):
        home_row, away_row = 2 * i, 2 * i + 1
        hfa_entry = 0 if g.neutral == 1 else 0.5

        # Home team's points = avg + home offense - away defense + hfa/2.
        # Splitting hfa across both rows makes the coefficient the full margin effect.
        X[home_row, col[g.home_team]] = 1
        X[home_row, DEF_OFFSET + col[g.away_team]] = -1
        X[home_row, AVG_COL] = 1
        X[home_row, HFA_COL] = hfa_entry
        y[home_row] = g.home_score

        # Away team's points = avg + away offense - home defense - hfa/2
        X[away_row, col[g.away_team]] = 1
        X[away_row, DEF_OFFSET + col[g.home_team]] = -1
        X[away_row, AVG_COL] = 1
        X[away_row, HFA_COL] = -hfa_entry
        y[away_row] = g.away_score

    # X is rank deficient: a constant shift in all offenses, all defenses or the
    # average can be offset by the others. lstsq picks the minimum-norm solution.
    solution, *_ = np.linalg.lstsq(X, y, rcond=None)

    osrs = solution[:n_teams]
    dsrs = solution[DEF_OFFSET:DEF_OFFSET + n_teams]
    league_avg = solution[AVG_COL]
    hfa = solution[HFA_COL]

    # Recenter offense and defense to average zero, moving the shifts into
    # the league average so every prediction stays the same
    league_avg += osrs.mean() - dsrs.mean()
    osrs = osrs - osrs.mean()
    dsrs = dsrs - dsrs.mean()

    result_df = pd.DataFrame({'season': season, 'team': teams,
                              'osrs': osrs, 'dsrs': dsrs})
    result_df = (result_df.assign(_srs=result_df['osrs'] + result_df['dsrs'])
                 .sort_values('_srs', ascending=False)
                 .drop(columns='_srs')
                 .reset_index(drop=True))

    return result_df, hfa


def get_observed_srs(df):
    """Calculate cumulative SRS by week and the per-game error of the latest fit.

    Writes the weekly SRS and hfa results to CSVs in SRS_DIR. The error
    calculation reads the module-level `past` dataframe rather than df.

    Args:
        df: Game results with a week column plus the columns required by
            od_srs_rankings.

    Returns:
        A tuple of (srs, sigma). srs is the SRS table through the latest week
        with total SRS added as srs_obs. sigma is the estimated standard
        deviation of a game's actual margin around its predicted margin.
    """
    # Refit on all games through each week so ratings show the season's progression
    srs_list = []
    hfa_list = {}
    for week in sorted(df['week'].unique()):
        df_t = df[df['week']<=week]
        srs_t: DataFrame
        hfa_t: float
        srs_t, hfa_t = od_srs_rankings(df_t, CURRENT_SEASON)
        srs_t['week'] = week
        srs_list.append(srs_t)
        hfa_list[week] =[hfa_t]

    # Concatenate weekly srs and hfa results into dataframes and write them to csvs
    srs_weekly = pd.concat(srs_list).reset_index(drop = True)
    hfa_weekly = pd.DataFrame(hfa_list).T.reset_index()
    hfa_weekly.columns = ['week', 'hfa']
    srs_weekly.to_csv(SRS_DIR+f'{CURRENT_SEASON}_weekly_srs.csv', index = False)
    hfa_weekly.to_csv(SRS_DIR + f'/{CURRENT_SEASON}_weekly_hfa.csv', index = False)

    # The last week's fit uses all games played, so it's the current result
    srs = srs_t.copy()
    hfa = hfa_t.copy()
    srs['srs_obs'] = srs['osrs'] + srs['dsrs']

    # Predict each game's margin from the current ratings. Sigma is therefore
    # in-sample: these ratings were fit on the same games they're predicting.
    merge1 = past.merge(srs[['team', 'srs_obs']], left_on=['home_team'], right_on=['team']).drop(columns=['team']).rename(
        columns={'srs_obs': 'home_srs'})
    past_pred = merge1.merge(srs[['team', 'srs_obs']], left_on=['away_team'], right_on=['team']).drop(
        columns=['team']).rename(columns={'srs_obs': 'away_srs'})
    past_pred['pred_margin'] = past_pred['home_srs'] - past_pred['away_srs'] + (
        ~past_pred['neutral'].astype(bool)) * hfa
    past_pred['error'] = past_pred['actual_margin'] - past_pred['pred_margin']

    # Inflate RMSE to correct for the fitted parameters: n_teams - 1 free ratings
    # (they're centered on zero) plus hfa
    rmse = np.sqrt((past_pred['error'] ** 2).sum() / len(past_pred))
    n = len(past_pred)
    deg_free = len(srs) - 1 + 1
    sigma = rmse * np.sqrt((n) / (n - deg_free))

    return srs, sigma


def add_prior_srs(srs):
    """Add each team's prior SRS, implied by its preseason win total over/under.

    Win totals are centered on the league mean and converted to SRS points
    using SLOPE, so the priors average zero.

    Args:
        srs: Dataframe with a team column.

    Returns:
        srs with an added srs_prior column.
    """
    df_ous = pd.read_csv(PRESEASON_OVERUNDERS)
    df_ous['srs'] = (df_ous['wins_ou'] - df_ous['wins_ou'].mean()) * SLOPE
    prior_srs_dict = pd.Series(df_ous.srs.values, index=df_ous.team).to_dict()
    srs['srs_prior'] = srs['team'].apply(lambda x: prior_srs_dict.get(x))
    return srs


def add_posterior_srs(srs, sigma_obs, sigma_prior):
    """Add each team's posterior SRS, blending its observed and prior SRS.

    Uses the normal-normal Bayesian update, which weights each estimate by its
    precision (1 / variance). The observed SRS carries n_obs / sigma_obs^2, so
    it takes over from the prior as a team plays more games.

    Args:
        srs: Dataframe with columns srs_obs, srs_prior and n_obs (games played).
        sigma_obs: Standard deviation of a single game's margin around its prediction.
        sigma_prior: Standard deviation of true SRS around the prior.

    Returns:
        srs with an added srs_post column.
    """
    srs['srs_post'] = srs.apply(
        lambda row: (
                (row['n_obs']/sigma_obs**2 * (row['srs_obs'])+1/sigma_prior**2*row['srs_prior'])/(row['n_obs']/sigma_obs**2+1/sigma_prior**2)
        ),
        axis = 1
    )
    return srs


past = pd.read_csv(PAST_FILE)
games_played = pd.concat([past['home_team'], past['away_team']]).value_counts()

# Fit the observed ratings, then shrink them toward the preseason priors
srs, sigma = get_observed_srs(past)
srs['n_obs'] = srs['team'].apply(lambda x: games_played[x])
srs = add_prior_srs(srs)
srs = add_posterior_srs(srs, sigma, TAU)

# Add team logos and write the final table
teams = nfl.load_teams().select(["team_abbr", "team_logo_espn"]).to_pandas()
srs = srs.merge(teams, left_on="team", right_on="team_abbr", how="left").drop(columns = 'team_abbr')
week = past['week'].max()
srs.to_csv(SRS_DATA_FILE, index = False)
