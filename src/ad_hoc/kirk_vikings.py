
import pandas as pd
import numpy as np

from config import *
past = pd.read_csv(PAST_FILE)
OBS_WEEK = 9
obs = past[past['week']<=OBS_WEEK].copy()

avg_scored = (pd.concat([obs['home_score'], obs['away_score']]).mean())
sigma = pd.read_csv(SRS_DATA_FILE)['std_obs'].iloc[0]
games_played = pd.concat([obs['home_team'], obs['away_team']]).value_counts()
n_min = games_played.loc['MIN']
n_gb = games_played.loc['GB']

srs_data = pd.read_csv(weekly_srs_file)
min_osrs = srs_data[(srs_data['team'] == 'MIN') & (srs_data['week'] == OBS_WEEK)]['osrs'].iloc[0]
gb_dsrs = srs_data[(srs_data['team'] == 'GB') & (srs_data['week'] == OBS_WEEK)]['dsrs'].iloc[0]
hfa_data = pd.read_csv(weekly_hfa_file)
hfa = hfa_data[hfa_data['week'] == OBS_WEEK]['hfa'].iloc[0]

mean = avg_scored + min_osrs - gb_dsrs - hfa/2

# sigma (std_obs) is the noise in a game *margin*. For a single team's score we need the
# noise in points scored, estimated from the score-level residuals of the week-9 fit.
wk = srs_data[srs_data['week'] == OBS_WEEK].set_index('team')
home_adj = np.where(obs['neutral'] == 1, 0, hfa / 2)
home_pred = avg_scored + wk.loc[obs['home_team'], 'osrs'].values - wk.loc[obs['away_team'], 'dsrs'].values + home_adj
away_pred = avg_scored + wk.loc[obs['away_team'], 'osrs'].values - wk.loc[obs['home_team'], 'dsrs'].values - home_adj
resid = np.concatenate([obs['home_score'] - home_pred, obs['away_score'] - away_pred])
n_params = 2 * len(wk)  # 32 offenses + 32 defenses + avg + hfa, minus 2 redundant directions
sigma_score = np.sqrt((resid ** 2).sum() / (len(resid) - n_params))
print(f'Margin sigma: {sigma:.2f} | Score-level sigma: {sigma_score:.2f}')

std = np.sqrt(sigma_score**2 + sigma_score**2/n_min + sigma_score**2/n_gb)

ACTUAL_MIN_SCORED = 30
print(f'Avg MIN score expected {mean}. Sigma: {std}')

# ---- Plot expected score distribution vs. actual ----
import math
import matplotlib.pyplot as plt

def norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

def norm_cdf(x, mu, sd):
    return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))

# Truncate the normal at 0 (negative scores are impossible) and renormalize so the
# remaining density still integrates to 1.
p_nonneg = 1 - norm_cdf(0, mean, std)                              # share of the normal at or above 0

def trunc_pdf(x, mu, sd):
    return np.where(np.asarray(x) >= 0, norm_pdf(x, mu, sd) / p_nonneg, 0.0)

def trunc_cdf(x, mu, sd):
    return 0.0 if x < 0 else (norm_cdf(x, mu, sd) - norm_cdf(0, mu, sd)) / p_nonneg

z = (ACTUAL_MIN_SCORED - mean) / std
p_at_least = 1 - trunc_cdf(ACTUAL_MIN_SCORED, mean, std)          # P(score >= actual | score >= 0)
p_two_sided = 2 * min(p_at_least, 1 - p_at_least)                  # as or more extreme, either direction
print(f'Share of normal below 0 (removed): {1 - p_nonneg:.1%}')
print(f'z = {z:.2f} | P(MIN >= {ACTUAL_MIN_SCORED}) = {p_at_least:.1%} | two-sided = {p_two_sided:.1%}')

x = np.linspace(0, mean + 4 * std, 500)
y = trunc_pdf(x, mean, std)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(x, y, color='#4F2683', lw=2)  # Vikings purple
tail = x >= ACTUAL_MIN_SCORED
ax.fill_between(x[tail], y[tail], color='#FFC62F', alpha=0.6, linewidth=0,
                label=f'P(score ≥ {ACTUAL_MIN_SCORED}) = {p_at_least:.1%}')
ax.axvline(mean, color='#888888', ls='--', lw=1)
ax.axvline(ACTUAL_MIN_SCORED, color='#222222', lw=2)

ymax = y.max()
ax.annotate(f'Expected: {mean:.1f}\n(σ = {std:.1f})', xy=(mean, ymax), xytext=(1, ymax * 0.9),
            fontsize=10, color='#444444', arrowprops=dict(arrowstyle='-', color='#888888'))
ax.annotate(f'Actual: {ACTUAL_MIN_SCORED}\nz = {z:+.2f}\n{p_at_least:.1%} chance of ≥ {ACTUAL_MIN_SCORED}',
            xy=(ACTUAL_MIN_SCORED, trunc_pdf(ACTUAL_MIN_SCORED, mean, std)),
            xytext=(ACTUAL_MIN_SCORED + 0.4 * std, ymax * 0.6), fontsize=10, weight='bold',
            arrowprops=dict(arrowstyle='->', color='#222222'))

ax.set_title(f'MIN points vs GB (Week {OBS_WEEK + 1}): model expectation from data through Week {OBS_WEEK}', fontsize=12)
ax.set_xlabel('MIN points scored')
ax.set_ylabel('Probability density (normal truncated at 0)')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.grid(axis='y', alpha=0.25)
ax.legend(frameon=False, loc='upper right')
fig.tight_layout()

out_path = DATA_DIR + 'predictions/min_week10_score_distribution.png'
fig.savefig(out_path, dpi=150)
print(f'Saved plot to {out_path}')
plt.show()
