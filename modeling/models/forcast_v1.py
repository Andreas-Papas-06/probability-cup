"""
forcast_v1.py
==================
A unified, count-based forecasting engine for soccer match markets, built for
calibrated-probability competitions (e.g. weighted-Brier-scored contests).

WHY ONE ENGINE FOR MANY MARKETS
--------------------------------
Goals, corners, fouls, offsides and shots-on-target are all COUNT processes:
"how many discrete events happened in a time window." So they all share the
same model. For any statistic S and any match we estimate an expected rate
(lambda) for each team using a multiplicative attack x defense structure:

    log E[home_count] = mu + home_adv + atk[home] + def_[away]
    log E[away_count] = mu           + atk[away] + def_[home]

Each statistic is fit independently (a team strong at winning corners may be
weak at avoiding fouls). Once you have (lambda_home, lambda_away) for a stat,
EVERY market on that stat is read off the resulting count distribution:
    over/under .......... P(X >= k)
    both teams >=1 ...... P(home>=1) * P(away>=1)
    team A > team B ..... sum of grid cells where A > B
    penalty yes/no ...... 1 - exp(-lambda_pen)        [rare-event rate]

SEGMENTS (1H / 2H / FT)
-----------------------
Halves are NOT interchangeable (2H has more goals, fouls, subs). Fit one model
per (stat, segment). If 1H ~ Poisson(l1) and 2H ~ Poisson(l2) independently,
the full game is Poisson(l1 + l2) -- just add the rates. If you only have
full-game data, THIN it: 1H ~ Poisson(s*lambda), 2H ~ Poisson((1-s)*lambda),
with s the empirical first-half share.

DISTRIBUTIONS
-------------
Goals: Poisson with the Dixon-Coles low-score correction (use_dc=True).
Corners / fouls / offsides / shots: overdispersed -> Negative Binomial; the
dispersion alpha is estimated from data (var = mu + alpha*mu^2).

This module fits the rate structure by weighted maximum likelihood (recent
matches weighted more via an exponential half-life) and exposes market readouts.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson, nbinom


# ----------------------------------------------------------------------------- 
# Dixon-Coles low-score dependence correction (goals only)
# -----------------------------------------------------------------------------
def _dc_tau(yh, ya, lh, la, rho):
    """Multiplicative correction for (0,0),(0,1),(1,0),(1,1); 1 elsewhere."""
    out = np.ones_like(lh, dtype=float)
    m = (yh == 0) & (ya == 0); out[m] = 1.0 - lh[m] * la[m] * rho
    m = (yh == 0) & (ya == 1); out[m] = 1.0 + lh[m] * rho
    m = (yh == 1) & (ya == 0); out[m] = 1.0 + la[m] * rho
    m = (yh == 1) & (ya == 1); out[m] = 1.0 - rho
    return np.clip(out, 1e-9, None)


# ----------------------------------------------------------------------------- 
# Core rating model -- fit ONE per (statistic, segment)
# -----------------------------------------------------------------------------
class CountRatingModel:
    """
    Attack/defense rate model for a single per-team count statistic.

    Parameters
    ----------
    use_dc : bool       Dixon-Coles correction (set True only for goals).
    half_life_days : float
        Recency weighting: a match this many days old counts half as much.
        Set None to disable time weighting.
    ridge : float       Small L2 penalty on attack/defense for stability.
    """

    def __init__(self, use_dc=False, half_life_days=365.0, ridge=1e-3):
        self.use_dc = use_dc
        self.half_life_days = half_life_days
        self.ridge = ridge
        self.teams_ = None
        self.idx_ = None
        self.params_ = None
        self.alpha_ = 0.0          # NegBin dispersion (0 -> Poisson)

    # -- internals ----------------------------------------------------------
    def _unpack(self, theta, n):
        mu = theta[0]
        home_adv = theta[1]
        atk = theta[2:2 + n]
        dfn = theta[2 + n:2 + 2 * n]
        atk = atk - atk.mean()     # identifiability: sum-to-zero
        dfn = dfn - dfn.mean()
        rho = theta[-1] if self.use_dc else 0.0
        return mu, home_adv, atk, dfn, rho

    def _neg_ll(self, theta, hi, ai, yh, ya, w, neutral, n):
        mu, home_adv, atk, dfn, rho = self._unpack(theta, n)
        ha = np.where(neutral, 0.0, home_adv)
        lh = np.exp(mu + ha + atk[hi] + dfn[ai])
        la = np.exp(mu + atk[ai] + dfn[hi])
        ll = w * (yh * np.log(lh) - lh + ya * np.log(la) - la)
        if self.use_dc:
            ll = ll + w * np.log(_dc_tau(yh, ya, lh, la, rho))
        pen = self.ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))
        return -(ll.sum()) + pen

    # -- fit ----------------------------------------------------------------
    def fit(self, home, away, home_count, away_count, dates=None, neutral=None):
        """
        home, away          : arrays of team names
        home_count, away_count : observed counts of THIS stat in THIS segment
        dates               : array of np.datetime64 (for recency weighting)
        neutral             : bool array (True drops home advantage)
        """
        home = np.asarray(home); away = np.asarray(away)
        yh = np.asarray(home_count, float); ya = np.asarray(away_count, float)
        m = len(home)
        neutral = np.zeros(m, bool) if neutral is None else np.asarray(neutral, bool)

        teams = sorted(set(home) | set(away))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = np.array([idx[t] for t in home])
        ai = np.array([idx[t] for t in away])

        if dates is not None and self.half_life_days:
            d = np.asarray(dates, dtype="datetime64[D]")
            age = (d.max() - d) / np.timedelta64(1, "D")
            w = 0.5 ** (age / self.half_life_days)
        else:
            w = np.ones(m)

        theta0 = np.concatenate([[np.log(max(yh.mean(), 0.1)), 0.1],
                                 np.zeros(n), np.zeros(n),
                                 ([-0.05] if self.use_dc else [])])
        res = minimize(self._neg_ll, theta0,
                       args=(hi, ai, yh, ya, w, neutral, n),
                       method="L-BFGS-B")

        self.teams_, self.idx_, self.params_ = teams, idx, res.x

        # NegBin dispersion via method of moments on fitted means (pooled)
        mu, ha_, atk, dfn, rho = self._unpack(res.x, n)
        ha = np.where(neutral, 0.0, ha_)
        mh = np.exp(mu + ha + atk[hi] + dfn[ai])
        ma = np.exp(mu + atk[ai] + dfn[hi])
        y = np.concatenate([yh, ya]); mhat = np.concatenate([mh, ma])
        num = np.mean((y - mhat) ** 2 - mhat)
        den = np.mean(mhat ** 2)
        self.alpha_ = max(num / den, 0.0) if den > 0 else 0.0
        return self

    # -- predict ------------------------------------------------------------
    def rates(self, home, away, neutral=False):
        """Return (lambda_home, lambda_away) expected counts for a fixture."""
        mu, ha_, atk, dfn, rho = self._unpack(self.params_, len(self.teams_))
        h, a = self.idx_[home], self.idx_[away]
        ha = 0.0 if neutral else ha_
        lh = float(np.exp(mu + ha + atk[h] + dfn[a]))
        la = float(np.exp(mu + atk[a] + dfn[h]))
        return lh, la


# ----------------------------------------------------------------------------- 
# Distribution helpers (mean-dispersion Negative Binomial; Poisson if alpha=0)
# -----------------------------------------------------------------------------
def _nb_params(mu, alpha):
    r = 1.0 / alpha
    p = r / (r + mu)
    return r, p

def pmf(k, mu, alpha=0.0):
    if alpha <= 0:
        return poisson.pmf(k, mu)
    r, p = _nb_params(mu, alpha)
    return nbinom.pmf(k, r, p)

def prob_over(line, mu, alpha=0.0):
    """P(count > line). For a .5 line, line=2.5 -> P(X>=3)."""
    k = int(np.floor(line))
    if alpha <= 0:
        return float(poisson.sf(k, mu))
    r, p = _nb_params(mu, alpha)
    return float(nbinom.sf(k, r, p))

def prob_under(line, mu, alpha=0.0):
    return 1.0 - prob_over(line, mu, alpha)

def prob_at_least_one(mu, alpha=0.0):
    return 1.0 - pmf(0, mu, alpha)


# ----------------------------------------------------------------------------- 
# Two-team grid -> any home-vs-away market (BTTS, totals, comparisons, scores)
# -----------------------------------------------------------------------------
def two_team_grid(lh, la, alpha=0.0, max_n=15, use_dc=False, rho=0.0):
    """Joint P[x, y] over home count x and away count y."""
    x = np.arange(max_n + 1)
    ph = pmf(x, lh, alpha)
    pa = pmf(x, la, alpha)
    grid = np.outer(ph, pa)
    if use_dc:
        xs = x[:, None] * np.ones((1, max_n + 1))
        ys = np.ones((max_n + 1, 1)) * x[None, :]
        grid = grid * _dc_tau(xs.astype(int), ys.astype(int),
                              np.full_like(grid, lh), np.full_like(grid, la), rho)
    return grid / grid.sum()

def btts(lh, la, alpha=0.0):
    """Both teams record at least one (independent)."""
    return prob_at_least_one(lh, alpha) * prob_at_least_one(la, alpha)

def total_over(line, lh, la, alpha=0.0, max_n=15, use_dc=False, rho=0.0):
    """P(home + away > line)."""
    g = two_team_grid(lh, la, alpha, max_n, use_dc, rho)
    tot = np.add.outer(np.arange(max_n + 1), np.arange(max_n + 1))
    return float(g[tot > line].sum())

def prob_home_more(lh, la, alpha=0.0, max_n=15):
    """P(home count > away count), e.g. 'more corners'."""
    g = two_team_grid(lh, la, alpha, max_n)
    i = np.arange(max_n + 1)
    return float(g[i[:, None] > i[None, :]].sum())

def outcome_probs(lh, la, alpha=0.0, max_n=10, use_dc=False, rho=0.0):
    """Win/draw/loss from a goals grid (use_dc=True for goals)."""
    g = two_team_grid(lh, la, alpha, max_n, use_dc, rho)
    i = np.arange(max_n + 1)
    home = g[i[:, None] > i[None, :]].sum()
    draw = np.trace(g)
    away = g[i[:, None] < i[None, :]].sum()
    return {"home": float(home), "draw": float(draw), "away": float(away)}


# ----------------------------------------------------------------------------- 
# Segment composition
# -----------------------------------------------------------------------------
def ft_from_halves(l1, l2):
    """Full-game rate = sum of half rates (independent-Poisson assumption)."""
    return l1 + l2

def split_ft_to_halves(lambda_ft, first_half_share=0.45):
    """Thin a full-game rate into (1H, 2H) rates."""
    return lambda_ft * first_half_share, lambda_ft * (1 - first_half_share)


# ----------------------------------------------------------------------------- 
# Market anchoring: de-vig bookmaker odds (proportional method)
# -----------------------------------------------------------------------------
def devig(decimal_odds):
    """Decimal odds -> de-vigged probabilities (normalize out the margin)."""
    inv = np.array([1.0 / o for o in decimal_odds])
    return inv / inv.sum()

def blend(model_prob, market_prob, w_market=0.6):
    """Weighted blend of model and (de-vigged) market probabilities."""
    p = w_market * np.asarray(market_prob) + (1 - w_market) * np.asarray(model_prob)
    return p / p.sum() if p.ndim and p.sum() > 0 else p


# ----------------------------------------------------------------------------- 
# Demo on synthetic data
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D", "E", "F"]
    true_atk = {t: a for t, a in zip(teams, [.4, .2, 0, -.1, -.2, -.3])}
    true_def = {t: d for t, d in zip(teams, [-.3, -.1, 0, .1, .2, .3])}

    H, Aw, HG, AG, D, NEU = [], [], [], [], [], []
    base = np.datetime64("2025-06-01")
    for k in range(1200):
        h, a = rng.choice(teams, 2, replace=False)
        lh = np.exp(0.1 + 0.25 + true_atk[h] + true_def[a])   # 0.25 home edge
        la = np.exp(0.1 + true_atk[a] + true_def[h])
        H.append(h); Aw.append(a)
        HG.append(rng.poisson(lh)); AG.append(rng.poisson(la))
        D.append(base + np.timedelta64(int(rng.integers(0, 365)), "D"))
        NEU.append(False)

    # Fit the GOALS model (Dixon-Coles on); same call works for any stat/segment.
    gm = CountRatingModel(use_dc=True, half_life_days=200).fit(
        H, Aw, HG, AG, dates=D, neutral=NEU)

    lh, la = gm.rates("A", "E", neutral=True)     # strong vs weak, neutral venue
    print(f"Expected goals  A {lh:.2f}  -  {la:.2f} E   (dispersion alpha={gm.alpha_:.3f})")
    print("Match result   :", {k: round(v, 3)
          for k, v in outcome_probs(lh, la, use_dc=True,
                                    rho=gm.params_[-1]).items()})
    print("Over 2.5 goals :", round(total_over(2.5, lh, la, use_dc=True,
                                                rho=gm.params_[-1]), 3))
    print("BTTS           :", round(btts(lh, la), 3))
    print("A scores 1H+   :", round(prob_at_least_one(lh * 0.45), 3),
          "(thinned to first-half rate)")

    # Penalty yes/no from a rare-event rate (you'd fit lambda_pen the same way).
    lam_pen = 0.55
    print("Penalty in game:", round(1 - np.exp(-lam_pen), 3))

    # Anchoring to a book: de-vig and blend.
    mkt = devig([1.55, 4.2, 6.0])                 # home / draw / away decimals
    print("De-vigged market:", {k: round(v, 3) for k, v in
          zip(["home", "draw", "away"], mkt)})