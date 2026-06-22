"""
predict_suite.py
================
Fits the full model suite and predicts every target for a fixture, built to sit
on top of soccer_forecast.CountRatingModel and consume the dataframe produced by
Statsapi.get_match_stats().

TARGETS PRODUCED, per 1H / 2H / FT, for home / away / total:
    goals, corners, fouls, offsides, shots_on_target (sot), cards
PLUS match-level binaries:
    red_card  (yes/no)   -- data-driven IF you pull red cards
    penalty   (yes/no)   -- PLACEHOLDER base rate (no penalty data in the feed)

DATA REALITY (from the StatsAPI parser):
  - goals: FULL-TIME only (home_score/away_score). Half goals are THINNED from
    the FT rate, because the feed has no half-time score. If you later add
    home_goals_1h/away_goals_1h columns, they're used directly instead.
  - corners / offsides / sot / cards: real 1H/2H/FT splits.
  - fouls: FT real; halves are null in the feed -> thinned.
  - cards: built as yellow + red. ADD "reds": "red_cards" to your parser's
    `wanted` dict to include reds (and to enable the red-card binary).
  - penalty: NOT in the feed. Predicted from a base rate until you source
    penalty + referee data; treat as a placeholder.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from forcast_v1 import (
    CountRatingModel, prob_over, prob_at_least_one, btts,
    two_team_grid, total_over, prob_home_more, outcome_probs,
)

# Which stats are modelled as counts, and whether they use the Dixon-Coles
# low-score correction (goals only). Everything else leans on NegBin dispersion.
COUNT_STATS = {
    "goals":    {"use_dc": True,  "half_life": 730},
    "corners":  {"use_dc": False, "half_life": 540},
    "fouls":    {"use_dc": False, "half_life": 540},
    "offsides": {"use_dc": False, "half_life": 540},
    "sot":      {"use_dc": False, "half_life": 540},
    "cards":    {"use_dc": False, "half_life": 540},
}

# Over/under lines to report per stat (sensible defaults; tune to the market).
LINES = {
    "goals":    [0.5, 1.5, 2.5, 3.5],
    "corners":  [7.5, 8.5, 9.5, 10.5],
    "fouls":    [20.5, 22.5, 24.5],
    "offsides": [1.5, 2.5, 3.5],
    "sot":      [6.5, 7.5, 8.5],
    "cards":    [2.5, 3.5, 4.5, 5.5],
}

SEGMENTS = ("1h", "2h", "ft")

# Default first-half share when a stat has no usable half data (most lean 2H,
# so first-half share < 0.5). Used only as a fallback for thinning.
DEFAULT_FH_SHARE = {
    "goals": 0.45, "corners": 0.46, "fouls": 0.48,
    "offsides": 0.47, "sot": 0.46, "cards": 0.43,
}

# Red cards skew heavily to the second half (fatigue, game state).
RED_FH_SHARE = 0.30


# ---------------------------------------------------------------------------
# 1) Prepare the raw dataframe: build derived columns, clean, filter.
# ---------------------------------------------------------------------------
def prepare_df(df, red_weight=1):
    """
    Build goals/cards columns, parse dates, add neutral flag, filter to played
    matches. `red_weight`: count a red as this many cards (1 or 2 per market).
    """
    df = df.copy()

    # dates -> datetime, tz-naive (fit_all_models casts to datetime64[D])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)

    # goals: FT = regulation score (IGNORE final_score; it includes shootouts).
    df["home_goals_ft"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_goals_ft"] = pd.to_numeric(df["away_score"], errors="coerce")

    # half-time goals -> fit goal halves DIRECTLY (you have these columns).
    if "home_score_1h" in df.columns and "away_score_1h" in df.columns:
        df["home_goals_1h"] = pd.to_numeric(df["home_score_1h"], errors="coerce")
        df["away_goals_1h"] = pd.to_numeric(df["away_score_1h"], errors="coerce")
        df["home_goals_2h"] = df["home_goals_ft"] - df["home_goals_1h"]
        df["away_goals_2h"] = df["away_goals_ft"] - df["away_goals_1h"]

    # cards = yellows + red_weight * reds. NaN in either component -> NaN
    # (a failed stats call must not fabricate a 0-card observation).
    for seg in SEGMENTS:
        for side in ("home", "away"):
            y = pd.to_numeric(df.get(f"{side}_yellows_{seg}"), errors="coerce")
            r = pd.to_numeric(df.get(f"{side}_reds_{seg}"), errors="coerce")
            df[f"{side}_cards_{seg}"] = y + red_weight * r

    # neutral flag: default False (qualifiers/friendlies have a home side).
    if "neutral" not in df.columns:
        df["neutral"] = False

    # keep only finished matches (scores present)
    df = df[df["home_goals_ft"].notna() & df["away_goals_ft"].notna()].copy()
    return df


# ---------------------------------------------------------------------------
# 2) Empirical first-half shares (for thinning when a half model is missing).
# ---------------------------------------------------------------------------
def first_half_shares(df):
    shares = {}
    for stat in COUNT_STATS:
        h1, a1 = f"home_{stat}_1h", f"away_{stat}_1h"
        hf, af = f"home_{stat}_ft", f"away_{stat}_ft"
        if all(c in df.columns for c in (h1, a1, hf, af)):
            first = (df[h1] + df[a1]).sum(skipna=True)
            full = (df[hf] + df[af]).sum(skipna=True)
            shares[stat] = float(first / full) if full and full > 0 else DEFAULT_FH_SHARE[stat]
        else:
            shares[stat] = DEFAULT_FH_SHARE[stat]
    return shares


# ---------------------------------------------------------------------------
# 3) Fit the whole suite. Returns models keyed (stat, seg) + the shares.
# ---------------------------------------------------------------------------
def fit_all_models(df, min_rows=300):
    shares = first_half_shares(df)
    models = {}
    for stat, cfg in COUNT_STATS.items():
        for seg in SEGMENTS:
            hcol, acol = f"home_{stat}_{seg}", f"away_{stat}_{seg}"
            if hcol not in df.columns or acol not in df.columns:
                continue
            sub = df.dropna(subset=[hcol, acol])
            if len(sub) < min_rows:
                continue  # too sparse to fit directly -> will thin from ft
            m = CountRatingModel(use_dc=cfg["use_dc"], half_life_days=cfg["half_life"])
            m.fit(sub["home_id"].values, sub["away_id"].values,
                  sub[hcol].values, sub[acol].values,
                  dates=sub["date"].values.astype("datetime64[D]"),
                  neutral=sub["neutral"].values)
            models[(stat, seg)] = m
    return models, shares


# ---------------------------------------------------------------------------
# 4) Per-team red-card rate (reds are too rare for a full rating model).
# ---------------------------------------------------------------------------
def red_rates(df):
    """team_id -> reds committed per game; plus a global base rate."""
    has = "home_reds_ft" in df.columns and "away_reds_ft" in df.columns
    if not has:
        return None, None
    counts, games = {}, {}
    for _, r in df.iterrows():
        for side, opp_field in (("home", "home_reds_ft"), ("away", "away_reds_ft")):
            tid = r[f"{side}_id"]
            val = r[opp_field]
            if pd.notna(val):
                counts[tid] = counts.get(tid, 0) + val
                games[tid] = games.get(tid, 0) + 1
    rates = {t: counts[t] / games[t] for t in counts if games[t] > 0}
    base = np.mean(list(rates.values())) if rates else 0.10
    return rates, base


# ---------------------------------------------------------------------------
# 5) Rate for one stat+segment: direct model if it exists, else thin the FT.
# ---------------------------------------------------------------------------
def _seg_rates(models, shares, stat, seg, home, away, neutral):
    if (stat, seg) in models:
        return models[(stat, seg)].rates(home, away, neutral=neutral)
    # no direct half model -> thin the FT rate
    if (stat, "ft") not in models:
        return None, None
    lh_ft, la_ft = models[(stat, "ft")].rates(home, away, neutral=neutral)
    s = shares.get(stat, 0.45)
    frac = s if seg == "1h" else (1 - s) if seg == "2h" else 1.0
    return lh_ft * frac, la_ft * frac


# ---------------------------------------------------------------------------
# 6) Predict everything for one fixture.
# ---------------------------------------------------------------------------
def predict_fixture(models, shares, home, away, neutral=True,
                    red_rates_=None, red_base=None, penalty_base=0.25):
    """
    Returns a nested dict: out[stat][seg] = {home, away, total, over:{...}, ...}
    plus out['red_card'][seg] and out['penalty'][seg] as P(at least one).
    `home`/`away` are team IDs. neutral=True for World Cup fixtures.
    """
    out = {}

    for stat in COUNT_STATS:
        out[stat] = {}
        for seg in SEGMENTS:
            lh, la = _seg_rates(models, shares, stat, seg, home, away, neutral)
            if lh is None:
                continue
            alpha = models.get((stat, seg), models.get((stat, "ft"))).alpha_
            rho = 0.0
            if stat == "goals" and (stat, seg) in models and models[(stat, seg)].use_dc:
                rho = float(models[(stat, seg)].params_[-1])

            entry = {
                "home_exp": round(lh, 3),
                "away_exp": round(la, 3),
                "total_exp": round(lh + la, 3),
                "over": {str(L): round(total_over(L, lh, la, alpha,
                          use_dc=(rho != 0), rho=rho), 3) for L in LINES[stat]},
                "home_more": round(prob_home_more(lh, la, alpha), 3),
            }
            # goals & sot get BTTS-style and match result
            if stat in ("goals", "sot"):
                entry["btts"] = round(btts(lh, la, alpha), 3)
            if stat == "goals":
                entry["result"] = {k: round(v, 3) for k, v in
                                   outcome_probs(lh, la, alpha, use_dc=(rho != 0),
                                                 rho=rho).items()}
            out[stat][seg] = entry

    # --- red card binary (data-driven if reds were pulled) ---
    out["red_card"] = {}
    if red_rates_ is not None:
        hr = red_rates_.get(home, red_base)
        ar = red_rates_.get(away, red_base)
        lam_ft = hr + ar
        for seg in SEGMENTS:
            frac = RED_FH_SHARE if seg == "1h" else (1 - RED_FH_SHARE) if seg == "2h" else 1.0
            out["red_card"][seg] = round(1 - np.exp(-lam_ft * frac), 3)
    else:
        out["red_card"] = None  # no red data pulled yet

    # --- penalty binary (PLACEHOLDER base rate; needs real data) ---
    out["penalty"] = {}
    for seg in SEGMENTS:
        frac = 0.45 if seg == "1h" else 0.55 if seg == "2h" else 1.0
        lam = penalty_base * frac
        out["penalty"][seg] = round(1 - np.exp(-lam), 3)
    out["penalty"]["_note"] = "placeholder base rate; source penalty/referee data"

    return out


# ---------------------------------------------------------------------------
# Demo on synthetic data shaped like the StatsAPI dataframe.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    teams = [f"tm_{i}" for i in range(40)]
    atk = dict(zip(teams, rng.normal(0, 0.35, len(teams))))
    dfn = dict(zip(teams, rng.normal(0, 0.25, len(teams))))

    recs = []
    base = np.datetime64("2022-06-01")
    for _ in range(5123):
        h, a = rng.choice(teams, 2, replace=False)
        lh_g = np.exp(0.1 + 0.2 + atk[h] + dfn[a])
        la_g = np.exp(0.1 + atk[a] + dfn[h])
        hg, ag = int(rng.poisson(lh_g)), int(rng.poisson(la_g))
        h1, a1 = rng.binomial(hg, 0.45), rng.binomial(ag, 0.45)   # half-time goals
        d = str(base + np.timedelta64(int(rng.integers(0, 1100)), "D")) + "T16:00:00.000Z"
        rec = {"match_id": "m", "comp_id": "c", "season_id": "s", "date": d,
               "home_name": h, "home_id": h, "away_name": a, "away_id": a,
               "home_score": hg, "away_score": ag,
               "home_score_1h": h1, "away_score_1h": a1}

        # ~27% of matches have NO stats block (mirrors the 1362/5123 gap)
        has_stats = rng.random() > 0.27
        has_halves = has_stats and rng.random() > 0.18   # some FT-only
        for stat, mult in (("corners", 5), ("fouls", 12), ("offsides", 1.5),
                           ("sot", 4), ("yellows", 1.5), ("reds", 0.06)):
            for side in ("home", "away"):
                if not has_stats:
                    ft = np.nan; f1 = np.nan; f2 = np.nan
                else:
                    ft = rng.poisson(mult)
                    if has_halves and stat != "fouls":     # fouls halves ~always null
                        f1 = rng.binomial(ft, 0.46); f2 = ft - f1
                    else:
                        f1 = np.nan; f2 = np.nan
                rec[f"{side}_{stat}_ft"] = ft
                rec[f"{side}_{stat}_1h"] = f1
                rec[f"{side}_{stat}_2h"] = f2
        recs.append(rec)
    df = pd.DataFrame(recs)

    df = prepare_df(df)
    models, shares = fit_all_models(df)
    rrates, rbase = red_rates(df)

    print("rows after prepare:", len(df))
    print("fitted models:", sorted(models.keys()))
    print("goals halves fit directly?",
          ("goals", "1h") in models and ("goals", "2h") in models)

    # SANITY CHECK: do strong teams get high attack ratings?
    gm = models[("goals", "ft")]
    mu, ha, a_, d_, rho = gm._unpack(gm.params_, len(gm.teams_))
    fitted_atk = dict(zip(gm.teams_, a_))
    # correlation between TRUE and FITTED attack should be strongly positive
    common = [t for t in teams if t in fitted_atk]
    corr = np.corrcoef([atk[t] for t in common],
                       [fitted_atk[t] for t in common])[0, 1]
    print(f"corr(true attack, fitted attack) = {corr:.2f}   (want > 0.8)")

    pred = predict_fixture(models, shares, teams[0], teams[5],
                           neutral=True, red_rates_=rrates, red_base=rbase)
    print("\n--- GOALS ---")
    for seg in SEGMENTS:
        print(seg, pred["goals"][seg])
    print("\n--- CORNERS ft ---", pred["corners"]["ft"])
    print("--- CARDS ft   ---", pred["cards"]["ft"])
    print("--- RED CARD   ---", pred["red_card"])
    print("--- PENALTY    ---", pred["penalty"])