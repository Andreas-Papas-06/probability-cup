"""
backtest2.py
==============
Head-to-head: does training the rate model on xG predict ACTUAL goal markets
better than training on raw goals? Both models are scored against real goals
(result / over2.5 / BTTS) on the SAME test matches, so the comparison isolates
signal quality from sample size.
 
Why this could help: a single match's goal count is a noisy estimate of a team's
true scoring rate; xG is a smoother estimate of the same rate. Fitting ratings on
xG can recover the rate better -> better goal predictions, especially on totals.
 
Caveats handled here:
  - xG is continuous, so the xG model fits with use_dc=False (the Dixon-Coles
    low-score correction is integer-specific and meaningless on xG).
  - When reading GOAL markets from xG-derived rates we use a plain Poisson
    (alpha=0): actual goals are ~Poisson around their rate; the xG model's own
    dispersion reflects xG variance, not goal variance.
  - Coverage cost (matches the xG model can't predict at all) is reported
    separately -- it does NOT enter the head-to-head Brier.
"""
 
from __future__ import annotations
import numpy as np
import pandas as pd
from modeling.models.predict import CountRatingModel, outcome_probs, total_over, btts
from backtest import base_rates
 
 
def fit_rate(df, half_life, target, use_dc, min_rows=300):
    """Fit attack/defense rate models on `target` (e.g. 'goals' or 'xg')."""
    models = {}
    for seg in ("ft",):  # head-to-head on FT goal markets
        hcol, acol = f"home_{target}_{seg}", f"away_{target}_{seg}"
        if hcol not in df.columns:
            continue
        sub = df.dropna(subset=[hcol, acol])
        if len(sub) < min_rows:
            continue
        m = CountRatingModel(use_dc=use_dc, half_life_days=half_life)
        m.fit(sub["home_id"].values, sub["away_id"].values,
              sub[hcol].values, sub[acol].values,
              dates=sub["date"].values.astype("datetime64[D]"),
              neutral=sub["neutral"].values)
        models[seg] = m
    return models
 
 
def _probs(model, row, use_dc):
    """Goal-market probabilities from a rate model, or None if a team is unseen."""
    try:
        lh, la = model.rates(row["home_id"], row["away_id"], neutral=bool(row["neutral"]))
    except KeyError:
        return None
    rho = float(model.params_[-1]) if use_dc else 0.0
    alpha = model.alpha_ if use_dc else 0.0   # goals model: own alpha(~0); xG: plain Poisson
    return {
        "result": outcome_probs(lh, la, alpha, use_dc=use_dc, rho=rho),
        "over2.5": total_over(2.5, lh, la, alpha, use_dc=use_dc, rho=rho),
        "btts": btts(lh, la, alpha),
    }
 
 
def _se(probs, row):
    hg, ag = row["home_goals_ft"], row["away_goals_ft"]
    y_res = {"home": int(hg > ag), "draw": int(hg == ag), "away": int(hg < ag)}
    y_over = int((hg + ag) > 2.5)
    y_btts = int((hg >= 1) and (ag >= 1))
    return {
        "result": sum((probs["result"][k] - y_res[k]) ** 2 for k in y_res),
        "over2.5": (probs["over2.5"] - y_over) ** 2,
        "btts": (probs["btts"] - y_btts) ** 2,
    }
 
 
def compare(df, cutoff, half_life=1095):
    cutoff = pd.Timestamp(cutoff)
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
 
    gm = fit_rate(train, half_life, target="goals", use_dc=True)["ft"]
    xm = fit_rate(train, half_life, target="xg", use_dc=False)["ft"]
    base = base_rates(train)
    markets = ["result", "over2.5", "btts"]
 
    g_acc = {m: [] for m in markets}
    x_acc = {m: [] for m in markets}
    b_acc = {m: [] for m in markets}
    common = xg_only_skipped = 0
 
    for _, row in test.iterrows():
        if pd.isna(row["home_goals_ft"]) or pd.isna(row["away_goals_ft"]):
            continue
        pg = _probs(gm, row, use_dc=True)
        px = _probs(xm, row, use_dc=False)
        if pg is None:
            continue
        if px is None:
            xg_only_skipped += 1   # goals model could predict, xG model couldn't
            continue
        # both can predict -> fair head-to-head on this match
        for m in markets:
            g_acc[m].append(_se(pg, row)[m])
            x_acc[m].append(_se(px, row)[m])
            y = {"home": int(row["home_goals_ft"] > row["away_goals_ft"]),
                 "draw": int(row["home_goals_ft"] == row["away_goals_ft"]),
                 "away": int(row["home_goals_ft"] < row["away_goals_ft"])}
            if m == "result":
                b_acc[m].append(sum((base["result"][k] - y[k]) ** 2 for k in y))
            elif m == "over2.5":
                b_acc[m].append((base["over2.5"] -
                                 int((row["home_goals_ft"] + row["away_goals_ft"]) > 2.5)) ** 2)
            else:
                b_acc[m].append((base["btts"] -
                                 int((row["home_goals_ft"] >= 1) and (row["away_goals_ft"] >= 1))) ** 2)
        common += 1
 
    print(f"common test matches scored : {common}")
    print(f"xG-model couldn't predict  : {xg_only_skipped}  "
          f"(coverage cost -- goals model handled these, xG model could not)")
    print("\n             GOALS-model   xG-model    BASELINE")
    for m in markets:
        g, x, b = np.mean(g_acc[m]), np.mean(x_acc[m]), np.mean(b_acc[m])
        win = "xG better" if x < g else "goals better"
        print(f"  {m:8s}    {g:.4f}      {x:.4f}     {b:.4f}   ({win})")
    print(f"\n  weighted    {np.mean([np.mean(g_acc[m]) for m in markets]):.4f}      "
          f"{np.mean([np.mean(x_acc[m]) for m in markets]):.4f}")
 
 
# ---------------------------------------------------------------------------
# Demo: goals are a NOISY draw from the true rate; xG is a SMOOTHER estimate.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    teams = [f"tm_{i}" for i in range(40)]
    atk = dict(zip(teams, rng.normal(0, 0.35, 40)))
    dfn = dict(zip(teams, rng.normal(0, 0.25, 40)))
 
    base0 = np.datetime64("2022-06-01")
    recs = []
    for _ in range(6000):
        day = int(rng.integers(0, 1100))
        h, a = rng.choice(teams, 2, replace=False)
        lh = np.exp(0.1 + 0.2 + atk[h] + dfn[a])
        la = np.exp(0.1 + atk[a] + dfn[h])
        hg, ag = int(rng.poisson(lh)), int(rng.poisson(la))    # noisy actual goals
        # xG = a smoother estimate of the same rate (low multiplicative noise)
        hxg = lh * np.exp(rng.normal(0, 0.22))
        axg = la * np.exp(rng.normal(0, 0.22))
        # ~25% of matches have NO xG (coverage gap)
        if rng.random() < 0.25:
            hxg = axg = np.nan
        d = str(base0 + np.timedelta64(day, "D"))
        recs.append({"home_id": h, "away_id": a, "neutral": False, "date": d,
                     "home_goals_ft": hg, "away_goals_ft": ag,
                     "home_xg_ft": hxg, "away_xg_ft": axg})
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].quantile(0.8)
 
    print(f"cutoff: {cutoff.date()}\n")
    compare(df, cutoff, half_life=730)