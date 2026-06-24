"""
backtest.py
===========
Leakage-free validation for the goals model (extends to any stat), scored with
weighted Brier -- the metric these competitions use.

TWO PRINCIPLES THIS ENFORCES
  1. No leakage: to predict a match on date D, fit ONLY on matches before D.
     We use a time cutoff (train < cutoff, test >= cutoff). A model trained on
     future results would score unrealistically well and lie to you.
  2. A baseline: Brier is only meaningful relative to "no skill". We score a
     constant base-rate forecast (the train-set frequencies) and report both,
     so you can see how much the model actually adds.

Brier scores (LOWER is better):
  - multiclass (1X2 result): sum_k (p_k - y_k)^2   over {home,draw,away}, in [0,2]
  - binary (over/under, BTTS): (p - y)^2                                  in [0,1]

The half-life sweep refits goals across a grid of half_life_days and reports
held-out Brier for each, so you can read off the optimum instead of guessing.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from modeling.models.predict import CountRatingModel, outcome_probs, total_over, btts


# ---------------------------------------------------------------------------
# Fit just the goals models (1h/2h/ft) at a given half-life, on a train frame.
# ---------------------------------------------------------------------------
def fit_goals(df, half_life, min_rows=300):
    models = {}
    for seg in ("1h", "2h", "ft"):
        hcol, acol = f"home_goals_{seg}", f"away_goals_{seg}"
        if hcol not in df.columns:
            continue
        sub = df.dropna(subset=[hcol, acol])
        if len(sub) < min_rows:
            continue
        m = CountRatingModel(use_dc=True, half_life_days=half_life)
        m.fit(sub["home_id"].values, sub["away_id"].values,
              sub[hcol].values, sub[acol].values,
              dates=sub["date"].values.astype("datetime64[D]"),
              neutral=sub["neutral"].values)
        models[seg] = m
    return models


# ---------------------------------------------------------------------------
# Base-rate baseline computed from the TRAIN set (no-skill reference).
# ---------------------------------------------------------------------------
def base_rates(df):
    tg = df["home_goals_ft"] + df["away_goals_ft"]
    res = np.select(
        [df["home_goals_ft"] > df["away_goals_ft"],
         df["home_goals_ft"] == df["away_goals_ft"]],
        ["home", "draw"], default="away")
    return {
        "result": {"home": np.mean(res == "home"),
                   "draw": np.mean(res == "draw"),
                   "away": np.mean(res == "away")},
        "over2.5": np.mean(tg > 2.5),
        "btts": np.mean((df["home_goals_ft"] >= 1) & (df["away_goals_ft"] >= 1)),
    }


# ---------------------------------------------------------------------------
# Score one test match: return squared-error contributions per market.
# ---------------------------------------------------------------------------
def _match_brier(gm, row, base):
    """gm = fitted goals 'ft' model. Returns (model_se, base_se) dicts or None."""
    home, away = row["home_id"], row["away_id"]
    try:
        lh, la = gm.rates(home, away, neutral=bool(row["neutral"]))
    except KeyError:
        return None  # team unseen in training -> can't predict, skip

    rho = float(gm.params_[-1]) if gm.use_dc else 0.0
    alpha = gm.alpha_

    # actual outcomes
    hg, ag = row["home_goals_ft"], row["away_goals_ft"]
    y_res = {"home": int(hg > ag), "draw": int(hg == ag), "away": int(hg < ag)}
    y_over = int((hg + ag) > 2.5)
    y_btts = int((hg >= 1) and (ag >= 1))

    # model probabilities
    p_res = outcome_probs(lh, la, alpha, use_dc=(rho != 0), rho=rho)
    p_over = total_over(2.5, lh, la, alpha, use_dc=(rho != 0), rho=rho)
    p_btts = btts(lh, la, alpha)

    model_se = {
        "result": sum((p_res[k] - y_res[k]) ** 2 for k in y_res),
        "over2.5": (p_over - y_over) ** 2,
        "btts": (p_btts - y_btts) ** 2,
    }
    base_se = {
        "result": sum((base["result"][k] - y_res[k]) ** 2 for k in y_res),
        "over2.5": (base["over2.5"] - y_over) ** 2,
        "btts": (base["btts"] - y_btts) ** 2,
    }
    return model_se, base_se


# ---------------------------------------------------------------------------
# Run one train/test split and score it.
# ---------------------------------------------------------------------------
def run_backtest(df, cutoff, half_life=730, weights=None):
    """
    cutoff : date string/Timestamp. Train on date < cutoff, test on date >= cutoff.
    weights: {market: weight} for the weighted Brier total (default equal).
    """
    cutoff = pd.Timestamp(cutoff)
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]

    gm_models = fit_goals(train, half_life)
    if "ft" not in gm_models:
        raise ValueError("not enough training data to fit goals")
    gm = gm_models["ft"]
    base = base_rates(train)

    markets = ["result", "over2.5", "btts"]
    model_acc = {m: [] for m in markets}
    base_acc = {m: [] for m in markets}
    scored = skipped = 0

    for _, row in test.iterrows():
        if pd.isna(row["home_goals_ft"]) or pd.isna(row["away_goals_ft"]):
            continue
        out = _match_brier(gm, row, base)
        if out is None:
            skipped += 1
            continue
        m_se, b_se = out
        for m in markets:
            model_acc[m].append(m_se[m])
            base_acc[m].append(b_se[m])
        scored += 1

    model_brier = {m: float(np.mean(model_acc[m])) for m in markets}
    base_brier = {m: float(np.mean(base_acc[m])) for m in markets}

    weights = weights or {m: 1.0 for m in markets}
    wsum = sum(weights[m] for m in markets)
    w_model = sum(weights[m] * model_brier[m] for m in markets) / wsum
    w_base = sum(weights[m] * base_brier[m] for m in markets) / wsum

    return {
        "n_train": len(train), "n_test_scored": scored, "n_skipped_unseen": skipped,
        "model": model_brier, "baseline": base_brier,
        "weighted_model": w_model, "weighted_baseline": w_base,
    }


# ---------------------------------------------------------------------------
# Sweep half-life: refit goals at each value, report held-out Brier.
# ---------------------------------------------------------------------------
def sweep_half_life(df, cutoff, values=(180, 365, 540, 730, 1095, 1460)):
    rows = []
    for hl in values:
        r = run_backtest(df, cutoff, half_life=hl)
        rows.append({"half_life_days": hl,
                     "result_brier": round(r["model"]["result"], 4),
                     "over2.5_brier": round(r["model"]["over2.5"], 4),
                     "weighted_brier": round(r["weighted_model"], 4)})
    out = pd.DataFrame(rows)
    best = out.loc[out["weighted_brier"].idxmin(), "half_life_days"]
    return out, int(best)


# ---------------------------------------------------------------------------
# Demo: synthetic data with DRIFTING strengths (so half-life has a real optimum).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    teams = [f"tm_{i}" for i in range(40)]
    n_days = 1100
    # each team's attack/defense random-walks over time -> recency matters
    drift_a = {t: np.cumsum(rng.normal(0, 0.015, n_days)) + rng.normal(0, 0.3)
               for t in teams}
    drift_d = {t: np.cumsum(rng.normal(0, 0.012, n_days)) + rng.normal(0, 0.2)
               for t in teams}

    base0 = np.datetime64("2022-06-01")
    recs = []
    for _ in range(6000):
        day = int(rng.integers(0, n_days))
        h, a = rng.choice(teams, 2, replace=False)
        lh = np.exp(0.1 + 0.2 + drift_a[h][day] + drift_d[a][day])
        la = np.exp(0.1 + drift_a[a][day] + drift_d[h][day])
        hg, ag = int(rng.poisson(lh)), int(rng.poisson(la))
        h1, a1 = int(rng.binomial(hg, 0.45)), int(rng.binomial(ag, 0.45))
        d = str(base0 + np.timedelta64(day, "D"))
        recs.append({"home_id": h, "away_id": a, "neutral": False,
                     "date": d,
                     "home_goals_ft": hg, "away_goals_ft": ag,
                     "home_goals_1h": h1, "away_goals_1h": a1,
                     "home_goals_2h": hg - h1, "away_goals_2h": ag - a1})
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])

    cutoff = df["date"].quantile(0.8)   # last 20% of time is the test set
    print(f"cutoff: {cutoff.date()}")

    r = run_backtest(df, cutoff, half_life=730)
    print(f"\ntrain={r['n_train']}  test={r['n_test_scored']}  "
          f"skipped(unseen)={r['n_skipped_unseen']}")
    print("\n            MODEL    BASELINE")
    for m in ("result", "over2.5", "btts"):
        flag = "  <-- beats baseline" if r["model"][m] < r["baseline"][m] else "  (worse!)"
        print(f"  {m:8s}  {r['model'][m]:.4f}   {r['baseline'][m]:.4f}{flag}")
    print(f"\n  weighted  {r['weighted_model']:.4f}   {r['weighted_baseline']:.4f}")

    print("\n--- HALF-LIFE SWEEP (held-out weighted Brier) ---")
    table, best = sweep_half_life(df, cutoff)
    print(table.to_string(index=False))
    print(f"\nbest half_life_days = {best}")