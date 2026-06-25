"""
referee.py
==========
Referee adjustment layer for the cards, fouls, and penalty predictions.
 
WHY: the referee is the dominant driver of cards, fouls and penalties -- a strict
ref inflates BOTH teams' counts regardless of who's playing. The team attack/
defense model captures team tendencies; the referee is a MATCH-LEVEL multiplier
on top. And penalties have no team model at all -- the referee history IS the model.
 
HOW:
  build_referee_table(df)  -> per-ref rates (cards/game, fouls/game, penalties/game)
                              with SHRINKAGE toward the league mean (few-game refs
                              don't get extreme rates).
  ref_factor(table, ref_id, stat) -> multiplier vs league average (1.0 = average ref).
  penalty_rate(table, ref_id)     -> expected penalties for the match (a rate).
 
USAGE with the suite:
  - cards/fouls: multiply the model's predicted (lh, la) by ref_factor(...).
  - penalty:     lam_pen = penalty_rate(...); P(>=1) = 1 - exp(-lam_pen).
 
DATA NOTE: penalties are NOT in the stats feed. If you have no penalty column,
penalty_rate falls back to a league base rate (still better than a flat constant
once you can scale it by the ref's CARD strictness as a weak proxy).
"""
 
from __future__ import annotations
import numpy as np
import pandas as pd
 
 
def build_referee_table(df, shrink_k=10.0):
    """
    Per-referee per-match rates, shrunk toward the league mean.
 
    shrink_k : pseudo-count. A ref with shrink_k games sits halfway between their
               own rate and the league mean; more games -> trust their own rate.
               This stops a ref with 2 games getting an extreme multiplier.
 
    Returns a dict: ref_id -> {cards, fouls, penalties: shrunk per-match rate}
    plus a 'league' entry with the means (used as the average reference).
    """
    # total cards / fouls / penalties PER MATCH (both teams combined)
    def match_total(stat):
        h, a = f"home_{stat}_ft", f"away_{stat}_ft"
        if h in df.columns and a in df.columns:
            return df[h] + df[a]
        return pd.Series(np.nan, index=df.index)
 
    work = pd.DataFrame({
        "ref_id": df.get("ref_id"),
        "cards": match_total("cards"),
        "fouls": match_total("fouls"),
        "penalties": match_total("penalties"),   # likely all-NaN if not pulled
    })
 
    league = {s: work[s].mean(skipna=True) for s in ("cards", "fouls", "penalties")}
 
    table = {"league": league}
    for ref, grp in work.dropna(subset=["ref_id"]).groupby("ref_id"):
        entry = {}
        for s in ("cards", "fouls", "penalties"):
            vals = grp[s].dropna()
            n = len(vals)
            lm = league[s]
            if n == 0 or pd.isna(lm):
                entry[s] = lm                       # no data -> league mean
            else:
                own = vals.mean()
                # shrink: weighted avg of own rate and league mean
                entry[s] = (n * own + shrink_k * lm) / (n + shrink_k)
        entry["n_games"] = len(grp)
        table[ref] = entry
    return table
 
 
def ref_factor(table, ref_id, stat):
    """
    Multiplier for `stat` (cards/fouls) vs an average referee.
    1.0 = league-average ref; 1.3 = this ref produces 30% more than average.
    Unknown ref -> 1.0 (no adjustment).
    """
    league = table["league"].get(stat)
    if league is None or pd.isna(league) or league == 0:
        return 1.0
    entry = table.get(ref_id)
    if entry is None or pd.isna(entry.get(stat)):
        return 1.0
    return float(entry[stat] / league)
 
 
def penalty_rate(table, ref_id, base_rate=0.25, use_card_proxy=True):
    """
    Expected penalties in the match (a rate -> P(>=1) = 1 - exp(-rate)).
 
    If real penalty data exists in the table, use the ref's shrunk penalty rate.
    Otherwise fall back to `base_rate`, optionally scaled by how card-strict the
    ref is (a weak proxy: strict refs award more penalties on average).
    """
    league = table["league"]
    entry = table.get(ref_id, {})
 
    pen = entry.get("penalties")
    if pen is not None and not pd.isna(pen):
        return float(pen)                          # real penalty data available
 
    # no penalty data -> base rate, optionally nudged by the ref's card strictness
    if use_card_proxy:
        return base_rate * ref_factor(table, ref_id, "cards")
    return base_rate
 
 
# ---------------------------------------------------------------------------
# Wrappers that apply the referee adjustment to suite predictions.
# ---------------------------------------------------------------------------
def adjust_cards_fouls(pred, table, ref_id):
    """
    Scale the cards and fouls rates (and their over-line probabilities) in a
    predict_fixture() output by the referee factor. Modifies a COPY and returns it.
    Re-reads the over-lines from the adjusted rate so probabilities stay consistent.
    """
    from modeling.models.forcast_v1 import total_over, prob_home_more
    import copy
    out = copy.deepcopy(pred)
 
    for stat in ("cards", "fouls"):
        if stat not in out:
            continue
        f = ref_factor(table, ref_id, stat)
        if f == 1.0:
            continue
        for seg, entry in out[stat].items():
            lh = entry["home_exp"] * f
            la = entry["away_exp"] * f
            entry["home_exp"] = round(lh, 3)
            entry["away_exp"] = round(la, 3)
            entry["total_exp"] = round(lh + la, 3)
            entry["ref_factor"] = round(f, 3)
            # NOTE: dispersion alpha is held from the unadjusted fit; the rate
            # shift is the dominant effect. Re-read over-lines at the new rate.
            for line_str in list(entry.get("over", {})):
                L = float(line_str)
                entry["over"][line_str] = round(total_over(L, lh, la, 0.0), 3)
    return out
 
 
def penalty_prediction(table, ref_id, base_rate=0.25):
    """Per-segment P(>=1 penalty), referee-aware."""
    lam_ft = penalty_rate(table, ref_id, base_rate=base_rate)
    out = {}
    for seg, frac in (("1h", 0.45), ("2h", 0.55), ("ft", 1.0)):
        out[seg] = round(1 - np.exp(-lam_ft * frac), 3)
    out["ref_factor"] = round(ref_factor(table, ref_id, "cards"), 3)
    return out