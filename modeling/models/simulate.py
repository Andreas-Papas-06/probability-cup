"""
simulate.py
===========
Monte Carlo match simulator. Reuses the ALREADY-FITTED suite (no new estimation)
to draw N synthetic playings of a match, then reads any market as the fraction of
sims where it happened.
 
WHEN TO USE THIS vs the analytical readouts in predict_suite:
  - single-stat markets (over 2.5, BTTS) -> use the analytical functions; they're
    exact and noise-free. The simulator MATCHES them (a built-in sanity check).
  - JOINT / CONDITIONAL markets -> use the simulator. "over 2.5 AND BTTS",
    "home win AND over 1.5", "red card AND over 4.5 cards" are trivial here
    (count sims where all legs hold) and painful analytically.
  - GAME-STATE dependence -> only the simulator can do it: draw the 1H, look at
    the score, adjust 2H scoring rates, then draw the 2H.
 
KEY PROPERTY: every simulated match has FT = 1H + 2H by construction (we draw the
halves and sum), so half/full markets are always internally consistent.
 
Integration:
  - rates come from predict_suite._seg_rates (direct half model, or thinned FT).
  - cards/fouls rates are scaled by the referee factor if a ref table is given.
  - reds drawn from per-team red rates -> enables red-card combo markets.
"""
 
from __future__ import annotations
import numpy as np
from modeling.models.predict import COUNT_STATS, _seg_rates

 
try:
    from modeling.models.referee import ref_factor
except Exception:
    def ref_factor(*a, **k):   # graceful fallback if referee.py absent
        return 1.0
 
 
# ---------------------------------------------------------------------------
# Draw n samples of (home, away) from a rate, Poisson or Negative Binomial.
# Rates may be scalars OR per-sim arrays (used for game-state conditioning).
# ---------------------------------------------------------------------------
def _draw(rng, lh, la, alpha, n):
    if alpha and alpha > 0:
        r = 1.0 / alpha
        ph = r / (r + np.asarray(lh))
        pa = r / (r + np.asarray(la))
        return rng.negative_binomial(r, ph, n), rng.negative_binomial(r, pa, n)
    return rng.poisson(lh, n), rng.poisson(la, n)
 
 
def simulate_match(models, shares, home, away, neutral=True, n_sims=50_000,
                   ref_table=None, ref_id=None,
                   red_rates=None, red_base=0.10,
                   pen_rate=None, game_state=True,
                   gs_push=1.15, gs_sit=0.90, seed=None):
    """
    Returns (sims, markets):
      sims[stat][seg] = (home_array, away_array)   raw simulated counts
      sims['red'] / sims['pen']                    boolean arrays (event occurred)
      markets = dict of probabilities incl. JOINT/CONDITIONAL ones
    """
    rng = np.random.default_rng(seed)
    sims = {}
 
    for stat in COUNT_STATS:
        lh1, la1 = _seg_rates(models, shares, stat, "1h", home, away, neutral)
        lh2, la2 = _seg_rates(models, shares, stat, "2h", home, away, neutral)
        if lh1 is None or lh2 is None:
            continue
 
        # referee multiplier for discipline stats
        f = 1.0
        if ref_table is not None and stat in ("cards", "fouls"):
            f = ref_factor(ref_table, ref_id, stat)
        lh1, la1, lh2, la2 = lh1 * f, la1 * f, lh2 * f, la2 * f
 
        alpha = models.get((stat, "1h"), models.get((stat, "ft"))).alpha_
 
        h1, a1 = _draw(rng, lh1, la1, alpha, n_sims)
 
        if game_state and stat == "goals":
            # 2H scoring depends on the 1H state: trailing side pushes, leader sits
            lead = h1 - a1
            home_boost = np.where(lead < 0, gs_push, np.where(lead > 0, gs_sit, 1.0))
            away_boost = np.where(lead > 0, gs_push, np.where(lead < 0, gs_sit, 1.0))
            h2, a2 = _draw(rng, lh2 * home_boost, la2 * away_boost, alpha, n_sims)
        else:
            h2, a2 = _draw(rng, lh2, la2, alpha, n_sims)
 
        sims[stat] = {"1h": (h1, a1), "2h": (h2, a2), "ft": (h1 + h2, a1 + a2)}
 
    # rare binaries drawn into the same sim so they correlate in combos
    if red_rates is not None:
        hr = red_rates.get(home, red_base) if isinstance(red_rates, dict) else red_base
        ar = red_rates.get(away, red_base) if isinstance(red_rates, dict) else red_base
        reds = rng.poisson(hr + ar, n_sims)
        sims["red"] = reds > 0
    if pen_rate is not None:
        sims["pen"] = rng.poisson(pen_rate, n_sims) > 0
 
    markets = _markets(sims)
    return sims, markets
 
 
# ---------------------------------------------------------------------------
# Read markets off the simulation arrays. SINGLE markets match the analytical
# ones; the COMBO / CONDITIONAL block is what the simulator is actually for.
# ---------------------------------------------------------------------------
def _markets(sims):
    m = {}
    hg, ag = sims["goals"]["ft"]
    hg1, ag1 = sims["goals"]["1h"]
    tot = hg + ag
 
    # --- single goal markets (sanity-checkable against analytical) ---
    m["home_win"] = float(np.mean(hg > ag))
    m["draw"] = float(np.mean(hg == ag))
    m["away_win"] = float(np.mean(hg < ag))
    m["over_2.5"] = float(np.mean(tot > 2.5))
    m["btts"] = float(np.mean((hg >= 1) & (ag >= 1)))
    m["1h_over_0.5"] = float(np.mean((hg1 + ag1) > 0.5))
 
    # --- JOINT / CONDITIONAL markets: the reason simulation exists ---
    m["over_2.5_AND_btts"] = float(np.mean((tot > 2.5) & (hg >= 1) & (ag >= 1)))
    m["home_win_AND_over_1.5"] = float(np.mean((hg > ag) & (tot > 1.5)))
    m["btts_AND_over_3.5"] = float(np.mean((hg >= 1) & (ag >= 1) & (tot > 3.5)))
    # goals in BOTH halves (conditional structure across segments)
    m["goal_in_both_halves"] = float(np.mean(((hg1 + ag1) > 0) &
                                             ((sims["goals"]["2h"][0] + sims["goals"]["2h"][1]) > 0)))
 
    if "corners" in sims:
        hc, ac = sims["corners"]["ft"]
        m["over_9.5_corners"] = float(np.mean((hc + ac) > 9.5))
        m["over_2.5_goals_AND_over_9.5_corners"] = float(
            np.mean((tot > 2.5) & ((hc + ac) > 9.5)))
 
    if "cards" in sims:
        hcd, acd = sims["cards"]["ft"]
        m["over_3.5_cards"] = float(np.mean((hcd + acd) > 3.5))
 
    if "red" in sims:
        m["red_card"] = float(np.mean(sims["red"]))
        if "cards" in sims:
            hcd, acd = sims["cards"]["ft"]
            m["red_AND_over_4.5_cards"] = float(np.mean(sims["red"] & ((hcd + acd) > 4.5)))
 
    if "pen" in sims:
        m["penalty"] = float(np.mean(sims["pen"]))
 
    return {k: round(v, 4) for k, v in m.items()}