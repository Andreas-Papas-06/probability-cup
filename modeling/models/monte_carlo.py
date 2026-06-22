import numpy as np

def simulate_match(models, home, away, neutral=True, n_sims=50_000, seed=None):
    """
    models : dict like {('goals','1h'): CountRatingModel, ('goals','2h'): ...,
                        ('corners','ft'): ..., ('fouls','ft'): ...}
    Returns a dict of market probabilities estimated from n_sims simulations.
    """
    rng = np.random.default_rng(seed)

    def draw(stat, seg):
        """Draw n_sims home/away samples for one stat+segment from its model."""
        m = models[(stat, seg)]
        lh, la = m.rates(home, away, neutral=neutral)
        # negative binomial if the model found overdispersion, else Poisson
        if m.alpha_ > 0:
            r = 1.0 / m.alpha_
            home_s = rng.negative_binomial(r, r / (r + lh), n_sims)
            away_s = rng.negative_binomial(r, r / (r + la), n_sims)
        else:
            home_s = rng.poisson(lh, n_sims)
            away_s = rng.poisson(la, n_sims)
        return home_s, away_s