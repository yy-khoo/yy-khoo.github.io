"""Bump-and-reprice experiments for a terminal Black--Scholes call payoff."""

import numpy as np


def growth_factors(rng, count, maturity, rate, volatility):
    """Exact terminal multipliers S_T / S_0; no time discretisation."""
    z = rng.standard_normal(count)
    return np.exp((rate - 0.5 * volatility**2) * maturity
                  + volatility * np.sqrt(maturity) * z)


def common_greeks(growth, spot, strike, maturity, rate, bump):
    """Means of paired path contributions using common random numbers."""
    if bump <= 0 or bump >= spot:
        raise ValueError("bump must lie strictly between zero and spot")
    discount = np.exp(-rate * maturity)
    distance = spot * growth - strike
    shift = bump * growth
    delta = discount * (np.maximum(distance + shift, 0)
                        - np.maximum(distance - shift, 0)) / (2 * bump)
    # Exact payoff identity: (x+a)+ - 2x+ + (x-a)+ = (a-|x|)+.
    # Avoid round-off residuals for paths with exactly zero curvature.
    gamma = discount * np.maximum(shift - np.abs(distance), 0) / bump**2
    return np.array([delta.mean(), gamma.mean()])


def independent_greeks(base_growth, up_growth, down_growth,
                       spot, strike, maturity, rate, bump):
    """Three independent path groups for the three price estimates."""
    if bump <= 0 or bump >= spot:
        raise ValueError("bump must lie strictly between zero and spot")
    discount = np.exp(-rate * maturity)
    base = discount * np.maximum(spot * base_growth - strike, 0).mean()
    up = discount * np.maximum((spot + bump) * up_growth - strike, 0).mean()
    down = discount * np.maximum((spot - bump) * down_growth - strike, 0).mean()
    return np.array([(up - down) / (2 * bump),
                     (up - 2 * base + down) / bump**2])


def bump_experiment(bumps, path_counts=(2048, 8192, 32768), repeats=64,
                    seed=20261126, spot=100.0, strike=100.0,
                    maturity=1.0, rate=0.03, volatility=0.20):
    """Return estimates with axes: method, N, repeat, bump, Greek.

    Method 0 has independent reprices; method 1 shares terminal shocks.
    Replications are independent; within one replication, sweeps and path
    counts share draws (nested prefixes). Each reprice uses N paths.
    """
    bumps = np.asarray(bumps, dtype=float)
    if np.any(bumps <= 0) or np.any(bumps >= spot):
        raise ValueError("all bumps must lie strictly between zero and spot")
    estimates = np.empty((2, len(path_counts), repeats, len(bumps), 2))
    for rep in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
        groups = [growth_factors(rng, max(path_counts), maturity, rate, volatility)
                  for _ in range(3)]
        for ni, count in enumerate(path_counts):
            base, up, down = [group[:count] for group in groups]
            for hi, bump in enumerate(bumps):
                estimates[0, ni, rep, hi] = independent_greeks(
                    base, up, down, spot, strike, maturity, rate, bump)
                estimates[1, ni, rep, hi] = common_greeks(
                    base, spot, strike, maturity, rate, bump)
    return estimates
