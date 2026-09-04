"""The Markowitz frontier itself: estimation, the solve, and the JSON contract.

WHY PyPortfolioOpt AND NOT A HAND-ROLLED SOLVER
-----------------------------------------------
The unconstrained frontier has a closed form -- two matrix inverses and a scalar -- and
writing it takes twenty lines. It is also the wrong object here. The moment weights are
required to be non-negative (and they are: this is a long-only portfolio, and a frontier
that shorts a 130-asset universe is a frontier of leveraged estimation error) the problem
stops having a closed form and becomes a quadratic program with 130 inequality
constraints. PyPortfolioOpt hands that to cvxpy, which hands it to a proper conic solver;
the alternative is reimplementing active-set or critical-line, which is a real
numerical-methods project whose bugs look like plausible portfolios. So: PyPortfolioOpt
1.6 for the QP, and this module owns the estimation choices and the output contract,
which are the parts that actually decide what the picture says.

LEDOIT-WOLF SHRINKAGE, AND WHAT IT MEASURED HERE
------------------------------------------------
A mean-variance optimiser searches for low-variance directions, so it finds the sample
covariance's smallest eigenvalues -- which are the ones most contaminated by estimation
noise -- and levers them. That is the "error maximisation" property. Ledoit-Wolf shrinks
the sample matrix toward a structured target (constant variance) at an analytically
optimal intensity, raising those eigenvalues.

The measured intensity on this configuration is delta ~ 0.006, i.e. almost none, because
daily bars over 15 years give T = 3,938 observations for N = 116 assets and the estimator
correctly concludes the sample matrix needs little help. So shrinkage here is cheap
insurance that turns out to be nearly inactive -- it is NOT what is holding the answer
together, and `shrinkage_delta` is written into the manifest precisely so that claim is
checkable rather than assumed. (It would bind hard on monthly data, where T ~ 190 < N.)

What shrinkage does not fix at any intensity is a near-exact LINEAR DEPENDENCY, and this
universe contains several by construction: UUP is a short basket of EUR/JPY/GBP, which
FXE, FXY and FXB are the long legs of; IAU and GLD hold the same bullion; VTI, SPY and VV
are the same beta. The minimum-variance portfolio duly turns up holding ~27% UUP against
~16% FXE and four other currency legs, netting to 0.9% annualised volatility. That is not
a low-risk portfolio anyone found; it is the optimiser exploiting a collinearity, and it
is left in view rather than pruned because seeing it is the point.

WHAT NONE OF THIS FIXES, AND WHY IT IS SAID OUT LOUD
----------------------------------------------------
Expected returns are the sample geometric mean. This is a BAD estimator and no amount of
covariance care rescues it: the standard error on a 15-year annualised mean return for a
20%-vol asset is 20%/sqrt(15) ~ 5.2 percentage points, which is the same order as the
equity risk premium being estimated. The tangency portfolio is therefore a statement
about which assets happened to do well in this sample, and over a window starting in
2011 that answer is known in advance -- US large-cap technology. This is not a bug in the
code; it is the property of mean-variance optimisation that the whole exercise is meant
to show, and the `weight_cap` variants are the honest way to show it: at cap=1.0 the
optimiser concentrates, at cap=0.10 it cannot, and comparing the two frontiers is a
direct read on how much of the "optimal" portfolio was estimation error. Any conclusion
drawn from the cap=1.0 tangency weights alone is a conclusion about the sample.

THE JSON CONTRACT: EVERY POINT CARRIES ITS OWN WEIGHTS
------------------------------------------------------
The SPA lets a reader drag a point along the frontier, which means the frontier has to be
continuous in the browser while the solve happens here. The obvious implementation --
fit a curve through a few points and interpolate the CURVE -- was rejected: it produces a
(vol, ret) pair with no portfolio attached, so the dragged point cannot answer "what do I
hold", which is the only question worth asking. Instead every solved point ships its full
weight vector, and the browser interpolates the WEIGHTS between adjacent points. A convex
combination of two long-only feasible portfolios is itself long-only feasible, so the
interpolated point is a portfolio someone could actually hold; its return is exact
(return is linear in w) and its variance is computed exactly in the browser from the
covariance matrix in `stats.json`, so the displayed risk is the true risk of the displayed
holdings and sits very slightly above the true frontier -- by construction, since a chord
of a convex curve lies above it. The gap is the honest cost of a finite point count, and
`--points 60` keeps it below a basis point of volatility across this universe.

Weights are cleaned (rounded to 1e-4, tiny positions dropped) and then RENORMALISED, and
every reported ret/vol/sharpe is recomputed from the cleaned vector. The output is
therefore self-consistent: a reader who recomputes w'mu and sqrt(w'Sw) from the shipped
numbers gets the shipped numbers back, which `tests/test_frontier_invariants.py` asserts
point by point. Reporting the solver's performance for an un-cleaned vector alongside the
cleaned weights would be a small lie that nothing downstream could detect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

from fetch import TRADING_DAYS_PER_YEAR

# Below this, a weight is noise from the solver rather than a position.
WEIGHT_FLOOR = 1e-4

# Key under which a frontier point carries its own unrounded (ret, vol, sharpe) while `build`
# is still working with it. Deleted before the point is written -- nothing downstream may see it.
EXACT = "_exact"


@dataclass(frozen=True)
class Estimates:
    mu: pd.Series  # annualised geometric mean return
    cov: pd.DataFrame  # annualised Ledoit-Wolf shrunk covariance
    shrinkage: float
    n_obs: int


def estimate(panel: pd.DataFrame) -> Estimates:
    mu = expected_returns.mean_historical_return(
        panel, compounding=True, frequency=TRADING_DAYS_PER_YEAR
    )
    shrink = risk_models.CovarianceShrinkage(panel, frequency=TRADING_DAYS_PER_YEAR)
    cov = shrink.ledoit_wolf()
    cov = risk_models.fix_nonpositive_semidefinite(cov, fix_method="spectral")
    return Estimates(mu=mu, cov=cov, shrinkage=float(shrink.delta), n_obs=len(panel) - 1)


def performance(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret = float(w @ mu)
    var = float(w @ cov @ w)
    vol = math.sqrt(max(var, 0.0))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return ret, vol, sharpe


def max_feasible_return(mu: pd.Series, cap: float) -> float:
    """The right-hand end of the frontier, solved exactly rather than searched for.

    Maximising a linear objective over the simplex intersected with a box has a greedy
    solution: pour `cap` into the highest-mu asset, then the next, until the weights sum
    to one. No solver needed, and getting it exactly right matters -- an approximate
    upper end makes the last few `efficient_return` calls infeasible, and an infeasible
    solve in PyPortfolioOpt surfaces as a generic OptimizationError that looks like a
    numerical problem rather than a bad target.
    """
    if not 0 < cap <= 1:
        raise ValueError(f"cap must be in (0, 1]: {cap}")
    order = np.sort(mu.to_numpy())[::-1]
    remaining, total = 1.0, 0.0
    for m in order:
        take = min(cap, remaining)
        total += take * m
        remaining -= take
        if remaining <= 1e-12:
            break
    if remaining > 1e-9:
        raise ValueError(f"cap {cap} too small for {len(mu)} assets to sum to 1")
    return float(total)


def _clean(weights: dict[str, float], symbols: list[str], cap: float) -> np.ndarray:
    """Solver output -> the exact vector that ships, summing to one AT THE SHIPPED PRECISION.

    The order here is the whole content of the function. Dropping sub-floor positions and
    then renormalising gets a float64 vector that sums to 1.0; rounding THAT to 6 decimals
    for transport breaks the sum again, by up to n/2 * 1e-6 -- about 1e-5 on a 20-holding
    portfolio. Small, invisible in a chart, and it means the JSON does not satisfy the one
    property everything downstream assumes about it. So the rounding happens here, before
    anything is measured, and the residual is pushed into a single position that has room
    for it. `performance()` is then computed on this exact vector, so the numbers in the
    file and the weights in the file describe the same portfolio.

    The residual goes to the LARGEST position with headroom under `cap`, not the largest
    outright: at cap=0.2 the tangency portfolio pins several holdings exactly at 0.2, and
    adding 6e-5 to one of those would ship a weight that breaches the cap the file reports.
    Largest-with-headroom keeps the relative size of the adjustment negligible (the
    alternative, spreading it over the smallest position, changes a 1e-4 holding by 60%).

    THE CAP ALSO HAS TO BE ENFORCED HERE, because the conic solver does not enforce it
    exactly. cvxpy's box constraint is satisfied to the solver's own tolerance, which on
    this problem is around 1e-5: at cap=0.2 the raw solution came back holding 0.200014,
    and `clean_weights`' 5-decimal rounding preserved it. Shipping that means the file
    states `weight_cap: 0.2` and contains a weight above 0.2 -- a claim the SPA prints
    verbatim and cannot check. So the vector is clipped to the cap and the freed mass is
    water-filled back into positions that have room, which is a projection onto the exact
    feasible set. The mass being moved is ~1e-5 of the portfolio, so it changes no number a
    reader sees; the point is that afterwards the constraint is exactly true rather than
    nearly true. Water-filling never OPENS a position: absorbing rounding error by creating
    a holding that the optimiser set to zero would be inventing a position out of noise.
    """
    w = np.array([weights.get(s, 0.0) for s in symbols], dtype=float)
    w[w < WEIGHT_FLOOR] = 0.0  # long-only, so a negative is solver noise by definition
    total = w.sum()
    if total <= 0:
        raise ValueError("cleaned weights sum to zero")
    w = np.minimum(w / total, cap)

    for _ in range(64):
        deficit = 1.0 - float(w.sum())
        if deficit <= 1e-15:
            break
        room = np.where(w > 0, cap - w, 0.0)
        total_room = float(room.sum())
        if total_room <= 0:
            raise ValueError(f"no headroom under cap={cap} to absorb deficit {deficit}")
        w = w + room * min(1.0, deficit / total_room)
    w = np.round(np.minimum(w, cap), 6)

    residual = round(1.0 - float(w.sum()), 9)
    if residual != 0.0:
        for j in np.argsort(-w):
            if w[j] <= 0:
                break
            if 0 < w[j] + residual <= cap + 1e-12:
                w[j] = round(w[j] + residual, 6)
                break
        else:
            raise ValueError(f"nowhere to absorb rounding residual {residual}")
        if abs(float(w.sum()) - 1.0) > 1e-9:
            raise ValueError(f"repair failed: weights sum to {w.sum()!r}")
    return w


def _solve(mu: pd.Series, cov: pd.DataFrame, cap: float, objective, *args, **kwargs) -> np.ndarray | None:
    ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, cap))
    try:
        getattr(ef, objective)(*args, **kwargs)
    except Exception:
        return None
    return _clean(ef.clean_weights(cutoff=WEIGHT_FLOOR), list(mu.index), cap)


def _point(w: np.ndarray, symbols: list[str], mu: np.ndarray, cov: np.ndarray, rf: float) -> dict:
    """One frontier point, rounded for transport, carrying its own unrounded values under
    `EXACT` for `build` to locate the extrema with and then delete. See `build`."""
    ret, vol, sharpe = performance(w, mu, cov, rf)
    return {
        "ret": round(ret, 8),
        "vol": round(vol, 8),
        "sharpe": round(sharpe, 6),
        "w": {s: round(float(x), 6) for s, x in zip(symbols, w) if x > 0},
        EXACT: (ret, vol, sharpe),
    }


def _prune_dominated(points: list[dict], tol: float = 1e-9) -> tuple[list[dict], int]:
    """Drop points that are not on the frontier, and count them.

    A solved point is dominated when another point has at least its return at no more
    risk. On an exact frontier none are; with a conic solver at 1e-8 tolerances a handful
    can cross by ~1e-7, and an out-of-order point breaks the browser's assumption that
    the point list is a monotone path it can interpolate along -- the dragged handle would
    jump backwards. Pruning here rather than papering over it in the SPA keeps the
    invariant where it can be tested, and a nonzero count in the output is the signal to
    look at solver settings rather than something to ignore.
    """
    kept: list[dict] = []
    for p in sorted(points, key=lambda q: (q["ret"], q["vol"])):
        # p has the highest return seen so far, so any kept point at p's risk or above is
        # dominated by it. A stack, not a single lookback: one badly-conditioned solve can
        # dominate several of its neighbours at once.
        while kept and kept[-1]["vol"] >= p["vol"] - tol:
            kept.pop()
        if kept and kept[-1]["ret"] >= p["ret"] - tol:
            continue  # p pays more risk for no more return
        kept.append(p)
    return kept, len(points) - len(kept)


def _extremum_indices(frontier: list[dict]) -> tuple[int, int]:
    """(min_vol_index, max_sharpe_index), located at FULL precision. Not a nicety.

    The obvious implementation takes the argmax over the `sharpe` FIELD -- but that field is
    rounded to 6 decimals for transport, and the frontier is STATIONARY at the tangency, so its
    neighbours sit within ~1e-9 of it and round to the identical 6 decimals. `max` over a tie
    returns whichever came first, which is a fact about the point ORDER rather than about the
    portfolios. Measured on the ETF universe at cap 1.0: point 40 beats point 39 by 1.1e-9 in
    Sharpe, both ship as 0.819164, and the rounded argmax therefore labelled 39 the tangency.
    The browser recomputes Sharpe from the weights and the covariance, so it saw 40 and
    disagreed with the file -- caught by `agrees with the pipeline about which solved point is
    the tangency` in `web/src/portfolio.test.ts`, and blessed until then by a pipeline test that
    resolved the tie the same wrong way the pipeline did.

    Same argument for `vol` at 8 decimals and the minimum-variance point. The nearest grid point
    happens to be further away there today, which is luck, not a reason.
    """
    return (
        min(range(len(frontier)), key=lambda i: frontier[i][EXACT][1]),
        max(range(len(frontier)), key=lambda i: frontier[i][EXACT][2]),
    )


def build(
    est: Estimates,
    rf: float,
    cap: float,
    n_points: int = 60,
) -> dict:
    symbols = list(est.mu.index)
    mu_v = est.mu.to_numpy()
    cov_v = est.cov.to_numpy()

    w_minvol = _solve(est.mu, est.cov, cap, "min_volatility")
    if w_minvol is None:
        raise RuntimeError(f"min_volatility infeasible at cap={cap}")
    r_lo = float(w_minvol @ mu_v)
    r_hi = max_feasible_return(est.mu, cap)

    points = [_point(w_minvol, symbols, mu_v, cov_v, rf)]
    failed_targets: list[float] = []
    # Skip the exact endpoints: r_lo is already solved, and r_hi sits on the feasibility
    # boundary where the QP is degenerate (it is an LP vertex, not a variance minimum).
    for target in np.linspace(r_lo, r_hi, n_points + 2)[1:-1]:
        w = _solve(est.mu, est.cov, cap, "efficient_return", target_return=float(target))
        if w is None:
            failed_targets.append(round(float(target), 6))
            continue
        points.append(_point(w, symbols, mu_v, cov_v, rf))

    w_tan = _solve(est.mu, est.cov, cap, "max_sharpe", risk_free_rate=rf)
    if w_tan is None:
        raise RuntimeError(f"max_sharpe infeasible at cap={cap}")
    points.append(_point(w_tan, symbols, mu_v, cov_v, rf))

    frontier, pruned = _prune_dominated(points)
    min_vol_index, max_sharpe_index = _extremum_indices(frontier)
    for p in frontier:
        del p[EXACT]

    return {
        "weight_cap": cap,
        "risk_free_rate": rf,
        "n_requested": n_points,
        "n_points": len(frontier),
        "pruned": pruned,
        "failed_targets": failed_targets,
        "return_range": [round(r_lo, 8), round(r_hi, 8)],
        "frontier": frontier,
        "min_vol_index": min_vol_index,
        "max_sharpe_index": max_sharpe_index,
    }


def asset_table(est: Estimates, panel: pd.DataFrame, rf: float, meta: dict[str, tuple[str, str, str]]) -> list[dict]:
    """Per-asset (vol, ret) for the scatter behind the frontier.

    `sigma` here is the diagonal of the SHRUNK covariance, not the raw sample standard
    deviation, and that is deliberate: the dot a reader sees must sit in the same
    coordinate system as the curve, or a single asset appears to plot below its own
    frontier. Ledoit-Wolf shrinks variances toward the average variance, so raw and
    shrunk differ by a percent or two per asset -- small, and exactly the size that makes
    a chart look subtly broken.
    """
    sigma = np.sqrt(np.diag(est.cov.to_numpy()))
    rows = []
    for i, sym in enumerate(est.mu.index):
        name, asset_class, group = meta[sym]
        s = panel[sym]
        rows.append(
            {
                "symbol": sym,
                "name": name,
                "asset_class": asset_class,
                "group": group,
                "ret": round(float(est.mu.iloc[i]), 8),
                "vol": round(float(sigma[i]), 8),
                "sharpe": round(float((est.mu.iloc[i] - rf) / sigma[i]), 6),
                "first_date": str(s.index.min().date()),
                "last_date": str(s.index.max().date()),
            }
        )
    return rows
