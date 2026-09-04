"""Unit invariants for the pieces `build.py` composes.

Split from `test_output_invariants.py` on purpose: that file checks the shipped artifact is
self-consistent, which it can be while every number in it is wrong. This file checks the
functions against independent definitions -- a brute-force LP for the frontier's right
endpoint, an analytic frontier for a case that has one, hand-built cases for the pruner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

import frontier as fr


@pytest.fixture(scope="module")
def synthetic():
    """A small panel with a known factor structure and one deliberate near-duplicate pair.

    The duplicate (A0 and A9 share 99% of their driver) is there because it is the shape
    that makes a covariance matrix ill-conditioned, and every test below should hold on an
    ill-conditioned matrix rather than only on a comfortable one.
    """
    rng = np.random.default_rng(20260903)
    n_obs, n = 2500, 10
    market = rng.normal(4e-4, 9e-3, n_obs)
    betas = np.linspace(0.3, 1.6, n)
    drifts = np.linspace(-2e-4, 6e-4, n)
    idio = rng.normal(0, 6e-3, (n_obs, n))
    rets = drifts + np.outer(market, betas) + idio
    rets[:, 9] = 0.99 * rets[:, 0] + 0.01 * rets[:, 9]
    prices = 100 * np.exp(np.cumsum(np.log1p(rets), axis=0))
    idx = pd.bdate_range("2012-01-02", periods=n_obs)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n)])


@pytest.fixture(scope="module")
def est(synthetic):
    return fr.estimate(synthetic)


# ------------------------------------------------------------------------------- estimation


def test_covariance_is_symmetric_psd_and_annualised(est, synthetic):
    cov = est.cov.to_numpy()
    assert np.allclose(cov, cov.T, atol=1e-14)
    assert np.linalg.eigvalsh(cov).min() > 0
    # Ledoit-Wolf at T=2500, N=10 shrinks almost not at all, so the annualisation factor is
    # checkable against the raw sample covariance: a 252 vs 365 mix-up (a real hazard in a
    # repo that also handles crypto) shows up as a 45% error here.
    raw = synthetic.pct_change().dropna().cov().to_numpy() * 252.0
    assert np.allclose(cov, raw, rtol=0.05), "annualisation factor or shrinkage target is off"


def test_expected_return_is_the_geometric_mean_not_the_arithmetic_one(est, synthetic):
    """Arithmetic and geometric means differ by ~sigma^2/2, which is 2 percentage points on
    a 20%-vol asset -- enough to reorder the universe and move the tangency portfolio."""
    n_returns = len(synthetic) - 1
    for sym in synthetic.columns:
        p = synthetic[sym]
        expected = (p.iloc[-1] / p.iloc[0]) ** (252.0 / n_returns) - 1.0
        assert est.mu[sym] == pytest.approx(expected, rel=1e-9)


def test_shrinkage_intensity_is_reported_and_in_range(est):
    assert 0.0 <= est.shrinkage <= 1.0


# ------------------------------------------------------------------- max_feasible_return


# 0.1 is the tightest feasible cap on a 10-asset universe (10 x 0.1 = 1 exactly), and it is
# included because that boundary -- every asset pinned at the cap -- is where the greedy
# fill's termination condition has to be right.
@pytest.mark.parametrize("cap", [1.0, 0.5, 0.34, 0.25, 0.1])
def test_max_feasible_return_matches_a_brute_force_linear_program(est, cap):
    """The greedy fill is claimed to be exact; an LP solver is the independent check.

    This is the endpoint every `efficient_return` target is measured against, so an error
    here does not produce a wrong number -- it produces an infeasible solve that surfaces as
    a generic OptimizationError and looks like a numerical problem.
    """
    mu = est.mu.to_numpy()
    n = len(mu)
    lp = linprog(
        c=-mu,
        A_eq=np.ones((1, n)),
        b_eq=[1.0],
        bounds=[(0.0, cap)] * n,
        method="highs",
    )
    assert lp.success
    assert fr.max_feasible_return(est.mu, cap) == pytest.approx(-lp.fun, abs=1e-10)


def test_max_feasible_return_rejects_a_cap_that_cannot_reach_one(est):
    with pytest.raises(ValueError):
        fr.max_feasible_return(est.mu, 1.0 / (len(est.mu) + 1))
    with pytest.raises(ValueError):
        fr.max_feasible_return(est.mu, 0.0)


# ------------------------------------------------------------------------------- _clean


def test_clean_ships_a_vector_that_sums_to_one_and_respects_the_cap():
    symbols = [f"A{i}" for i in range(10)]
    # Deliberately the three things a conic solver actually returns: a weight 1.4e-5 OVER
    # the box constraint, a tiny negative from the dual, and sub-floor dust.
    raw = {
        "A0": 0.200014,
        "A1": 0.2,
        "A2": 0.2,
        "A3": 0.199_9,
        "A4": 0.100_02,
        "A5": 0.099_9,
        "A6": -3e-7,
        "A7": 2e-5,
        "A8": 0.0,
        "A9": 0.0,
    }
    w = fr._clean(raw, symbols, cap=0.2)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    # AND at the precision it ships at. Both halves are needed: an unrounded float vector
    # sums to 1.0 exactly and then stops doing so when `_point` rounds it for transport,
    # which is the bug `_clean` was reordered to prevent. Asserting only the sum passes on a
    # `_clean` that skips the rounding entirely -- measured, not guessed (tests/_mutate.py).
    assert np.array_equal(w, np.round(w, 6)), "not exact at the 6-decimal transport precision"
    assert w.max() <= 0.2 + 1e-12, f"cap breached: {w.max()}"
    assert (w >= 0).all()
    assert w[6] == 0.0 and w[7] == 0.0, "sub-floor dust was kept"
    assert w[8] == 0.0 and w[9] == 0.0, "a position was opened to absorb rounding"


def test_clean_never_opens_a_position_the_optimiser_set_to_zero():
    """Absorbing rounding error by creating a holding invents a position out of noise, and a
    reader looking at a 12-line weight table has no way to know which line is real."""
    symbols = [f"A{i}" for i in range(50)]
    raw = {"A0": 0.5, "A1": 0.3, "A2": 0.199_999_4}
    w = fr._clean(raw, symbols, cap=1.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.count_nonzero(w) == 3


def test_clean_rejects_an_all_zero_vector():
    with pytest.raises(ValueError):
        fr._clean({"A0": 1e-9}, ["A0", "A1"], cap=1.0)


# --------------------------------------------------------------------- _prune_dominated


def _pt(ret, vol):
    return {"ret": ret, "vol": vol, "sharpe": ret / vol, "w": {"A0": 1.0}}


# ------------------------------------------------------------------- _extremum_indices


def _exact_pt(vol, sharpe):
    """A frontier point as `build` sees it: transport-rounded fields plus the unrounded values."""
    return {"ret": round(0.1, 8), "vol": round(vol, 8), "sharpe": round(sharpe, 6),
            "w": {"A0": 1.0}, fr.EXACT: (0.1, vol, sharpe)}


def test_extremum_indices_break_a_transport_rounding_tie_by_the_unrounded_value():
    """The regression this exists for, constructed rather than hoped for.

    Two points whose Sharpe differs by 1.1e-9 -- the real margin measured between solved points
    39 and 40 at cap 1.0 -- ship the IDENTICAL 6-decimal `sharpe`. An implementation reading that
    field has no information left to choose with and returns whichever came first; the browser
    recomputes from the weights and the covariance, sees the difference, and disagrees with the
    file about which point is the tangency.

    Asserted in both orders on purpose: checking only one lets `min`/`max` over the rounded field
    pass by accident, since a first-tie rule is right half the time.
    """
    lo, hi = 0.819164364705, 0.819164365815
    assert round(lo, 6) == round(hi, 6), "the tie this test is about has stopped being a tie"

    _, k = fr._extremum_indices([_exact_pt(0.15, 0.7), _exact_pt(0.166, lo), _exact_pt(0.167, hi)])
    assert k == 2
    _, k = fr._extremum_indices([_exact_pt(0.15, 0.7), _exact_pt(0.166, hi), _exact_pt(0.167, lo)])
    assert k == 1

    # Same argument on the other axis: two volatilities that share 8 decimals.
    v_lo, v_hi = 0.11190000001, 0.11190000002
    assert round(v_lo, 8) == round(v_hi, 8)
    i, _ = fr._extremum_indices([_exact_pt(v_hi, 0.7), _exact_pt(v_lo, 0.7), _exact_pt(0.2, 0.6)])
    assert i == 1


def test_prune_keeps_a_monotone_path_untouched():
    pts = [_pt(0.02, 0.05), _pt(0.05, 0.08), _pt(0.09, 0.15)]
    kept, pruned = fr._prune_dominated(list(reversed(pts)))
    assert pruned == 0
    assert [p["ret"] for p in kept] == [0.02, 0.05, 0.09]


def test_prune_removes_a_point_that_pays_more_risk_for_less_return():
    pts = [_pt(0.02, 0.05), _pt(0.04, 0.09), _pt(0.05, 0.08)]
    kept, pruned = fr._prune_dominated(pts)
    assert pruned == 1
    assert [(p["ret"], p["vol"]) for p in kept] == [(0.02, 0.05), (0.05, 0.08)]


def test_prune_pops_a_run_of_dominated_points_not_just_the_last():
    """One point must be able to evict SEVERAL already-kept points, which is why the pruner
    uses a stack rather than a single-step lookback.

    The shape of this case is load-bearing and was got wrong first time. A descending
    staircase -- (.02,.05) (.03,.12) (.04,.11) (.05,.10) (.09,.09) -- looks like the test for
    this and is not: each arrival dominates exactly ONE kept point, so `if` and `while`
    produce identical output and the mutation `while -> if` survives it. (Confirmed by
    `tests/_mutate.py`, which is what this rewrite came out of.)

    Discriminating instead requires the kept list to hold two points that a single later
    point dominates at once. Kept points ascend in both axes, so (0.03, 0.10) and
    (0.04, 0.12) both survive until (0.09, 0.08) arrives and undercuts both. A single
    lookback pops only the last, leaving vols [0.05, 0.10, 0.08] -- non-monotone, and the
    browser would drag the handle backwards through it.
    """
    pts = [_pt(0.02, 0.05), _pt(0.03, 0.10), _pt(0.04, 0.12), _pt(0.09, 0.08)]
    kept, pruned = fr._prune_dominated(pts)
    assert pruned == 2, "only one of the two dominated points was evicted"
    assert [(p["ret"], p["vol"]) for p in kept] == [(0.02, 0.05), (0.09, 0.08)]
    assert [p["vol"] for p in kept] == sorted(p["vol"] for p in kept)


def test_prune_walks_down_a_descending_staircase():
    """The one-eviction-per-arrival case, kept as its own test now that it is known not to
    exercise the stack. It still checks the pruner drains a long dominated tail."""
    pts = [_pt(0.02, 0.05), _pt(0.03, 0.12), _pt(0.04, 0.11), _pt(0.05, 0.10), _pt(0.09, 0.09)]
    kept, pruned = fr._prune_dominated(pts)
    assert pruned == 3
    assert [(p["ret"], p["vol"]) for p in kept] == [(0.02, 0.05), (0.09, 0.09)]


def test_prune_output_is_always_strictly_increasing_in_both_axes():
    rng = np.random.default_rng(7)
    for _ in range(200):
        pts = [_pt(float(r), float(v)) for r, v in rng.uniform(0.01, 0.3, (12, 2))]
        kept, pruned = fr._prune_dominated(pts)
        rets = [p["ret"] for p in kept]
        vols = [p["vol"] for p in kept]
        assert rets == sorted(rets) and len(set(rets)) == len(rets)
        assert vols == sorted(vols) and len(set(vols)) == len(vols)
        assert pruned == len(pts) - len(kept)


# ----------------------------------------------------------------------------- performance


def test_performance_matches_the_closed_form_on_a_two_asset_case():
    mu = np.array([0.05, 0.12])
    sd = np.array([0.10, 0.25])
    rho = -0.3
    cov = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]])
    w = np.array([0.4, 0.6])
    ret, vol, sharpe = fr.performance(w, mu, cov, rf=0.02)
    assert ret == pytest.approx(0.4 * 0.05 + 0.6 * 0.12)
    hand = np.sqrt(
        (0.4 * 0.10) ** 2 + (0.6 * 0.25) ** 2 + 2 * 0.4 * 0.6 * rho * 0.10 * 0.25
    )
    assert vol == pytest.approx(hand)
    assert sharpe == pytest.approx((ret - 0.02) / hand)
    # Diversification with negative correlation must beat the weighted average of the vols.
    assert vol < 0.4 * 0.10 + 0.6 * 0.25


# ----------------------------------------------------------------------------------- build


@pytest.mark.parametrize("cap", [1.0, 0.25])
def test_build_produces_a_frontier_no_asset_can_beat(est, cap):
    doc = fr.build(est, rf=0.03, cap=cap, n_points=25)
    f = doc["frontier"]
    assert doc["failed_targets"] == [], (
        f"infeasible targets {doc['failed_targets']}: max_feasible_return disagrees with the QP"
    )
    vols = np.array([p["vol"] for p in f])
    rets = np.array([p["ret"] for p in f])
    assert (np.diff(vols) > 0).all() and (np.diff(rets) > 0).all()
    if cap == 1.0:
        sigma = np.sqrt(np.diag(est.cov.to_numpy()))
        for i, sym in enumerate(est.mu.index):
            if sigma[i] > vols[-1]:
                continue
            assert float(np.interp(sigma[i], vols, rets)) >= est.mu.iloc[i] - 1e-6, sym


def test_build_min_variance_end_is_the_global_minimum_variance_portfolio(est):
    """Independent check on the left endpoint: 200 random feasible portfolios must all be
    riskier. A frontier whose left end is not the minimum is the one error that makes every
    point on it wrong, because the whole curve is parameterised from that return level."""
    doc = fr.build(est, rf=0.03, cap=1.0, n_points=15)
    cov = est.cov.to_numpy()
    floor = doc["frontier"][doc["min_vol_index"]]["vol"]
    rng = np.random.default_rng(11)
    for _ in range(200):
        w = rng.dirichlet(np.ones(len(est.mu)))
        assert np.sqrt(w @ cov @ w) >= floor - 1e-9


def test_build_tangency_is_the_best_sharpe_among_random_feasible_portfolios(est):
    doc = fr.build(est, rf=0.03, cap=1.0, n_points=25)
    mu, cov = est.mu.to_numpy(), est.cov.to_numpy()
    best = doc["frontier"][doc["max_sharpe_index"]]["sharpe"]
    rng = np.random.default_rng(12)
    for alpha in (0.2, 1.0, 5.0):
        for _ in range(150):
            w = rng.dirichlet(np.full(len(mu), alpha))
            _, _, s = fr.performance(w, mu, cov, rf=0.03)
            assert s <= best + 1e-9


def test_build_reports_the_cap_it_was_given_and_honours_it(est):
    doc = fr.build(est, rf=0.03, cap=0.25, n_points=15)
    assert doc["weight_cap"] == 0.25
    for p in doc["frontier"]:
        assert max(p["w"].values()) <= 0.25 + 1e-12
        assert sum(p["w"].values()) == pytest.approx(1.0, abs=1e-9)


def test_build_asset_table_uses_the_shrunk_diagonal(est, synthetic):
    meta = {s: (s, "synthetic", "equity") for s in synthetic.columns}
    rows = fr.asset_table(est, synthetic, rf=0.03, meta=meta)
    sigma = np.sqrt(np.diag(est.cov.to_numpy()))
    for i, row in enumerate(rows):
        assert row["vol"] == pytest.approx(float(sigma[i]), abs=1e-6)
