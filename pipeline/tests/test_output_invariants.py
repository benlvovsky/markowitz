"""Invariants on the JSON the SPA actually reads.

This file does not test that the optimiser is correct in the abstract; it tests that the
ARTIFACT is internally consistent, because the SPA reads only the artifact and the browser
has no way to notice a lie in it. Three classes of bug this is aimed at:

1. A reported (ret, vol, sharpe) that does not match the weights shipped beside it. This
   is the plausible-looking failure: the solver's own `portfolio_performance()` is
   computed for the RAW weight vector, while what ships is the CLEANED one, so reporting
   the former next to the latter is off by the rounding -- invisible in a chart, and the
   number every reader would quote.
2. An asset dot plotted ABOVE the frontier curve. Mathematically impossible (every single
   asset is a feasible portfolio, so the frontier is at least as good at that risk level),
   which makes it a pure artifact bug -- a units mismatch, a stale file, sample vs shrunk
   variance on the dots. It is also the one error a reader WILL spot, and it discredits
   the whole picture.
3. A frontier that is not a monotone path. The browser interpolates between consecutive
   array entries; if the array is not sorted by both risk and return, dragging the handle
   moves it backwards.

The tolerances are set by what the JSON can express, not by taste: weights are rounded to
1e-6 and renormalised, so a 116-term dot product carries ~1e-6 * sqrt(116) ~ 1e-5 of
slack, and volatility is a square root of a quadratic form in those weights.
"""
from __future__ import annotations

import numpy as np
import pytest

CONSISTENCY_TOL = 2e-5


def _w_vec(point, index, n):
    w = np.zeros(n)
    for sym, x in point["w"].items():
        w[index[sym]] = x
    return w


def test_manifest_lists_every_universe_member_as_kept_or_excluded(manifest):
    """Every symbol is accounted for: an asset that vanished silently is the same bug as a
    wrong price, because the frontier is a statement about a specific universe."""
    kept = {a["symbol"] for a in manifest["assets"]}
    excluded = set(manifest["excluded"]["fetch_failed"]) | set(manifest["excluded"]["window_or_coverage"])
    assert kept & excluded == set(), f"symbol both kept and excluded: {kept & excluded}"
    assert len(kept) == manifest["n_assets"]
    assert len(kept) + len(excluded) == manifest["n_universe"], (
        f"{manifest['n_universe']} in universe, {len(kept)} kept + {len(excluded)} excluded"
    )


def test_stats_symbol_order_matches_manifest_assets(manifest, stats):
    """The covariance matrix is indexed positionally; a different order in the two files
    would silently pair every asset with the wrong row."""
    assert [a["symbol"] for a in manifest["assets"]] == stats["symbols"]


def test_asset_dot_coordinates_are_the_shrunk_diagonal(manifest, stats):
    """`assets[i].vol` must be sqrt(cov[i][i]) -- the dot and the curve share one
    coordinate system, or single assets appear to beat the frontier by a percent."""
    for i, a in enumerate(manifest["assets"]):
        assert a["vol"] == pytest.approx(float(np.sqrt(stats["cov_v"][i, i])), abs=1e-6), a["symbol"]
        assert a["ret"] == pytest.approx(float(stats["mu_v"][i]), abs=1e-8), a["symbol"]


def test_covariance_is_symmetric_and_positive_semidefinite(stats):
    cov = stats["cov_v"]
    n = len(stats["symbols"])
    assert cov.shape == (n, n)
    assert np.allclose(cov, cov.T, atol=1e-9), "covariance is not symmetric"
    assert (np.diag(cov) > 0).all(), "a zero or negative variance on the diagonal"
    lo = float(np.linalg.eigvalsh(cov).min())
    # Rounding to 1e-9 for transport can push a barely-PSD matrix slightly negative; the
    # browser computes w'Sw from these exact numbers, and a materially negative eigenvalue
    # would let it display a negative variance.
    assert lo > -1e-8, f"smallest eigenvalue {lo:.3e}: the shipped matrix is not PSD"


def test_no_nonfinite_numbers_anywhere(manifest, stats, history, frontiers):
    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, float):
            assert np.isfinite(node), f"non-finite at {path}"

    for name, doc in [("manifest", manifest), ("history", history), *frontiers.items()]:
        walk(doc, name)
    walk(stats["mu"], "stats.mu")


# --------------------------------------------------------------------------- per-cap file


def test_every_frontier_point_is_a_feasible_long_only_portfolio(frontiers, stats, cap_slug):
    doc = frontiers[cap_slug]
    cap = doc["weight_cap"]
    n = len(stats["symbols"])
    for i, p in enumerate(doc["frontier"]):
        w = _w_vec(p, stats["index"], n)
        assert w.sum() == pytest.approx(1.0, abs=1e-6), f"point {i} weights sum to {w.sum()}"
        assert (w >= 0).all(), f"point {i} has a negative weight"
        assert w.max() <= cap + 1e-9, f"point {i} breaches cap {cap}: max {w.max()}"
        assert p["w"], f"point {i} ships no weights"
        assert all(x > 0 for x in p["w"].values()), f"point {i} ships an explicit zero"


def test_no_frontier_point_ships_a_working_field(frontiers, cap_slug):
    """`_point` attaches unrounded values under `frontier.EXACT` for `build` to locate the
    extrema with, and `build` deletes them. If that delete is ever lost the file grows a
    duplicate, higher-precision copy of every number in it -- which no other test would notice,
    and which a reader would reasonably believe over the rounded fields."""
    for i, p in enumerate(frontiers[cap_slug]["frontier"]):
        assert set(p) == {"ret", "vol", "sharpe", "w"}, f"point {i} ships {sorted(p)}"


def test_reported_performance_is_recomputable_from_the_shipped_weights(frontiers, stats, cap_slug):
    """The load-bearing test: ret == w.mu and vol == sqrt(w'Sw) for the CLEANED weights."""
    doc = frontiers[cap_slug]
    mu, cov = stats["mu_v"], stats["cov_v"]
    n = len(stats["symbols"])
    rf = doc["risk_free_rate"]
    for i, p in enumerate(doc["frontier"]):
        w = _w_vec(p, stats["index"], n)
        assert p["ret"] == pytest.approx(float(w @ mu), abs=CONSISTENCY_TOL), f"point {i} return"
        vol = float(np.sqrt(w @ cov @ w))
        assert p["vol"] == pytest.approx(vol, abs=CONSISTENCY_TOL), f"point {i} volatility"
        assert p["sharpe"] == pytest.approx((p["ret"] - rf) / p["vol"], abs=1e-4), f"point {i} sharpe"


def test_frontier_is_a_strictly_monotone_path_in_both_risk_and_return(frontiers, cap_slug):
    """The browser drags along array order; a non-monotone array makes the handle jump back."""
    f = frontiers[cap_slug]["frontier"]
    assert len(f) >= 10, f"only {len(f)} points: not enough to interpolate"
    rets = [p["ret"] for p in f]
    vols = [p["vol"] for p in f]
    assert rets == sorted(rets), "returns are not ascending"
    assert vols == sorted(vols), "volatilities are not ascending"
    assert len(set(rets)) == len(rets), "duplicate return level"


def test_no_frontier_point_is_dominated_by_another(frontiers, cap_slug):
    f = frontiers[cap_slug]["frontier"]
    for i, a in enumerate(f):
        for j, b in enumerate(f):
            if i == j:
                continue
            dominated = b["ret"] >= a["ret"] + 1e-9 and b["vol"] <= a["vol"] - 1e-9
            assert not dominated, f"point {i} dominated by {j}"


def test_min_vol_and_max_sharpe_indices_point_at_the_actual_extrema(frontiers, stats, cap_slug):
    """The indices must be the extrema of the QUANTITIES, not of the rounded fields.

    This test used to read `f[i]["sharpe"]` and it was worthless in exactly the way CLAUDE.md
    warns about: that field is rounded to 6 decimals, the frontier is stationary at the
    tangency, so the tangency and its neighbour ship as the SAME number, and both this test and
    the pipeline resolved the tie by taking whichever came first. Two implementations agreeing
    on a tie-break is not a check. It shipped `max_sharpe_index = 39` where the genuinely best
    solved point was 40 (by 1.1e-9), and the browser -- which recomputes from the weights --
    caught what this test had blessed.

    So: recompute-to-recompute. Both sides evaluate the same quadratic forms from the shipped
    weights and the shipped covariance, which is also what the browser does. That is robust here
    despite the 1.1e-9 margin, and not by luck: adjacent frontier points differ in weights by
    ~1e-4, so rounding the covariance to 9 decimals perturbs BOTH Sharpe ratios almost
    identically and the DIFFERENCE between them survives to ~1e-12. Measured: the gap computed
    from the shipped 9-decimal covariance is 1.111e-9 against 1.110e-9 at full precision.
    """
    doc = frontiers[cap_slug]
    f = doc["frontier"]
    mu, cov = stats["mu_v"], stats["cov_v"]
    n = len(stats["symbols"])
    rf = doc["risk_free_rate"]

    vols, sharpes = [], []
    for p in f:
        w = _w_vec(p, stats["index"], n)
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        vols.append(vol)
        sharpes.append((ret - rf) / vol)

    assert doc["min_vol_index"] == min(range(len(f)), key=lambda i: vols[i])
    assert doc["max_sharpe_index"] == max(range(len(f)), key=lambda i: sharpes[i])
    # The frontier is ordered by risk, so the minimum-variance portfolio is its left end.
    assert doc["min_vol_index"] == 0


def test_the_shipped_rounded_fields_do_not_contradict_the_extremum_indices(frontiers, cap_slug):
    """The weaker claim, from the file alone: nothing in it may LOOK better than the tangency.

    A reader (or a chart tooltip) comparing the shipped `sharpe` column would otherwise find a
    point that beats the one labelled tangency. The bound is 1e-6 -- two units in the field's
    last place -- so a genuine tie passes and a real off-by-one does not: one point either side
    of the tangency is 3e-5 to 6e-5 worse, thirty times the bound.
    """
    f = frontiers[cap_slug]["frontier"]
    best = f[frontiers[cap_slug]["max_sharpe_index"]]["sharpe"]
    worst = max(p["sharpe"] - best for p in f)
    assert worst <= 1e-6, f"a shipped point's sharpe exceeds the tangency's by {worst:.2e}"
    floor = f[frontiers[cap_slug]["min_vol_index"]]["vol"]
    assert min(p["vol"] - floor for p in f) >= -1e-8


def test_tangency_beats_every_interpolated_point_by_more_than_rounding(frontiers, stats, cap_slug):
    """The point count has to be dense enough that the drawn tangency IS the tangency.

    The browser can drag to any convex combination of adjacent frontier points, so it can
    land on Sharpe ratios the solver never evaluated. If a midpoint between two solved
    points beat the reported max-Sharpe portfolio by a visible margin, the marker labelled
    "tangency" would sit somewhere a reader could beat by hand -- which is a real defect in
    the picture even though every individual number in the file is correct.
    """
    doc = frontiers[cap_slug]
    f = doc["frontier"]
    mu, cov = stats["mu_v"], stats["cov_v"]
    n = len(stats["symbols"])
    rf = doc["risk_free_rate"]
    best = f[doc["max_sharpe_index"]]["sharpe"]
    ws = [_w_vec(p, stats["index"], n) for p in f]
    worst_excess = 0.0
    for a, b in zip(ws, ws[1:]):
        for t in np.linspace(0.05, 0.95, 19):
            w = (1 - t) * a + t * b
            ret = float(w @ mu)
            vol = float(np.sqrt(w @ cov @ w))
            worst_excess = max(worst_excess, (ret - rf) / vol - best)
    assert worst_excess < 5e-4, f"an interpolated point beats the tangency by {worst_excess:.2e}"


def test_interpolating_weights_between_neighbours_stays_feasible(frontiers, stats, cap_slug):
    """What the SPA does to produce a dragged portfolio, asserted to produce a real one."""
    doc = frontiers[cap_slug]
    cap = doc["weight_cap"]
    n = len(stats["symbols"])
    ws = [_w_vec(p, stats["index"], n) for p in doc["frontier"]]
    for i, (a, b) in enumerate(zip(ws, ws[1:])):
        for t in (0.25, 0.5, 0.75):
            w = (1 - t) * a + t * b
            assert w.sum() == pytest.approx(1.0, abs=1e-6), f"segment {i}"
            assert (w >= 0).all(), f"segment {i}"
            assert w.max() <= cap + 1e-9, f"segment {i} breaches cap"


def test_no_single_asset_plots_above_the_frontier(manifest, frontiers, cap_slug):
    """Every asset is a feasible portfolio, so the curve must weakly dominate every dot.

    Only tested at cap=1.0 as an equality-grade claim: a per-asset cap of 20% or 10% makes
    a single 100%-weight asset INFEASIBLE, so a concentrated high-return asset legitimately
    plots above its own constrained frontier. Asserting otherwise there would be asserting
    that the constraint does nothing.
    """
    doc = frontiers[cap_slug]
    if doc["weight_cap"] < 1.0:
        pytest.skip("a single asset is not feasible under a per-asset cap")
    f = doc["frontier"]
    vols = np.array([p["vol"] for p in f])
    rets = np.array([p["ret"] for p in f])
    for a in manifest["assets"]:
        assert a["vol"] >= vols[0] - 1e-6, f"{a['symbol']} is less risky than the min-variance portfolio"
        if a["vol"] > vols[-1]:
            continue  # riskier than the highest-return portfolio: no curve to compare against
        best_here = float(np.interp(a["vol"], vols, rets))
        assert best_here >= a["ret"] - 1e-4, (
            f"{a['symbol']} plots above the frontier: ret {a['ret']:.5f} vs curve {best_here:.5f} "
            f"at vol {a['vol']:.5f}"
        )


# ------------------------------------------------------------------------------ across caps


def test_tightening_the_cap_cannot_improve_the_optimum(frontiers):
    """A smaller feasible set cannot contain a better point. True for the tangency Sharpe
    and for the minimum achievable variance, and it is the one invariant that catches a
    cap that was parsed, logged, and then not actually applied to the QP."""
    by_cap = sorted(frontiers.values(), key=lambda d: -d["weight_cap"])
    if len(by_cap) < 2:
        pytest.skip("only one cap built")
    for looser, tighter in zip(by_cap, by_cap[1:]):
        lo, ti = looser["weight_cap"], tighter["weight_cap"]
        s_loose = looser["frontier"][looser["max_sharpe_index"]]["sharpe"]
        s_tight = tighter["frontier"][tighter["max_sharpe_index"]]["sharpe"]
        assert s_tight <= s_loose + 1e-4, f"cap {ti} beat cap {lo} on Sharpe: {s_tight} > {s_loose}"
        v_loose = looser["frontier"][looser["min_vol_index"]]["vol"]
        v_tight = tighter["frontier"][tighter["min_vol_index"]]["vol"]
        assert v_tight >= v_loose - 1e-6, f"cap {ti} reached lower variance than cap {lo}"
        assert tighter["return_range"][1] <= looser["return_range"][1] + 1e-6


def test_a_tighter_cap_forces_more_holdings_at_the_tangency(frontiers):
    """Not arithmetic, but the property the three files exist to show: the cap is the knob
    that trades away concentration. If it stopped doing that, the comparison the README
    draws would be wrong and nothing else in the suite would notice."""
    by_cap = sorted(frontiers.values(), key=lambda d: -d["weight_cap"])
    if len(by_cap) < 2:
        pytest.skip("only one cap built")
    counts = [len(d["frontier"][d["max_sharpe_index"]]["w"]) for d in by_cap]
    assert counts == sorted(counts), f"holdings did not grow as the cap tightened: {counts}"
    for d, k in zip(by_cap, counts):
        assert k >= int(np.ceil(1.0 / d["weight_cap"])), (
            f"cap {d['weight_cap']} tangency holds {k}, below the {np.ceil(1 / d['weight_cap']):.0f} "
            "a cap arithmetically requires"
        )


# ---------------------------------------------------------------------------------- history


def test_history_rows_line_up_with_dates_for_every_symbol(history, stats):
    n = len(history["dates"])
    assert history["symbols"] == stats["symbols"]
    for sym in history["symbols"]:
        assert len(history["returns"][sym]) == n, f"{sym}: {len(history['returns'][sym])} != {n}"
    assert history["dates"] == sorted(history["dates"]), "months out of order"
    assert len(set(history["dates"])) == n, "duplicate month"


def test_monthly_returns_are_returns_not_prices(history):
    """A `pct_change` dropped or applied to the wrong frame ships price levels here, and a
    growth curve built from levels still renders -- as a wildly wrong one."""
    flat = np.array([x for sym in history["symbols"] for x in history["returns"][sym]])
    assert np.abs(flat).max() < 3.0, f"largest monthly move {flat.max():.2f}: these look like levels"
    assert np.abs(flat).mean() < 0.1, f"mean absolute monthly move {np.abs(flat).mean():.3f}"
    assert (flat > -1.0).all(), "a monthly return at or below -100%"


def test_monthly_compounding_reproduces_the_annualised_mean_return(history, manifest):
    """The bridge between the two artifacts, at EXACT tolerance.

    `history` is anchored at the daily panel's first bar, so chained month-end price ratios
    telescope to the same P_last / P_first the daily geometric mean is computed from. The
    two paths must therefore agree to rounding, not merely to a percentage point -- and
    that sharpness is the whole value of the test. Any window or resample-rule mistake
    (resampling on 'M' instead of 'ME', dropping the anchor, a stale history file against a
    fresh manifest) shifts one path relative to the other and shows up here immediately.

    The tolerance is set by the transport precision and nothing else: the telescoping is
    exact to float64, and the only slack is that ~189 returns ship rounded to 1e-6, which
    contributes at most ~3e-7 of annualised return (measured across this universe). 2e-6
    leaves an order of magnitude of headroom while still being 25x tighter than the
    off-by-one in `years` that this test caught -- annualising over the 3,939 price BARS
    instead of the 3,938 RETURNS, worth 5e-5.
    """
    years = history["years"]
    assert years == pytest.approx(manifest["window"]["years"], abs=1e-6)
    assert history["anchor"] == manifest["window"]["start"]
    for a in manifest["assets"]:
        r = np.asarray(history["returns"][a["symbol"]], dtype=float)
        monthly_annualised = float(np.prod(1.0 + r) ** (1.0 / years) - 1.0)
        assert monthly_annualised == pytest.approx(a["ret"], abs=2e-6), (
            f"{a['symbol']}: monthly path gives {monthly_annualised:.6f}, mu says {a['ret']:.6f}"
        )
