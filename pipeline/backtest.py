"""Walk-forward test: estimate on data up to a cutoff, hold the answer, measure what happened.

    python -u pipeline/backtest.py                     # 1, 2 and 3 years back
    python -u pipeline/backtest.py --years 1,2,3,5 --out pipeline/results/backtest.json
    python -u pipeline/backtest.py --mu momentum --out pipeline/results/backtest_momentum.json

Run from the repo root. Reads the same price store `build.py` reads and produces one JSON of
results plus a printed table; it writes NOTHING into `web/public/data`. That is deliberate --
`data.loadBundle` accepts exactly six files stamped with one `generated_at`, and a seventh
would either break that invariant or have to be excluded from it, which is a change to the
shipped page's contract for the sake of an offline analysis.

WHAT IS ACTUALLY BEING TESTED
-----------------------------
The page shows an IN-SAMPLE frontier: `mu` and `Sigma` are estimated over the whole window
and the tangency portfolio is the best Sharpe ratio *in that window*, which the optimiser was
shown. Its growth curve says so and cannot be read as evidence. This file is the missing
half. For each cutoff:

  1. Truncate the panel at the cutoff. Estimate `mu` and `Sigma` on the TRAINING half only.
  2. Take `rf` as the actual 13-week T-bill yield ON the cutoff date.
  3. Solve the tangency portfolio -- the point where the capital market line touches the
     frontier -- with `frontier._solve`, the same single-QP path the shipped frontier uses.
     Not an interpolation along a stored curve: the same function, so "tangency" cannot mean
     two things.
  4. Hold those weights across the TEST half and measure what they returned.

`rf` IS A PIPELINE PARAMETER HERE, AND ONLY HERE. Everywhere else in this repo it is a
browser-side slider precisely because the frontier does not depend on it. The tangency
portfolio does, and a backtest is a claim about the past: the investor standing at the cutoff
faced the T-bill yield of that day, not the one the reader drags to today. Pinning it to
history is what makes the answer a measurement instead of a view setting. `--rf` overrides it
for a sensitivity check; the default is `^IRX` as of the cutoff.

THREE CUTOFFS IS THREE OBSERVATIONS, AND THEY OVERLAP
----------------------------------------------------
The 3-year test window CONTAINS the 2-year window, which contains the 1-year window, so the
three rows are not three independent trials of anything -- they share most of their returns
and will agree or disagree together. Nor is the universe innocent: it is 116 funds that exist
today and have quotes back to 2011, so every one of them survived the period being tested.
Read the table as "here is what this particular portfolio did", not as an estimate of what
mean-variance optimisation does. The comparators are in the table for that reason -- an
equal-weight basket of the same 116 assets and the benchmark are what the optimiser has to
beat before any of its machinery has earned anything.

THE DIAGNOSTIC IS MORE INFORMATIVE THAN THE RESULT
--------------------------------------------------
`mu_rank_correlation` is the cross-sectional Spearman correlation between the annualised
return each asset showed in the training window and the one it went on to show in the test
window. It is the assumption the whole exercise rests on, measured directly: if past
annualised return does not rank future annualised return, then the tangency portfolio is
sorting on noise and no amount of covariance care can fix it. `vol_rank_correlation` is the
same for risk, and the two are expected to differ enormously -- volatility persists, mean
returns do not. That contrast is the point, and it is why `--caps` matters: the cap is the
only lever in the file that limits how far the optimiser can act on the weaker of the two.

WHICH EXPECTED RETURN, AND WHY THERE IS A CHOICE
------------------------------------------------
`mu_rank_correlation` measured +0.25 on average for the sample mean, so the obvious question is
whether a better forecast exists. `--mu momentum` swaps in the one with the strongest published
record -- each asset's return over the past year, stopping a month short -- leaving EVERYTHING
else identical: same covariance, same caps, same solver, same windows, same matched-risk
comparison. So the two runs differ in one input and the diagnostic is directly comparable.

Two things make that comparison honest rather than a horse race:

  `momentum_top{N}`, an equal-weight basket of the N highest-scoring assets, is emitted under
  BOTH models. It is the signal held without the optimiser, so a bad momentum result can be
  attributed -- either the signal did not rank the assets, or the optimiser spent it.

  A weight vector is chosen ONCE per holding period. At a one-year hold that is momentum
  refreshed annually, and the effect is documented at monthly refresh. This mode therefore
  understates momentum, and the amount is not knowable from inside this file; `run_rolling`
  steps in whole years because `pd.DateOffset` refuses a fractional one. It is recorded in
  `method.caveats` rather than corrected for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build  # noqa: E402
import fetch  # noqa: E402
import frontier as fr  # noqa: E402
import store  # noqa: E402
import universe as uni  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICE_DIR = ROOT / "pipeline" / "data" / "prices"
DEFAULT_OUT = ROOT / "pipeline" / "results" / "backtest.json"
IRX_CACHE = ROOT / "pipeline" / "data" / "irx.parquet"


# ----------------------------------------------------- prices from outside the price store

def join_extra(panel: pd.DataFrame, path: Path, start: str, min_coverage: float,
               allow_empty: bool = False) -> tuple[pd.DataFrame, dict]:
    """Add second-vendor total-return columns to the panel, on the PANEL's trading days.

    The one seam through which `delisted.py` reaches the optimiser, and the reason it is a seam
    rather than an extra `price_dir` is in that module's docstring: the store's invariants rest on
    one vendor shipping `close`, an event log and an `adjclose` to check the reconstruction
    against, and this vendor ships a level and nothing else.

    THE JOIN IS AN INTERSECTION AND THE FILTERS ARE THE PANEL'S OWN. `load_panel` has already
    dropped the store's late arrivals and thin coverage; an extra column let in on looser terms
    would be an asset admitted because it came from a different file, which is exactly the
    comparison this whole exercise exists to avoid. So the same two filters run again here, and
    the panel's date index is authoritative -- a column that does not cover it is dropped with a
    reason rather than forward filled.

    Reindexing onto the panel index CANNOT lengthen the panel: the columns are added to a fixed
    set of rows, so restoring twelve delisted names cannot move the window a single bar, and the
    restored run is comparable with the run that excluded them bar for bar. That is the property
    worth having, and it is why nothing here touches `panel.index`.

    `allow_empty` IS FOR A CALLER THAT JOINS ONCE PER WINDOW, AND THE DEFAULT IS THE RIGHT ONE
    FOR EVERY OTHER CALLER. On a single-window run, `--extra-panel` restoring nothing means the
    flag did nothing, and the run must not proceed looking like a restored one -- so it raises.
    `pit.py` joins the same file into ten windows and the earliest window legitimately has no
    restorable name in it: the second vendor serves ten years back from each ticker's last bar,
    which does not reach a training window that opens in 2011. Refusing there would make the
    honest outcome unrunnable. The count that survived is recorded per window either way, so
    "restored nothing" is reported rather than assumed.
    """
    extra = pd.read_parquet(path)
    if not isinstance(extra.index, pd.DatetimeIndex):
        raise SystemExit(f"{path}: the index is {type(extra.index).__name__}, not dates")
    clash = [c for c in extra.columns if c in panel.columns]
    if clash:
        raise SystemExit(
            f"{path}: {clash} are already in the panel. An extra column that shadows a stored "
            "one would silently swap vendors for that asset."
        )
    kept: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}
    for col in extra.columns:
        s = extra[col].dropna()
        if s.index.min() > pd.Timestamp(start):
            dropped[col] = f"history starts {s.index.min().date()}, after {start}"
            continue
        on_panel = s.reindex(panel.index)
        cov = float(on_panel.notna().mean())
        if cov < min_coverage:
            dropped[col] = f"coverage {cov:.3f} < {min_coverage} of the panel's bars"
            continue
        kept[col] = on_panel
    if not kept and not allow_empty:
        raise SystemExit(f"{path}: no column survived the panel's own filters ({dropped})")
    joined = pd.concat([panel, pd.DataFrame(kept, index=panel.index)], axis=1)
    before = len(joined)
    joined = joined.dropna(how="any")
    if len(joined) != before:
        # Only reachable when a kept column has a hole INSIDE the window, which `min_coverage`
        # tolerates by design. Refuse rather than delete the bar from every other asset.
        raise SystemExit(
            f"{path}: the join would delete {before - len(joined)} bars from all "
            f"{panel.shape[1]} stored assets to accommodate a hole in an extra column"
        )
    info = {"file": path.name, "added": sorted(kept), "dropped": dropped,
            "source": "second vendor via pipeline/delisted.py, NOT the price store"}
    return joined, info


# --------------------------------------------------------------------- the risk-free series

def load_irx(cache: Path = IRX_CACHE, refresh: bool = False) -> pd.Series:
    """The 13-week T-bill yield as a decimal, daily, cached to its own file.

    NOT written into the price store, which holds the universe and only the universe: `^IRX`
    is a yield rather than a price, `store.total_return` would happily apply a dividend
    adjustment to it, and `stored_symbols` is how `--skip-fetch` decides what it has. A
    quoted yield sitting in there as a 133rd symbol is a thing every one of those paths
    would have to learn to ignore.

    NOR does it go through `fetch.parse`, which is the same distinction with teeth on it:
    `parse` rejects a non-positive adjusted close, and it is right to -- a share that trades
    at zero is bad data. A 13-week bill quoted at 0.00% is not bad data, it is 2020 and 2021,
    where the yield printed zero for months at a stretch. `build.fetch_rf` never meets this
    because it asks for one month; a 30-year window walks straight into it. So the yield is
    validated on its own terms, and the floor is BELOW ZERO rather than at it: seven sessions
    in March 2020 print between -0.03% and -0.105%, which is a real market -- bills bid above
    par in a flight to quality -- and clamping them to zero would fabricate a rate nobody was
    paid. The band is a sanity check on the feed, not a model of what yields may do.
    """
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)["rf"]
    payload = fetch._get(build.RF_TICKER, rng="30y")
    r = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not r or not r.get("timestamp"):
        raise ValueError(f"{build.RF_TICKER}: no timestamps in the chart payload")
    closes = r["indicators"]["quote"][0]["close"]
    rf = pd.Series(
        [None if c is None else float(c) / 100.0 for c in closes],
        index=pd.Index([fetch._ny_date(t) for t in r["timestamp"]], name="date"),
        dtype=float,
        name="rf",
    ).dropna()
    rf = rf[~rf.index.duplicated(keep="last")].sort_index()
    if rf.empty or not ((rf > -0.01) & (rf < 0.25)).all():
        raise ValueError(f"{build.RF_TICKER}: {len(rf)} quotes, yields outside (-1%, 25%)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    rf.to_frame().to_parquet(cache)
    return rf


def rf_asof(irx: pd.Series, when: pd.Timestamp) -> float:
    """The yield quoted ON the cutoff, or the last one before it. `asof` and not
    interpolation or a nearby mean: a rate the investor could not yet have seen is a
    lookahead, and the whole file is worthless if one creeps in."""
    v = irx.asof(when)
    if pd.isna(v):
        raise ValueError(f"no {build.RF_TICKER} quote at or before {when.date()}")
    return float(v)


# ------------------------------------------------------------------- holding a weight vector

def value_path(prices: pd.DataFrame, w: np.ndarray, rebalance: str) -> pd.Series:
    """Daily portfolio value, starting at 1.0 on the panel's first bar.

    `prices` is a TOTAL-RETURN panel, so this is a total return and dividends are already in
    it. Two rules, because the difference between them is a real decision and not a detail:

      "hold"    -- buy once at the first bar and never trade. Weights drift with performance,
                   so by the end of a three-year window the portfolio is no longer the one
                   the optimiser chose. Zero cost, and it is what someone who bought and
                   walked away would actually have.
      "monthly" -- reset to `w` at every month end, which is what the SPA's growth curve
                   does and therefore the rule that keeps the two comparable.

    Computed on the VALUE PATH rather than as a weighted sum of asset returns, because those
    two are only the same thing under continuous rebalancing. `sum_i w_i * r_i` is the
    daily-rebalanced portfolio; using it while calling the result "monthly" overstates the
    return of anything with dispersion in it, and the error grows with the window.
    """
    if rebalance not in ("hold", "monthly"):
        raise ValueError(f"rebalance must be 'hold' or 'monthly': {rebalance!r}")
    P = prices.to_numpy(dtype=float)
    if rebalance == "hold":
        return pd.Series((P / P[0]) @ w, index=prices.index)

    v = np.empty(len(prices))
    v[0] = 1.0
    period = prices.index.to_period("M")
    base_row, base_val = 0, 1.0
    for t in range(1, len(prices)):
        # Rebalanced at the CLOSE of the last bar of the previous month, so the new segment
        # is priced off row t-1. Rebalancing on the first bar of the new month instead would
        # let one day of drift leak into every segment.
        if period[t] != period[t - 1]:
            base_row, base_val = t - 1, v[t - 1]
        v[t] = base_val * float((P[t] / P[base_row]) @ w)
    return pd.Series(v, index=prices.index)


def turnover_path(prices: pd.DataFrame, w: np.ndarray, rebalance: str) -> pd.Series:
    """Fraction of the portfolio TRADED on each bar: `sum_i |w_target_i - w_drifted_i|`.

    Buys plus sells, not the one-way half of it. A cost quoted per dollar traded is paid on
    both legs, so the two conventions differ by exactly 2x -- which is the size of the entire
    effect being measured, and is therefore worth being explicit about instead of tidy.

    Bar 0 is `sum |w| == 1.0`: the position has to be bought. Under "hold" that is the only
    trade there ever is. Under "monthly" the drifted weights are recomputed at each month
    boundary from the same base row `value_path` prices the segment off, so the two functions
    agree about when a trade happened and about what the portfolio looked like when it did.

    Separate from `value_path` and not folded into it, because the cost is PROPORTIONAL: a
    charge of c% of the portfolio scales its value and leaves every subsequent return and every
    subsequent *weight* untouched. So the turnover schedule of the net path is identical to the
    gross path's, and the net path is the gross path times a cumulative product. That identity
    is what lets both readings ship from one solve rather than from two.
    """
    if rebalance not in ("hold", "monthly"):
        raise ValueError(f"rebalance must be 'hold' or 'monthly': {rebalance!r}")
    t = np.zeros(len(prices))
    t[0] = float(np.abs(w).sum())
    if rebalance == "monthly":
        P = prices.to_numpy(dtype=float)
        period = prices.index.to_period("M")
        base_row = 0
        for i in range(1, len(prices)):
            if period[i] != period[i - 1]:
                rel = P[i - 1] / P[base_row]
                drifted = w * rel / float(rel @ w)
                t[i - 1] += float(np.abs(w - drifted).sum())
                base_row = i - 1
    return pd.Series(t, index=prices.index)


@dataclass(frozen=True)
class Costs:
    """What the investor pays that the optimiser never sees.

    A backtest with no costs in it is not neutral between two strategies -- it is a subsidy to
    whichever one trades more, and here that is never the index. An index fund charges a
    published annual fee and is bought once; an annually-reoptimised, monthly-rebalanced
    400-stock portfolio pays no management fee and trades constantly. Zeroing both is the
    comparison that flatters the optimiser, so both are modelled or neither is.

    `trade_bps` is the all-in one-way cost per dollar traded: commission plus half-spread plus
    impact. It is a PARAMETER and not a measurement -- nothing in this repo observes a spread --
    so the honest use is a sensitivity sweep, which is why `main` accepts a list.

    `expense` is per-symbol and annual, charged daily against whatever weight the portfolio
    holds in that symbol. A symbol absent from it is charged NOTHING, which is correct for a
    single stock and wrong for a fund: on the 116-fund ETF universe every holding charges a fee
    this map does not know, so a net run there under-charges the optimised side and only the
    benchmark pays. Stated rather than fixed, because the numbers needed are 116 prospectuses.

    Defaults are zero, and `zero()` is checked before any net column is emitted, so every
    already-published result is reproduced bit for bit rather than approximately.
    """

    trade_bps: float = 0.0
    expense: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trade_bps < 0:
            raise ValueError(f"trade_bps must not be negative: {self.trade_bps}")
        bad = {s: e for s, e in self.expense.items() if not 0.0 <= e < 1.0}
        if bad:
            raise ValueError(f"expense ratios must be in [0, 1): {bad}")

    def zero(self) -> bool:
        return self.trade_bps == 0.0 and not any(self.expense.values())

    def describe(self) -> dict:
        return {"trade_bps_per_dollar_traded": self.trade_bps,
                "turnover_convention": "buys + sells (sum |dw|), not the one-way half",
                "expense_ratios": dict(self.expense),
                "not_modelled": ["taxes", "market impact beyond trade_bps",
                                 "the expense ratios of any fund absent from expense_ratios",
                                 "the bid-ask cost of the daily mix reset in `lever` is charged, "
                                 "but borrowing is charged at ^IRX with no spread over it"]}


# SPY's published annual expense ratio. The only fee number in this repo, and it is here rather
# than in a universe file because it is a property of the comparison, not of the symbol set.
EXPENSE_RATIOS = {"SPY": 0.000945}
# One-way, all-in, large-cap US equity. A default and not a finding -- see `Costs`.
DEFAULT_TRADE_BPS = 5.0


def cost_factor(turnover: pd.Series, expense_annual: float, costs: Costs) -> pd.Series:
    """The multiplicative haircut a gross value path takes to become a net one.

    Two pieces, both proportional and therefore commutative with returns: the traded fraction
    charged at `trade_bps` on the bars it happened, and the annual fee accrued on every bar.
    Compounded, so a 1.0 factor on a zero-cost run is exact and not merely close.
    """
    trade = (1.0 - costs.trade_bps / 1e4 * turnover.to_numpy()).cumprod()
    daily = (1.0 - expense_annual) ** (1.0 / fetch.TRADING_DAYS_PER_YEAR)
    fee = daily ** np.arange(len(turnover), dtype=float)
    return pd.Series(trade * fee, index=turnover.index)


def weighted_expense(columns, w: np.ndarray, costs: Costs) -> float:
    return float(sum(wi * costs.expense.get(c, 0.0) for c, wi in zip(columns, w)))


def mix_turnover(v: pd.Series, k: float, rf_annual: pd.Series) -> pd.Series:
    """Turnover of `lever`'s daily reset back to `k`, in units of equity.

    `lever` holds `k` of the fund and `1 - k` in cash and resets the mix every bar. After one
    day's return the fund leg has drifted to `k(1+r) / (1+r_mix)` of a portfolio that is again
    worth 1, so both legs trade that far -- hence the factor of 2, matching `turnover_path`'s
    buys-plus-sells convention. The docstring on `lever` says the daily reset "is free here, and
    it is free nowhere"; this is the function that stops that being true.
    """
    cash = (1.0 + rf_annual) ** (1.0 / fetch.TRADING_DAYS_PER_YEAR) - 1.0
    r = v.pct_change()
    mixed = (k * r + (1.0 - k) * cash).fillna(0.0)
    drift = (k * (1.0 + r) / (1.0 + mixed) - k).abs().fillna(0.0)
    return 2.0 * drift


def lever(v: pd.Series, k: float, rf_annual: pd.Series) -> pd.Series:
    """The capital market line PAST the tangency point, as a value path: `k` of the portfolio,
    financed by holding `1 - k` in cash (borrowing when `k > 1`), the mix reset every bar.

    This is what the CML is FOR. The tangency portfolio is not a rival to the benchmark -- it is
    the risky half of a two-asset choice whose other half is a T-bill, and where you sit on the
    line between them is a separate decision about how much risk you want. Comparing an
    unlevered 7%-volatility portfolio's return against a 17%-volatility index answers a question
    nobody asked; levering it to the index's risk asks the one that has an answer.

    THE LEVERAGE FACTOR IS A DECISION MADE AT THE CUTOFF (see `evaluate`); the financing RATE is
    a cost and is paid as it came. Only the first could be a lookahead. Rates moved by more than
    500bp inside these windows, so charging a whole year of borrowing at the rate quoted on the
    cutoff would be a subsidy nobody received.

    Reset every bar rather than levered once and left, because a constant exposure is what
    matching a risk level MEANS: borrow once and the leverage falls as the position gains, so the
    portfolio stops carrying the risk it was chosen to carry. It is also the flattering reading
    -- daily rebalancing of a levered position is free here, and it is free nowhere.
    """
    if len(rf_annual) != len(v) or not (rf_annual.index == v.index).all():
        raise ValueError("the financing rate must be quoted on exactly the value path's bars")
    cash = (1.0 + rf_annual) ** (1.0 / fetch.TRADING_DAYS_PER_YEAR) - 1.0
    mixed = (k * v.pct_change() + (1.0 - k) * cash).fillna(0.0)
    return (1.0 + mixed).cumprod()


def realized(v: pd.Series, rf: float) -> dict:
    """What a value path actually did. Annualised on the same convention `frontier.estimate`
    uses for `mu` -- geometric, 252 trading days, denominator in RETURNS not bars -- so the
    predicted and realized columns of the table are the same kind of number.

    `max_drawdown` is a POSITIVE fraction, as it is in `export.ts`: a bare drawdown is
    ambiguous in sign the moment it leaves the page that produced it.
    """
    n = len(v) - 1
    if n < 2:
        raise ValueError(f"a {len(v)}-bar test window measures nothing")
    growth = float(v.iloc[-1])
    ret = growth ** (fetch.TRADING_DAYS_PER_YEAR / n) - 1.0
    d = v.pct_change().dropna()
    vol = float(d.std(ddof=1)) * np.sqrt(fetch.TRADING_DAYS_PER_YEAR)
    return {
        "growth": round(growth, 6),
        "ret": round(float(ret), 6),
        "vol": round(vol, 6),
        "sharpe": round(float((ret - rf) / vol), 4) if vol > 0 else None,
        "max_drawdown": round(float((1.0 - v / v.cummax()).max()), 6),
    }


def predicted(w: np.ndarray, est: fr.Estimates, rf: float) -> dict:
    ret, vol, sharpe = fr.performance(w, est.mu.to_numpy(), est.cov.to_numpy(), rf)
    return {"ret": round(ret, 6), "vol": round(vol, 6), "sharpe": round(sharpe, 4)}


# --------------------------------------------------------- what the optimiser is told to expect

def _bar_at_or_before(panel: pd.DataFrame, when: pd.Timestamp, what: str) -> pd.Timestamp:
    """The last bar AT OR BEFORE a date, so every window edge is a real trading day the
    investor could have acted on."""
    bar = panel.index[panel.index <= when].max()
    if pd.isna(bar):
        raise ValueError(f"{what} ({when.date()}) is before the panel starts")
    return bar


MU_MODELS = ("history", "momentum")


@dataclass(frozen=True)
class MuModel:
    """WHICH EXPECTED RETURN THE OPTIMISER IS HANDED. One object rather than four loose
    parameters, because it has to be threaded through `run_cutoff` and `run_rolling` unchanged
    and a bare `12` arriving in the wrong slot of a seven-argument call would be a silently
    different test.

      "history"  -- the sample geometric mean over the whole training window. What `build.py`
                    ships and what `frontier.estimate` returns. Measured here at a mean
                    cross-sectional rank correlation of +0.25 against what the assets went on
                    to do, which is the finding that motivated the alternative.
      "momentum" -- the return over `lookback_months` ending `skip_months` before the cutoff.

    THE COVARIANCE IS NOT TOUCHED by either. It stays the Ledoit-Wolf estimate over the full
    training window, and that is a decision: risk is the half of the input that measured +0.91,
    so re-estimating it from an 11-month window as well would change two things at once and the
    result could not be attributed to either. Momentum is a claim about returns only.
    """
    kind: str = "history"
    lookback_months: int = 12
    skip_months: int = 1
    top_n: int = 10

    def __post_init__(self) -> None:
        if self.kind not in MU_MODELS:
            raise ValueError(f"unknown expected-return model {self.kind!r}, expected one of {MU_MODELS}")
        if not 0 <= self.skip_months < self.lookback_months:
            raise ValueError(
                f"need 0 <= skip < lookback, got skip={self.skip_months} "
                f"lookback={self.lookback_months}"
            )
        if self.top_n < 1:
            raise ValueError(f"top_n must be at least 1: {self.top_n}")

    def describe(self) -> dict:
        return {"kind": self.kind, "lookback_months": self.lookback_months,
                "skip_months": self.skip_months, "top_n": self.top_n}


def momentum_mu(train: pd.DataFrame, lookback_months: int = 12, skip_months: int = 1) -> pd.Series:
    """Each asset's annualised return over the lookback window, as an expected return.

    The window ENDS `skip_months` before the cutoff and STARTS `lookback_months` before it, so
    the default 12/1 measures eleven months and stops a month short of the decision. Both edges
    are choices with a reason:

      the LOOKBACK is 12 months rather than the whole training window because that is the
      horizon over which relative performance persists. The five-year mean is what `"history"`
      already is, and it is the thing being replaced.

      the SKIP exists because the most recent month reverses. A one-month winner is
      disproportionately an asset that just gapped on news or a bid, and buying it earns the
      giveback rather than the trend. Dropping the skip is the single most common way a
      momentum backtest is quietly built to fail, and it fails by an amount large enough to
      look like the signal not working.

    Annualised over the RETURNS in the window (`n = bars - 1`) at 252 a year, the same
    convention `realized` and `frontier.estimate` use, so this number can sit in the same
    column as the ones they produce. Nothing after `train.index.max()` is read -- the whole
    file depends on that, and the window is built from the cutoff backwards for that reason.
    """
    cutoff = train.index.max()
    end = _bar_at_or_before(train, cutoff - pd.DateOffset(months=skip_months),
                            f"the bar {skip_months} month(s) before {cutoff.date()}")
    start = _bar_at_or_before(train, cutoff - pd.DateOffset(months=lookback_months),
                              f"the bar {lookback_months} month(s) before {cutoff.date()}")
    n = len(train.loc[start:end]) - 1
    if n < 20:
        raise ValueError(
            f"a {lookback_months}/{skip_months} momentum window spans {n} returns of this "
            f"training panel, too few to rank on -- the training window is too short"
        )
    growth = train.loc[end] / train.loc[start]
    if not (growth > 0).all():
        raise ValueError("a non-positive price relative in the momentum window")
    return growth ** (fetch.TRADING_DAYS_PER_YEAR / n) - 1.0


def estimates(train: pd.DataFrame, model: MuModel) -> tuple[fr.Estimates, pd.Series]:
    """The training-window estimates plus the momentum score, whichever model is in use.

    THE ONE PLACE `mu` IS CHOSEN. `est.mu` is substituted here and nowhere else, so everything
    downstream -- the solves, the predicted columns, `mu_rank_correlation`, the risk target --
    reads one field and cannot disagree about which forecast was tested.

    The momentum score is returned even under `"history"` because the `momentum_top` comparator
    needs it either way: a run that tests the optimiser on historical means should still print
    what a plain momentum basket did over the same periods, or there is nothing to attribute the
    difference to.
    """
    est = fr.estimate(train)
    mom = momentum_mu(train, model.lookback_months, model.skip_months).reindex(est.mu.index)
    if mom.isna().any():
        raise ValueError(f"no momentum score for {list(mom.index[mom.isna()])}")
    if model.kind == "momentum":
        est = replace(est, mu=mom)
    return est, mom


# ------------------------------------------------------------------------------- one cutoff

def predicted_vol(est: fr.Estimates, symbol: str) -> float:
    """One asset's annualised volatility AS THE TRAINING WINDOW ESTIMATED IT -- the diagonal of
    the same shrunk covariance the optimiser is given, not a separate recompute.

    This is the risk target, so it has to come from the training window. The realized volatility
    of the benchmark over the test window is the obviously "fairer" number and is a lookahead:
    nobody at the cutoff knows it. See `matched_risk` below.
    """
    i = list(est.mu.index).index(symbol)
    return float(np.sqrt(est.cov.to_numpy()[i, i]))


# How close to the target volatility counts as matched. `_clean` rounds every weight to 6dp
# after the solve, which moves the achieved volatility by a few parts in 1e6 -- so this cannot be
# exact. It is a RELATIVE floor rather than a two-sided band because the solver's constraint is
# one-sided: overshooting is not a thing `efficient_risk` can do, and undershooting by more than
# this means the target was out of reach.
RISK_MATCH_TOLERANCE = 1e-4


def reaches(w: np.ndarray, est: fr.Estimates, target: float) -> bool:
    """Did this portfolio actually get to the target volatility, or only as close as its
    constraints allowed? See the call site -- the difference is a mislabelled comparison."""
    vol = float(np.sqrt(w @ est.cov.to_numpy() @ w))
    return vol >= target * (1.0 - RISK_MATCH_TOLERANCE)


def strategies(est: fr.Estimates, caps: list[float], rf: float, benchmark: str | None,
               risk_target: float | None = None, momentum: pd.Series | None = None,
               mom_top: int = 10) -> list[tuple]:
    """(label, cap-or-None, weights) for everything evaluated at one cutoff.

    The comparators are not decoration. An optimiser that does not beat 1/N has not earned
    its covariance matrix, and one that does not beat the benchmark has not earned being
    used; both are computable from the same panel at no cost, so leaving them out would be a
    choice to not know.

    `momentum_top{N}` is the comparator that makes a momentum run interpretable. It is the
    signal held WITHOUT the optimiser -- equal weight in the N highest-scoring assets, no
    covariance, no solve -- and it separates the two things a momentum result confounds. If the
    optimised version beats it, the covariance matrix earned something; if it loses to it, the
    optimiser spent the signal on estimation error, and the answer "momentum does not work"
    would have been about the wrong component. Emitted under both `mu` models for the same
    reason: it is the same portfolio either way, so it is the fixed point the two runs share.

    `matched_risk` is the comparator the other two cannot be. The tangency portfolio holds far
    less risk than the benchmark -- about 7% annualised against SPY's 17% -- so "it returned less
    than SPY" is partly a statement about how much risk each one took, and the two columns are
    not comparable as they stand. This solves the frontier for the highest return available AT
    THE BENCHMARK'S OWN ESTIMATED VOLATILITY, so the answer is a long-only portfolio carrying the
    risk the benchmark carries, and the return columns can be read side by side. Omitted when the
    cap makes that risk unreachable, which is a real answer and not a failure.

    `risk_target` is PASSED IN rather than derived here, and `evaluate` is the only caller that
    supplies it. Deriving it in both places is how the levered rows and the long-only row would
    come to aim at two slightly different numbers, and -- the reason that matters -- it would give
    a lookahead two homes when the whole file is built so it has one. Defaults to the benchmark's
    own forecast volatility so a direct call still does the sane thing.
    """
    symbols = list(est.mu.index)
    if risk_target is None and benchmark in symbols:
        risk_target = predicted_vol(est, benchmark)
    # The target is used WHENEVER ONE WAS SUPPLIED, whether or not the benchmark is among the
    # candidates. On a stock universe it is not: an index fund sitting in the candidate list
    # would be bought by the minimum-variance solve, and "did this beat the index" would have
    # been asked of a portfolio allowed to BE the index. `evaluate` prices the benchmark and
    # keeps it out of the estimates, so `benchmark in symbols` is False there and the risk
    # target still has to reach `efficient_risk` -- gating on membership dropped `matched_risk`,
    # which is the one comparison the whole file exists to make.
    target = risk_target
    out: list[tuple] = []
    for cap in caps:
        w_tan = fr._solve(est.mu, est.cov, cap, "max_sharpe", risk_free_rate=rf)
        if w_tan is None:
            raise RuntimeError(f"max_sharpe infeasible at cap={cap}")
        out.append((f"tangency@{build.cap_slug(cap)}", cap, w_tan))
        w_mv = fr._solve(est.mu, est.cov, cap, "min_volatility")
        if w_mv is None:
            raise RuntimeError(f"min_volatility infeasible at cap={cap}")
        out.append((f"min_variance@{build.cap_slug(cap)}", cap, w_mv))
        if target is not None:
            w_mr = fr._solve(est.mu, est.cov, cap, "efficient_risk", target_volatility=target)
            # `efficient_risk` constrains volatility to be AT MOST the target, so an unreachable
            # target does not fail -- the constraint goes slack and it returns the highest-return
            # portfolio available, at whatever lower volatility that carries. Solved and feasible
            # is therefore not the same as matched, and a row labelled `matched_risk` sitting at
            # 12% when the target was 22% would be worse than no row at all: it is the exact
            # false comparison this strategy was added to remove. So the achieved volatility is
            # checked against the target, and the row is dropped when it falls short.
            if w_mr is not None and reaches(w_mr, est, target):
                out.append((f"matched_risk@{build.cap_slug(cap)}", cap, w_mr))

    n = len(symbols)
    out.append(("equal_weight", None, np.full(n, 1.0 / n)))
    if momentum is not None:
        # `cap` is None, as it is for the other two comparators: this is not a solve and the cap
        # is a constraint on the QP. The implied per-position weight is 1/N by construction, so
        # the only cap it could breach is one below 1/N, which no run here uses.
        k = min(mom_top, n)
        w = np.zeros(n)
        for sym in momentum.reindex(symbols).nlargest(k).index:
            w[symbols.index(sym)] = 1.0 / k
        out.append((f"momentum_top{k}", None, w))
    if benchmark in symbols:
        w = np.zeros(n)
        w[symbols.index(benchmark)] = 1.0
        out.append((f"benchmark:{benchmark}", None, w))
    return out


def restrict(est: fr.Estimates, keep: list[str]) -> fr.Estimates:
    """The same estimates over a SUBSET of the assets, with the shrinkage left as it was.

    Used to hold a benchmark out of the optimisation while still pricing it. The covariance is
    sliced rather than re-shrunk on purpose: re-estimating over the subset would change the
    shrinkage intensity and therefore every variance the solver sees, so the risk target (read
    off the full estimate's diagonal) and the risk the solver aims at would come from two
    different covariance matrices. One asset in or out moves the shrinkage target -- the average
    variance -- by about 1/N, and this way the numbers a benchmark row reports are the same ones
    the target was computed from.
    """
    missing = [s for s in keep if s not in est.mu.index]
    if missing:
        raise ValueError(f"cannot restrict to assets that were not estimated: {missing}")
    return fr.Estimates(mu=est.mu.loc[keep], cov=est.cov.loc[keep, keep],
                        shrinkage=est.shrinkage, n_obs=est.n_obs)


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    irx: pd.Series,
    caps: list[float],
    benchmark: str | None,
    rf_override: float | None,
    top_n: int,
    model: MuModel = MuModel(),
    investable: list[str] | None = None,
    costs: Costs = Costs(),
) -> dict:
    """Estimate on `train`, hold the answers across `test`, report both.

    THE ONE PLACE THE SPLIT IS ACTED ON, called by both the fixed-cutoff and the rolling
    driver below. They differ only in which bars they hand it -- the fixed one trains on
    everything up to the cutoff, the rolling one on a window of fixed length -- and that has
    to be their only difference, or "the same test at two window lengths" would be two tests.
    It is also why there is exactly one place a lookahead could be introduced.

    `train` must END on the cutoff and `test` must BEGIN on it. That shared bar is the
    position's entry price: the last return in training, and the zeroth (returnless) row of
    the value path.

    `investable` NARROWS WHAT MAY BE HELD WITHOUT NARROWING WHAT IS PRICED, and exists for one
    case: a universe of single stocks measured against an index fund. Estimation runs over the
    whole panel -- so the benchmark's forecast volatility, which is the risk target, comes off
    the same shrunk covariance as everything else -- and the solves run over the subset. Leave
    it None and every path below is the one the ETF universe takes, with the benchmark among the
    candidates as before.
    """
    cutoff = train.index.max()
    if test.index.min() != cutoff:
        raise ValueError(f"test starts at {test.index.min().date()}, not at the cutoff {cutoff.date()}")
    if len(train) < 250 or len(test) < 60:
        raise ValueError(f"split of {len(train)}/{len(test)} bars is too thin to measure")

    est, mom = estimates(train, model)
    # The estimates the SOLVER sees. Identical objects when nothing is held out, so the ETF
    # universe's numbers are unchanged rather than merely close.
    est_s = est if investable is None else restrict(est, investable)
    mom_s = mom if investable is None else mom.reindex(investable)
    # And the prices the value paths are read off, column-aligned with the weight vectors.
    # `value_path` takes `w` positionally against `prices`, so a restricted weight vector
    # against the full panel would silently price the wrong assets.
    test_s = test if investable is None else test[investable]
    rf_cut = rf_override if rf_override is not None else rf_asof(irx, cutoff)
    # The cash alternative over the TEST window, for the realized Sharpe. Using the cutoff
    # rate for both columns would charge the realized return a hurdle nobody faced: rates
    # moved by more than 200bp inside these windows.
    rf_test = float(irx.reindex(test.index).ffill().mean()) if rf_override is None else rf_override

    mu_test = fr.estimate(test).mu
    mu_test_s = mu_test.reindex(est_s.mu.index)
    vol_train = pd.Series(np.sqrt(np.diag(est_s.cov.to_numpy())), index=est_s.mu.index)
    vol_test = test_s.pct_change().dropna().std(ddof=1) * np.sqrt(fetch.TRADING_DAYS_PER_YEAR)

    # THE RISK TARGET, computed once here from the TRAINING window and passed to everything that
    # aims at it. The benchmark's realized volatility over `test` is the tempting number and is
    # not knowable at the cutoff; see `predicted_vol`.
    target_vol = predicted_vol(est, benchmark) if benchmark in est.mu.index else None

    # THE NET COLUMNS, added to a row rather than replacing its gross ones. Both readings in one
    # document is the point: "the optimiser wins before costs and loses after them" is a sentence
    # a reader can only check if the file carries both halves of it, and a file that carried only
    # the net numbers would make the size of the friction unrecoverable. Emitted only when
    # `costs` is non-zero, so every result published before this existed reproduces exactly.
    def net(v: pd.Series, prices: pd.DataFrame, w: np.ndarray, rebalance: str,
            extra_turnover: pd.Series | None = None, scale: float = 1.0) -> dict:
        t = scale * turnover_path(prices, w, rebalance)
        if extra_turnover is not None:
            t = t + extra_turnover
        er = scale * weighted_expense(prices.columns, w, costs)
        out = realized(v * cost_factor(t, er, costs), rf_test)
        years = max((len(v) - 1) / fetch.TRADING_DAYS_PER_YEAR, 1e-9)
        out["turnover_per_year"] = round(float(t.sum()) / years, 4)
        out["expense_ratio"] = round(er, 6)
        out["cost_drag_per_year"] = round(float(realized(v, rf_test)["ret"] - out["ret"]), 6)
        return out

    rows = []
    solved: dict[str, np.ndarray] = {}
    for label, cap, w in strategies(est_s, caps, rf_cut, benchmark, target_vol, mom_s, model.top_n):
        held = {s: round(float(x), 6) for s, x in zip(est_s.mu.index, w) if x > 0}
        solved[label] = w
        v_hold, v_month = value_path(test_s, w, "hold"), value_path(test_s, w, "monthly")
        row = {
            "strategy": label,
            "cap": cap,
            "n_holdings": len(held),
            "top_holdings": dict(sorted(held.items(), key=lambda kv: -kv[1])[:top_n]),
            "predicted": predicted(w, est_s, rf_cut),
            # What the model's own formula would have said with perfect foresight of `mu`:
            # the same linear combination, against the returns the assets actually had. It
            # separates "the estimate was wrong" from "the arithmetic does not carry over",
            # which the realized geometric return below conflates (variance drag is in it).
            "realized_linear_ret": round(float(w @ mu_test_s.to_numpy()), 6),
            "realized_hold": realized(v_hold, rf_test),
            "realized_monthly": realized(v_month, rf_test),
        }
        if not costs.zero():
            row["net_hold"] = net(v_hold, test_s, w, "hold")
            row["net_monthly"] = net(v_month, test_s, w, "monthly")
        rows.append(row)

    # THE BENCHMARK ROW, when the benchmark is priced but not investable. `strategies` cannot
    # emit it -- it builds weight vectors over the candidate set, and the benchmark is not in it
    # -- so it is built here against the FULL estimates, exactly as a one-hot over the candidate
    # set would have been. Appended before the levered rows so the row order is the one every
    # printer already expects: solves, comparators, benchmark, then leverage.
    # The condition is `not in est_s` -- the candidate set -- and NOT `investable is not None`.
    # An `investable` list that happens to contain the benchmark is the ETF case spelled out
    # longhand, `strategies` emits the row itself, and gating on `is not None` appended a SECOND
    # identical benchmark row. Every printer takes the first match, so the tables were right and
    # the JSON carried a duplicate;
    # `test_holding_nothing_out_is_the_same_run_as_not_holding_out_at_all` is what noticed.
    if benchmark is not None and benchmark in est.mu.index and benchmark not in est_s.mu.index:
        w_bm = np.zeros(len(est.mu))
        w_bm[list(est.mu.index).index(benchmark)] = 1.0
        bm_hold, bm_month = value_path(test, w_bm, "hold"), value_path(test, w_bm, "monthly")
        row = {
            "strategy": f"benchmark:{benchmark}",
            "cap": None,
            "n_holdings": 1,
            "top_holdings": {benchmark: 1.0},
            "predicted": predicted(w_bm, est, rf_cut),
            "realized_linear_ret": round(float(w_bm @ mu_test.to_numpy()), 6),
            "realized_hold": realized(bm_hold, rf_test),
            "realized_monthly": realized(bm_month, rf_test),
        }
        if not costs.zero():
            # A one-hot weight vector has zero rebalancing turnover by construction, so the index
            # investor's whole bill is the initial purchase plus the fund's annual fee. That
            # asymmetry against a monthly-rebalanced 400-stock portfolio is not an artefact of
            # the model -- it is the thing the model exists to put a number on.
            row["net_hold"] = net(bm_hold, test, w_bm, "hold")
            row["net_monthly"] = net(bm_month, test, w_bm, "monthly")
        rows.append(row)

    # The levered rows come AFTER the loop, not out of `strategies`, because leverage is not a
    # portfolio. The weights are the tangency portfolio's, unchanged; `k` is a financing decision
    # sitting on top of them. Emitting a weight vector that sums to `k` would silently break
    # every consumer that reads these rows as portfolios -- `n_holdings`, the cap check, and the
    # weight table all assume a convex combination.
    if target_vol is not None:
        financing = pd.Series([irx.asof(d) for d in test.index], index=test.index, dtype=float)
        if financing.isna().any() and rf_override is None:
            raise ValueError(f"no {build.RF_TICKER} quote at or before {test.index.min().date()}")
        if rf_override is not None:
            financing = pd.Series(rf_override, index=test.index, dtype=float)
        for row in [r for r in rows if r["strategy"].startswith("tangency@")]:
            w = solved[row["strategy"]]
            own_vol = row["predicted"]["vol"]
            if own_vol <= 0:
                continue
            # BOTH volatilities are the training window's estimates, so `k` is knowable at the
            # cutoff. Using either one's REALIZED volatility would size the position with
            # knowledge of the test window and is the lookahead this whole file guards against.
            k = target_vol / own_vol
            lev_hold = lever(value_path(test_s, w, "hold"), k, financing)
            lev_month = lever(value_path(test_s, w, "monthly"), k, financing)
            levered = {
                "strategy": row["strategy"].replace("tangency@", "tangency_levered@"),
                "cap": row["cap"],
                "n_holdings": row["n_holdings"],
                "top_holdings": row["top_holdings"],
                # Leverage scales the excess return and the volatility by the same factor, so
                # the predicted Sharpe is the tangency portfolio's and is unchanged by design --
                # that identity is the reason the comparison is fair.
                "leverage": round(float(k), 4),
                "risk_target": round(float(target_vol), 6),
                "predicted": {
                    "ret": round(rf_cut + k * (row["predicted"]["ret"] - rf_cut), 6),
                    "vol": round(k * own_vol, 6),
                    "sharpe": row["predicted"]["sharpe"],
                },
                "realized_linear_ret": round(float(rf_cut + k * (w @ mu_test_s.to_numpy() - rf_cut)), 6),
                "realized_hold": realized(lev_hold, rf_test),
                "realized_monthly": realized(lev_month, rf_test),
            }
            if not costs.zero():
                # `k` units of the fund, so `k` times its turnover and `k` times its fee -- plus
                # the daily mix reset, which is where a levered position's bill actually is.
                levered["net_hold"] = net(lev_hold, test_s, w, "hold",
                                          mix_turnover(value_path(test_s, w, "hold"), k, financing), k)
                levered["net_monthly"] = net(lev_month, test_s, w, "monthly",
                                             mix_turnover(value_path(test_s, w, "monthly"), k, financing), k)
            rows.append(levered)

    return {
        "cutoff": str(cutoff.date()),
        "train": {"start": str(train.index.min().date()), "bars": len(train),
                  "years": round((len(train) - 1) / fetch.TRADING_DAYS_PER_YEAR, 3),
                  "shrinkage": round(est.shrinkage, 6)},
        "test": {"end": str(test.index.max().date()), "bars": len(test),
                 "years": round((len(test) - 1) / fetch.TRADING_DAYS_PER_YEAR, 3)},
        "rf_at_cutoff": round(rf_cut, 6),
        "rf_mean_over_test": round(rf_test, 6),
        "mu_model": model.describe(),
        "costs": None if costs.zero() else costs.describe(),
        # Spearman, not Pearson: the question is whether the training window RANKS the assets
        # the test window rewards, and one 50%-vol outlier moves a Pearson correlation on 116
        # points far more than it moves the portfolio.
        #
        # THIS IS `est.mu`, SO IT SCORES WHICHEVER FORECAST WAS USED -- the five-year mean under
        # "history", the momentum score under "momentum". That is the number the two models have
        # to be compared on: the returns each one produced depend on the caps, the covariance and
        # the solver, and this depends on nothing but whether the forecast ranked the assets.
        "n_investable": len(est_s.mu),
        "mu_rank_correlation": round(float(stats.spearmanr(est_s.mu, mu_test_s).statistic), 4),
        # And the momentum score's own version of it, always, so a "history" run still measures
        # whether momentum would have ranked the cross-section -- independent of any portfolio.
        "momentum_rank_correlation": round(float(stats.spearmanr(mom_s, mu_test.reindex(mom_s.index)).statistic), 4),
        "vol_rank_correlation": round(float(stats.spearmanr(vol_train, vol_test.reindex(est_s.mu.index)).statistic), 4),
        "strategies": rows,
    }


def run_cutoff(
    panel: pd.DataFrame,
    years_back: int,
    irx: pd.Series,
    caps: list[float],
    benchmark: str | None,
    rf_override: float | None,
    top_n: int,
    model: MuModel = MuModel(),
    investable: list[str] | None = None,
    costs: Costs = Costs(),
) -> dict:
    """One cutoff, trained on EVERYTHING before it and held to the end of the panel."""
    wanted = panel.index.max() - pd.DateOffset(years=years_back)
    cutoff = _bar_at_or_before(panel, wanted, f"{years_back} years back")
    train = panel.loc[:cutoff]
    test = panel.loc[cutoff:]
    if len(train) < 500 or len(test) < 60:
        raise ValueError(f"{years_back}y back splits {len(train)}/{len(test)} bars, too thin")
    return {"years_back": years_back,
            **evaluate(train, test, irx, caps, benchmark, rf_override, top_n, model, investable, costs)}


def run_rolling(
    panel: pd.DataFrame,
    train_years: float,
    hold_years: float,
    span_years: float,
    irx: pd.Series,
    caps: list[float],
    benchmark: str | None,
    rf_override: float | None,
    top_n: int,
    model: MuModel = MuModel(),
    investable: list[str] | None = None,
    costs: Costs = Costs(),
) -> list[dict]:
    """A FIXED-LENGTH training window walked forward in NON-OVERLAPPING holding periods.

    The difference from `run_cutoff` is the whole point of having both. There, three cutoffs
    are three nested test windows sharing most of their returns and a training window that
    grows with each one -- so the rows cannot disagree with each other and there is really one
    observation. Here the holding periods are laid end to end and touch only at their edges,
    so `span_years / hold_years` of them are `span_years / hold_years` separate answers to
    "estimate on the last `train_years`, then hold for `hold_years`". Ten one-year periods can
    come out 6-4 and say something a single ten-year window cannot.

    What is still NOT independent, and no arrangement of windows can fix: consecutive training
    windows overlap by `train_years - hold_years`, so a regime that persists is estimated the
    same way several times in a row; and the universe is still 116 funds that exist today.

    The training window is FIXED-LENGTH on purpose rather than everything-so-far. At five
    years the covariance is estimated from ~1,260 daily returns for 116 assets, where the
    Ledoit-Wolf shrinkage actually does something (compare the ~0.006 the shipped 15-year
    estimate measures) -- and the expected returns are a five-year mean, which is what someone
    running this in 2016 would have had rather than a window that keeps growing behind them.
    """
    end = panel.index.max()
    n = int(round(span_years / hold_years))
    if n < 2:
        raise ValueError(f"a {span_years}y span in {hold_years}y steps is {n} period(s)")

    out = []
    for k in range(n, 0, -1):
        cutoff = _bar_at_or_before(panel, end - pd.DateOffset(years=hold_years * k),
                                  f"{hold_years * k} years back")
        train_from = _bar_at_or_before(panel, cutoff - pd.DateOffset(years=train_years),
                                      f"{train_years}y of history before {cutoff.date()}")
        # `loc[a:b]` is inclusive at both ends, so the training window carries the cutoff bar
        # and the test window starts on it -- the entry price, in both and counted in neither.
        train = panel.loc[train_from:cutoff]
        test_to = _bar_at_or_before(panel, cutoff + pd.DateOffset(years=hold_years),
                                   f"{hold_years}y after {cutoff.date()}")
        test = panel.loc[cutoff:test_to]
        row = evaluate(train, test, irx, caps, benchmark, rf_override, top_n, model, investable, costs)
        out.append({"period": f"{cutoff.date()} to {test_to.date()}", **row})
    return out


def run_window(
    panel: pd.DataFrame,
    train_years: float,
    cutoff_years_back: float,
    hold_years: list[float],
    irx: pd.Series,
    caps: list[float],
    benchmark: str | None,
    rf_override: float | None,
    top_n: int,
    model: MuModel = MuModel(),
    investable: list[str] | None = None,
    costs: Costs = Costs(),
) -> list[dict]:
    """ONE cutoff, a fixed-length training window behind it, and several hold lengths in front.

    This is the shape of the question "train on 2019-2023, then how did it do over the next 1,
    2 and 3 years" -- and it is the NESTED shape, which is why this docstring is longer than the
    function. The three rows share one training window, one weight vector per strategy, and one
    entry price; the 3-year row's returns CONTAIN the 2-year row's, which contain the 1-year
    row's. So the three are not three trials. They are one trial reported at three points, and
    if all three beat the benchmark that is one win, not three. `run_rolling` is what produces
    independent periods, and any claim about whether the method works belongs there.

    Kept anyway, and not as a trap this time: "how does one portfolio's lead over the index
    evolve as it is held" is a real question with a real answer, and it is the one an investor
    who actually did this is asking. It just cannot be turned into a batting average.

    The difference from `run_cutoff`: that one trains on EVERYTHING before the cutoff and holds
    to the last bar of the panel, so the hold length is whatever the panel happens to give. Here
    both ends are stated -- `train_years` behind the cutoff, `hold_years` in front of it -- so
    the training window does not silently grow when the panel does, and a 1-year hold is a year.
    """
    end = panel.index.max()
    cutoff = _bar_at_or_before(panel, end - pd.DateOffset(years=cutoff_years_back),
                               f"{cutoff_years_back} years back")
    train_from = _bar_at_or_before(panel, cutoff - pd.DateOffset(years=train_years),
                                   f"{train_years}y of history before {cutoff.date()}")
    train = panel.loc[train_from:cutoff]

    out = []
    for h in sorted(hold_years):
        wanted = cutoff + pd.DateOffset(years=h)
        if wanted > end:
            raise ValueError(
                f"a {h}y hold from {cutoff.date()} ends {wanted.date()}, past the panel's last "
                f"bar {end.date()} -- it would be reported as a {h}y hold and be shorter"
            )
        test_to = _bar_at_or_before(panel, wanted, f"{h}y after {cutoff.date()}")
        test = panel.loc[cutoff:test_to]
        row = evaluate(train, test, irx, caps, benchmark, rf_override, top_n, model, investable, costs)
        out.append({"hold_years": h, "period": f"{cutoff.date()} to {test_to.date()}", **row})
    return out


# ------------------------------------------------------------------------------- the report

def table(result: dict) -> str:
    """The printed summary. Monthly-rebalanced realized numbers, because that is the rule the
    SPA's growth curve uses; the JSON carries buy-and-hold beside it."""
    lines = []
    for c in result["cutoffs"]:
        lines.append("")
        lines.append(
            f"=== {c['years_back']}y back: trained {c['train']['start']} -> {c['cutoff']} "
            f"({c['train']['years']}y), held {c['cutoff']} -> {c['test']['end']} "
            f"({c['test']['years']}y) at rf {c['rf_at_cutoff']:.2%}"
        )
        lines.append(
            f"    cross-sectional rank correlation, training vs test:  "
            f"mu {c['mu_rank_correlation']:+.3f}   vol {c['vol_rank_correlation']:+.3f}"
        )
        lines.append(
            f"    {'strategy':<22}{'hold':>5}{'pred ret':>10}{'real ret':>10}"
            f"{'pred vol':>10}{'real vol':>10}{'pred SR':>9}{'real SR':>9}{'maxDD':>8}"
        )
        for s in c["strategies"]:
            r = s["realized_monthly"]
            lines.append(
                f"    {s['strategy']:<22}{s['n_holdings']:>5}"
                f"{s['predicted']['ret']:>9.2%}{r['ret']:>10.2%}"
                f"{s['predicted']['vol']:>10.2%}{r['vol']:>10.2%}"
                f"{s['predicted']['sharpe']:>9.2f}{(r['sharpe'] or float('nan')):>9.2f}"
                f"{r['max_drawdown']:>8.1%}"
            )
    return "\n".join(lines)


def _row(period: dict, label: str) -> dict | None:
    return next((s for s in period["strategies"] if s["strategy"] == label), None)


def _compound(rets: list[float]) -> float:
    """The annual rate that turns into what the periods actually produced, laid end to end.

    NEVER the arithmetic mean of the per-period returns, which is what these tables printed
    first. The two differ by roughly half the variance, so the mean rewards a strategy for being
    volatile -- and this file exists to compare strategies at different volatilities. A row that
    made +45% and -25% shows a 10% mean and compounds to 3.3%, and it is the 3.3% the holder had.
    """
    return float(np.prod([1.0 + r for r in rets]) ** (1.0 / len(rets)) - 1.0)


def _mu_phrase(mu_model: dict) -> str:
    """The expected-return model in words, for the table headers. Spelled out rather than
    printed as `{'kind': 'momentum', ...}` because the two runs produce tables that are
    otherwise identical in shape, and a reader comparing two printouts has to be able to see
    which is which without decoding a dict."""
    if mu_model["kind"] == "history":
        return "the average return over the whole training window"
    span = mu_model["lookback_months"] - mu_model["skip_months"]
    return (
        f"momentum: the return over {span} months, ending "
        f"{mu_model['skip_months']} month(s) before the start date"
    )


def _ew_label(periods: list[dict]) -> str:
    """"all 116 equal" -- the count read off the equal-weight row rather than written in. It was
    written in, and a universe of stocks then printed a column headed 116 over a basket of 480."""
    n = next((s["n_holdings"] for p in periods for s in p["strategies"]
              if s["strategy"] == "equal_weight"), None)
    return f"all {n} equal" if n else "equal weight"


def summary_table(block: dict) -> str:
    """EVERY strategy in the block on one page, so the comparison does not require flipping
    between three tables that each show a slice of it.

    `return per unit` is the column that decides anything. The rows run at volatilities between
    about 1% and 21%, so their compounded returns are not comparable as they stand -- two
    strategies with the same return per unit of risk are the same strategy at two sizes, and
    preferring the bigger one is not a finding. The absolute returns are printed beside it
    because that ratio hides the thing an investor actually experiences, which is the last
    column.

    `periods` is printed because it is not always all of them: `matched_risk` is dropped in any
    period where the cap put the benchmark's risk out of reach, and a row compounded over 7 of
    10 periods must not be read next to one compounded over 10.
    """
    periods = block["periods"]
    bm_label = f"benchmark:{block['benchmark']}"
    # Label order from the first period, which is the order `strategies` emits and therefore
    # groups the caps together. A set would sort `cap100` before `cap20` and split them up.
    order = [s["strategy"] for s in periods[0]["strategies"]]
    for p in periods[1:]:
        for s in p["strategies"]:
            if s["strategy"] not in order:
                order.append(s["strategy"])

    lines = ["", "=" * 100,
             f"EVERY STRATEGY SIDE BY SIDE, {block['hold_years']}-year holds, "
             f"{len(periods)} periods",
             f"    Expected returns from {_mu_phrase(block['mu_model'])}.",
             "",
             f"    {'strategy':<26}{'compounded':>12}{'risk it ran':>13}"
             f"{'return per unit':>17}{'worst drop':>12}"
             f"{'beat ' + (block['benchmark'] or '--'):>10}{'periods':>9}"]
    # Aligned to `periods` with a None where the benchmark is absent, NOT filtered -- a filtered
    # list would zip a strategy's period 3 against the benchmark's period 4 and every win count
    # below would be against the wrong year.
    bm_rets = [(_row(p, bm_label) or {}).get("realized_monthly", {}).get("ret") for p in periods]
    for label in order:
        rows = [_row(p, label) for p in periods]
        present = [(r, b) for r, b in zip(rows, bm_rets) if r]
        if not present:
            continue
        rets = [r["realized_monthly"]["ret"] for r, _ in present]
        risk = float(np.mean([r["realized_monthly"]["vol"] for r, _ in present]))
        drop = max(r["realized_monthly"]["max_drawdown"] for r, _ in present)
        comp = _compound(rets)
        # Only the periods where BOTH ran can be won or lost, so both halves of `beat S&P` come
        # from `paired` -- `len(present)` there would count a period the benchmark is missing from
        # as one this strategy lost.
        paired = [(r, b) for r, b in present if b is not None]
        wins = sum(1 for r, b in paired if r["realized_monthly"]["ret"] > b)
        ratio = f"{comp / risk:>17.2f}" if risk > 0 else f"{'--':>17}"
        lines.append(
            f"    {label:<26}{comp:>11.1%}{risk:>13.1%}{ratio}"
            f"{drop:>12.1%}{wins:>7} / {len(paired):<2}{len(present):>9}"
        )
    mu_mean = float(np.mean([p["mu_rank_correlation"] for p in periods]))
    mom_mean = float(np.mean([p["momentum_rank_correlation"] for p in periods]))
    vol_mean = float(np.mean([p["vol_rank_correlation"] for p in periods]))
    lines.append("")
    lines.append(
        "    Did the forecast RANK the assets the next period rewarded? (1.0 = perfectly, "
        "0.0 = coin flip)"
    )
    lines.append(f"        the return forecast actually used:  {mu_mean:+.2f}")
    lines.append(f"        the momentum score:                 {mom_mean:+.2f}")
    lines.append(f"        the risk forecast:                  {vol_mean:+.2f}")
    return "\n".join(lines)


def cost_table(block: dict) -> str:
    """Gross beside net, because the gap is the answer and neither column alone shows it.

    Returns "" when the run had no costs in it, so the caller can print it unconditionally.

    The benchmark row is the one to read first. It pays a fee and no turnover; every other row
    pays turnover and no fee. A reader who only sees net numbers cannot tell which side the
    friction fell on, and that is the entire question.
    """
    periods = block["periods"]
    if not any(s.get("net_monthly") for p in periods for s in p["strategies"]):
        return ""
    bm_label = f"benchmark:{block['benchmark']}"
    order = [s["strategy"] for s in periods[0]["strategies"]]
    for p in periods[1:]:
        for s in p["strategies"]:
            if s["strategy"] not in order:
                order.append(s["strategy"])
    c = next(p["costs"] for p in periods if p.get("costs"))
    lines = ["", "=" * 100,
             "AFTER COSTS -- both sides charged what they actually pay",
             f"    {c['trade_bps_per_dollar_traded']}bp per dollar traded ({c['turnover_convention']}); "
             f"annual fund fees {c['expense_ratios']}.",
             "    Not modelled: " + "; ".join(c["not_modelled"][:2]) + ".",
             "",
             f"    {'strategy':<26}{'gross':>9}{'net':>9}{'cost/yr':>9}"
             f"{'traded/yr':>11}{'fee':>7}{'net per unit risk':>19}"
             f"{'beat ' + (block['benchmark'] or '--'):>10}"]
    bm_net = [(_row(p, bm_label) or {}).get("net_monthly", {}).get("ret") for p in periods]
    for label in order:
        rows = [_row(p, label) for p in periods]
        present = [(r, b) for r, b in zip(rows, bm_net) if r and r.get("net_monthly")]
        if not present:
            continue
        gross = _compound([r["realized_monthly"]["ret"] for r, _ in present])
        net = _compound([r["net_monthly"]["ret"] for r, _ in present])
        risk = float(np.mean([r["net_monthly"]["vol"] for r, _ in present]))
        trade = float(np.mean([r["net_monthly"]["turnover_per_year"] for r, _ in present]))
        fee = float(np.mean([r["net_monthly"]["expense_ratio"] for r, _ in present]))
        paired = [(r, b) for r, b in present if b is not None]
        wins = sum(1 for r, b in paired if r["net_monthly"]["ret"] > b)
        ratio = f"{net / risk:>19.2f}" if risk > 0 else f"{'--':>19}"
        lines.append(
            f"    {label:<26}{gross:>9.1%}{net:>9.1%}{gross - net:>9.2%}"
            f"{trade:>11.2f}{fee:>7.3%}{ratio}{wins:>6} / {len(paired):<3}"
        )
    return "\n".join(lines)


def rolling_table(block: dict) -> str:
    """The rolling result, one table per weight cap, in words rather than abbreviations.

    No "SR" and no "vol" column: this table answers "did it work", and the useful form of that
    is what it FORECAST beside what it GOT, next to the two things it had to beat. The
    per-period Sharpe ratios and volatilities are all in the JSON for anything that needs them.

    The last column is the finding. It is the cross-sectional rank correlation between the
    training window's annualised returns and the test window's -- 1.0 means the optimiser
    ranked the 116 funds exactly as the next period ordered them, 0.0 means it ranked them no
    better than a coin flip. Printed per period because an average of it would hide that it
    swings sign.
    """
    lines = []
    periods = block["periods"]
    lines.append("")
    lines.append("=" * 100)
    lines.append(
        f"{block['train_years']} YEARS OF HISTORY -> HOLD {block['hold_years']} YEAR"
        f"{'S' if block['hold_years'] != 1 else ''}, walked forward over the last "
        f"{block['span_years']} years: {len(periods)} periods, laid end to end, none overlapping."
    )
    lines.append(
        "Each row is a portfolio chosen knowing ONLY the data before the start date, then held "
        "to the end date."
    )
    lines.append(f"Expected returns from {_mu_phrase(block['mu_model'])}.")
    for cap in block["caps"]:
        label = f"tangency@{build.cap_slug(cap)}"
        lines.append("")
        lines.append(f"--- most-efficient portfolio, no single fund above {cap:.0%} of it")
        lines.append(
            f"    {'holding period':<26}{'forecast':>10}{'ACTUAL':>10}{'worst drop':>12}"
            f"{_ew_label(periods):>15}{block['benchmark'] or '--':>10}{'ranking score':>15}"
        )
        wins_spy = wins_ew = 0
        got: list[float] = []
        for p in periods:
            s, ew, bm = _row(p, label), _row(p, "equal_weight"), _row(p, f"benchmark:{block['benchmark']}")
            a = s["realized_monthly"]["ret"]
            e = ew["realized_monthly"]["ret"]
            b = bm["realized_monthly"]["ret"] if bm else float("nan")
            got.append(a)
            wins_spy += a > b
            wins_ew += a > e
            lines.append(
                f"    {p['period']:<26}{s['predicted']['ret']:>9.1%}{a:>10.1%}"
                f"{s['realized_monthly']['max_drawdown']:>12.1%}{e:>15.1%}{b:>10.1%}"
                f"{p['mu_rank_correlation']:>+15.2f}"
            )
        n = len(periods)
        comp_spy = _compound([_row(p, f"benchmark:{block['benchmark']}")["realized_monthly"]["ret"]
                              for p in periods])
        comp_ew = _compound([_row(p, "equal_weight")["realized_monthly"]["ret"] for p in periods])
        lines.append(f"    {'-' * 94}")
        lines.append(
            f"    {'compounded over all of it':<26}{'':>10}{_compound(got):>10.1%}{'':>12}"
            f"{comp_ew:>15.1%}{comp_spy:>10.1%}"
        )
        lines.append(
            f"    beat {block['benchmark']} in {wins_spy} of {n} periods; beat the equal-weight "
            f"basket in {wins_ew} of {n}"
        )
        lines.append(
            f"    worst period {min(got):.1%}, best period {max(got):.1%}"
        )
    return "\n".join(lines)


def matched_risk_table(block: dict) -> str:
    """Apples to apples: what each way of carrying the BENCHMARK'S risk actually returned.

    The table above compares an ~7%-volatility portfolio against a ~17%-volatility index and the
    return columns are therefore not comparable -- most of the gap is a difference in how much
    risk was taken, not in how well anything was chosen. This one removes that.

    Two ways to reach the target, because each answers the objection to the other:

      "long-only"  -- solve the frontier for the most return available AT the benchmark's own
                      forecast volatility. No borrowing, no financing assumption, and the answer
                      is a portfolio you could hold as-is. Missing when the weight cap makes that
                      much risk unreachable, which at a 10% cap it does.
      "borrowed"   -- the tangency portfolio scaled up the capital market line to the same target,
                      financed at the T-bill rate as it moved. Theoretically the right answer and
                      practically the optimistic one: nobody borrows at the bill rate, and the
                      exposure is reset daily for free.

    The target is the benchmark's FORECAST volatility from each training window, not its realized
    volatility over the test window -- that number is not knowable at the cutoff. So the risk
    columns are printed too: aiming at 17% and landing at 25% is the forecast being wrong, and it
    is the thing this table would otherwise hide.
    """
    periods = block["periods"]
    bm_label = f"benchmark:{block['benchmark']}"
    lines = ["", "=" * 100,
             f"SAME RISK AS THE {block['benchmark']}: three ways to carry it, "
             f"{block['hold_years']}-year holds"]
    lines.append(
        "    Each period aims at the risk the training window forecast for the benchmark, so the "
        "return columns compare."
    )
    for cap in block["caps"]:
        mr, lv = f"matched_risk@{build.cap_slug(cap)}", f"tangency_levered@{build.cap_slug(cap)}"
        if not any(_row(p, mr) or _row(p, lv) for p in periods):
            continue
        lines.append("")
        lines.append(f"--- no single fund above {cap:.0%} of the portfolio")
        lines.append(
            f"    {'holding period':<26}{'aimed at':>10}{'long-only':>11}{'its risk':>10}"
            f"{'borrowed':>10}{'its risk':>10}{block['benchmark']:>10}{'its risk':>10}"
            f"{'borrowing':>11}"
        )
        # ALIGNED TO `periods`, with a None wherever a strategy has no row -- never compacted.
        # `matched_risk` is dropped in any period whose cap could not reach the target, so a
        # compacted list pairs one strategy's period 4 against the benchmark's period 3 and every
        # win count below is then against the wrong year, silently and plausibly.
        cols: dict[str, list[float | None]] = {"mr": [], "lv": [], "bm": []}
        for p in periods:
            got = {"mr": _row(p, mr), "lv": _row(p, lv), "bm": _row(p, bm_label)}
            # The target is the BENCHMARK's forecast volatility, so the benchmark row carries it
            # even in a period where both risk-matched strategies were dropped -- which is the
            # period whose target the reader most wants to see, since the cap could not reach it.
            ref = got["lv"] or got["mr"] or got["bm"]
            aim = ref.get("risk_target", ref["predicted"]["vol"]) if ref else None
            for k, row in got.items():
                cols[k].append(row["realized_monthly"]["ret"] if row else None)
            def cell(k: str, field: str, width: int) -> str:
                row = got[k]
                return f"{row['realized_monthly'][field]:>{width}.1%}" if row else f"{'--':>{width}}"
            lev = f"{got['lv']['leverage']:>10.2f}x" if got["lv"] else f"{'--':>11}"
            lines.append(
                f"    {p['period']:<26}{f'{aim:.1%}' if aim is not None else '--':>9}"
                f"{cell('mr', 'ret', 11)}{cell('mr', 'vol', 10)}"
                f"{cell('lv', 'ret', 10)}{cell('lv', 'vol', 10)}"
                f"{cell('bm', 'ret', 10)}{cell('bm', 'vol', 10)}{lev}"
            )
        lines.append(f"    {'-' * 96}")
        lines.append(
            f"    {'':<26}{'compounded':>12}{'risk it ran':>13}{'return per unit':>17}"
            f"{'worst drop':>12}"
        )
        for k, name in (("mr", "long-only at that risk"), ("lv", "borrowed to that risk"),
                        ("bm", f"just holding {block['benchmark']}")):
            if not any(v is not None for v in cols[k]):
                continue
            # PRESENT PERIODS ONLY. `matched_risk` is dropped wherever the cap put the target out
            # of reach, so this list has holes -- and `cols[k]` being non-empty does not mean it
            # is full. Indexing straight into `_row(...)` here crashed the moment a run produced
            # a cap that reached the target in some periods and not others.
            rows_k = [r for r in (_row(p, {"mr": mr, "lv": lv, "bm": bm_label}[k])
                                  for p in periods) if r]
            comp = _compound([v for v in cols[k] if v is not None])
            risk = float(np.mean([r["realized_monthly"]["vol"] for r in rows_k]))
            drop = max(r["realized_monthly"]["max_drawdown"] for r in rows_k)
            # The column that decides it. Two strategies at the same return per unit of risk are
            # the same strategy at two sizes, and picking the bigger one is not skill.
            lines.append(
                f"    {name:<26}{comp:>11.1%}{risk:>13.1%}{comp / risk:>17.2f}{drop:>12.1%}"
                f"{'' if len(rows_k) == len(periods) else f'   ({len(rows_k)} of {len(periods)} periods)'}"
            )
        for k, name in (("mr", "long-only"), ("lv", "borrowed")):
            both = [(a, b) for a, b in zip(cols[k], cols["bm"]) if a is not None and b is not None]
            if both:
                wins = sum(a > b for a, b in both)
                lines.append(f"    {name} beat it in {wins} of {len(both)} periods")
    return "\n".join(lines)


def window_table(block: dict) -> str:
    """One training window, one weight vector per strategy, held for several lengths.

    Deliberately NOT laid out like `rolling_table`, whose rows are separate periods that can be
    counted. These rows are the SAME portfolio measured at three points, so there is no win count
    and no compounded-over-all-of-it line: compounding nested returns would multiply the first
    year in three times over. What the table can say is how the lead evolved, and whether it
    survived being held.

    Every row is annualised, which is the only way three different hold lengths sit in one column.
    The risk-matched pair is printed beside the raw pair for the usual reason -- a 7%-volatility
    portfolio beating a 17%-volatility index is mostly a statement about position size.
    """
    periods = block["periods"]
    bm_label = f"benchmark:{block['benchmark']}"
    bm_head = block["benchmark"] or "--"
    lines = ["", "=" * 100,
             f"ONE WINDOW: trained on {block['train_years']} years ending "
             f"{block['cutoff_years_back']} years back, then held forward",
             f"    {periods[0]['train']['start']} -> {periods[0]['cutoff']} chooses the weights; "
             f"every row below holds THOSE weights, just for longer.",
             f"    Expected returns from {_mu_phrase(block['mu_model'])}.",
             "    NESTED: the 3-year row contains the 2-year row. Three end dates, one trial -- "
             "see run_rolling for periods that count.",
             "    All returns annualised."]
    for cap in block["caps"]:
        tg, mr = f"tangency@{build.cap_slug(cap)}", f"matched_risk@{build.cap_slug(cap)}"
        lines.append("")
        lines.append(f"--- no single holding above {cap:.0%} of the portfolio")
        lines.append(
            f"    {'held':<10}{'to':<13}{'tangency':>10}{'its risk':>10}"
            f"{'at matched risk':>17}{'its risk':>10}"
            f"{bm_head:>10}{'its risk':>10}{'equal wt':>10}"
        )
        for p in periods:
            got = {"tg": _row(p, tg), "mr": _row(p, mr), "bm": _row(p, bm_label),
                   "ew": _row(p, "equal_weight")}

            def cell(k: str, field: str, width: int) -> str:
                row = got[k]
                return f"{row['realized_monthly'][field]:>{width}.1%}" if row else f"{'--':>{width}}"

            lines.append(
                f"    {str(p['hold_years']) + 'y':<10}{p['test']['end']:<13}"
                f"{cell('tg', 'ret', 10)}{cell('tg', 'vol', 10)}"
                f"{cell('mr', 'ret', 17)}{cell('mr', 'vol', 10)}"
                f"{cell('bm', 'ret', 10)}{cell('bm', 'vol', 10)}{cell('ew', 'ret', 10)}"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="1,2,3", help="cutoffs, in years back from the panel end")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    ap.add_argument("--universe", default=uni.DEFAULT)
    ap.add_argument("--start", default="2011-01-03")
    ap.add_argument("--caps", default="1.0,0.2,0.1")
    ap.add_argument("--min-coverage", type=float, default=0.98)
    ap.add_argument("--top", type=int, default=8, help="holdings listed per strategy in the JSON")
    ap.add_argument("--rf", default="auto", help="'auto' (^IRX at each cutoff) or a decimal")
    ap.add_argument("--refresh-rf", action="store_true", help="refetch ^IRX rather than use the cache")
    ap.add_argument("--train-years", type=float, default=5.0,
                    help="length of the rolling training window")
    ap.add_argument("--hold-years", default="1,2",
                    help="holding-period lengths for the rolling test, comma-separated")
    ap.add_argument("--span-years", type=float, default=10.0,
                    help="how far back the rolling test starts")
    ap.add_argument("--no-rolling", action="store_true", help="fixed cutoffs only")
    ap.add_argument("--benchmark", default=None,
                    help="a benchmark symbol to price WITHOUT letting the optimiser hold it; "
                         "overrides the universe's own. Use for a universe of single stocks, "
                         "where an index fund among the candidates would be bought by the "
                         "min-variance solve and 'did this beat the index' would then be asked "
                         "of a portfolio allowed to BE the index.")
    ap.add_argument("--window", action="store_true",
                    help="one cutoff, a fixed training window behind it, several holds in front "
                         "(--cutoff-years-back, --window-holds). NESTED: one trial reported at "
                         "several points, not several trials.")
    ap.add_argument("--cutoff-years-back", type=float, default=3.0,
                    help="where --window's single cutoff sits, in years back from the panel end")
    ap.add_argument("--window-holds", default="1,2,3",
                    help="--window's hold lengths in years, comma-separated")
    ap.add_argument("--mu", default="history", choices=MU_MODELS,
                    help="the expected-return input the optimiser is given")
    ap.add_argument("--mom-lookback", type=int, default=12,
                    help="momentum measurement window, in months back from the cutoff")
    ap.add_argument("--mom-skip", type=int, default=1,
                    help="months before the cutoff the momentum window stops short of")
    ap.add_argument("--extra-panel", type=Path, default=None,
                    help="a parquet of second-vendor total-return columns from delisted.py, "
                         "joined onto the panel's own bars as extra candidates")
    ap.add_argument("--mom-top", type=int, default=10,
                    help="holdings in the unoptimised equal-weight momentum comparator")
    ap.add_argument("--trade-bps", type=float, default=0.0,
                    help="all-in one-way cost per dollar traded, in basis points. Turns on the "
                         f"net-of-cost columns; {DEFAULT_TRADE_BPS} is a reasonable large-cap US "
                         "figure. 0 (the default) reproduces a run made before this existed, bit "
                         "for bit. The benchmark's published fund fee is charged whenever this is "
                         "non-zero, so the two sides are charged together or not at all.")
    args = ap.parse_args(argv)

    costs = Costs(trade_bps=args.trade_bps,
                  expense=EXPENSE_RATIOS if args.trade_bps else {})
    model = MuModel(kind=args.mu, lookback_months=args.mom_lookback,
                    skip_months=args.mom_skip, top_n=args.mom_top)
    years = [int(y) for y in args.years.split(",")]
    caps = [float(c) for c in args.caps.split(",")]
    build.cap_slugs(caps)
    u = uni.load(args.universe)

    # `--benchmark` is PRICED, NOT INVESTABLE: it joins the panel so its returns can be measured
    # and so `matched_risk` has a volatility to aim at, but it is held out of `investable` and the
    # optimiser never sees it. `wanted` is therefore what has to be on disk; `u.symbols` is what
    # may be held.
    benchmark = args.benchmark or u.benchmark
    held_out = bool(benchmark) and benchmark not in u.symbols
    wanted = list(u.symbols) + ([benchmark] if held_out else [])

    on_disk = store.stored_symbols(args.price_dir)
    missing = [s for s in wanted if s not in on_disk]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(wanted)} symbols are not in {args.price_dir} "
            f"(e.g. {missing[:5]}). Run `python -u pipeline/build.py` first -- this script "
            "never fetches prices, so that a backtest cannot silently move its own window."
        )

    # THE SAME ASSETS THE PAGE SHIPS, filtered over the FULL window and then truncated.
    # Re-running the coverage filter inside each training window would quietly change the
    # universe per cutoff, and the rows would no longer be comparable with each other
    # or with the site.
    panel, dropped = fetch.load_panel(
        wanted, args.price_dir, start=args.start, min_coverage=args.min_coverage
    )
    # Restored delisted constituents, if any, joined onto the panel's OWN bars -- so this cannot
    # move the window and the two runs are comparable. `None` when the flag is absent, which is
    # bit-identical to not having the feature.
    extra_info = None
    if args.extra_panel is not None:
        panel, extra_info = join_extra(panel, args.extra_panel, args.start, args.min_coverage)
        print(f"joined {len(extra_info['added'])} second-vendor columns from "
              f"{args.extra_panel.name}: {', '.join(extra_info['added'])}"
              + (f"; dropped {extra_info['dropped']}" if extra_info["dropped"] else ""),
              flush=True)
    # The survivors, in the panel's own order, minus the held-out benchmark. `None` when nothing
    # is held out, which is the ETF path and is bit-identical to not having this feature.
    investable = [s for s in panel.columns if s != benchmark] if held_out else None
    if held_out and benchmark not in panel.columns:
        raise SystemExit(
            f"benchmark {benchmark} did not survive the common-window filter, so there is nothing "
            f"to compare against. It needs quotes over the whole panel; {len(dropped)} symbols "
            "were dropped and it was one of them."
        )
    # `len(wanted)` counts what the STORE was asked for, so the extras are added to the
    # denominator too: 498/499 is the honest fraction, 498/487 reads as more than was asked for.
    asked = len(wanted) + (len(extra_info["added"]) + len(extra_info["dropped"])
                           if extra_info else 0)
    print(
        f"universe {u.key}: {panel.shape[1]}/{asked} assets x {len(panel)} bars "
        f"({panel.index.min().date()} -> {panel.index.max().date()}), {len(dropped)} dropped"
        + (f"; benchmark {benchmark} priced but not investable, "
           f"{len(investable)} candidates" if held_out else ""),
        flush=True,
    )

    irx = load_irx(refresh=args.refresh_rf)
    rf_override = None if args.rf == "auto" else float(args.rf)

    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "universe": {"key": u.key, "n_assets": panel.shape[1], "benchmark": benchmark,
                     "benchmark_investable": not held_out,
                     "n_investable": panel.shape[1] - (1 if held_out else 0),
                     "extra_panel": extra_info},
        "panel": {"start": str(panel.index.min().date()), "end": str(panel.index.max().date()),
                  "bars": len(panel)},
        "method": {
            "estimator": "geometric mean + Ledoit-Wolf shrunk covariance, 252-day annualised",
            "expected_returns": _mu_phrase(model.describe()),
            "mu_model": model.describe(),
            "tangency": "frontier._solve max_sharpe at the cutoff's ^IRX yield",
            "rf": "^IRX as of each cutoff" if rf_override is None else f"fixed {rf_override}",
            "caps": caps,
            "rebalancing": ["hold", "monthly"],
            "caveats": [
                "the test windows are nested, so the cutoffs are not independent trials",
                f"the universe is {panel.shape[1]} instruments that exist today with quotes back "
                "to the start date, so every asset tested survived the period",
                "no transaction costs, no taxes, no slippage" if costs.zero() else
                f"trading charged at {args.trade_bps}bp per dollar traded and the benchmark "
                f"charged its published fund fee; both readings ship (realized_* is gross, "
                f"net_* is after costs). No taxes.",
            ] + ([
                f"{benchmark} is priced but not investable: it is in the panel so its returns and "
                "its forecast volatility can be measured, and held out of the candidate set so "
                "the optimiser cannot buy the thing it is being compared against",
            ] if held_out else []) + ([
                f"{len(extra_info['added'])} constituents the price store cannot serve at all "
                f"were restored from a SECOND vendor via {extra_info['file']}, priced to their "
                "last-but-one traded bar and then continued under a stated assumption -- see "
                "pipeline/data/delisted/provenance_*.json. This removes most of the "
                "survivorship bias and replaces it with that assumption, which is why the run "
                "exists in more than one version",
            ] if extra_info else []) + ([
                # The one caveat that is specific to the momentum run, and it is the big one:
                # a signal is only tested at the frequency it is refreshed.
                f"the weights are chosen ONCE per holding period, so a {args.hold_years}-year "
                "hold tests momentum measured annually -- the effect is documented at monthly "
                "refresh and decays over months, so this understates it by an unknown amount",
                "momentum turns the portfolio over far faster than the historical mean does, "
                "and 'no transaction costs' above is therefore a larger subsidy here",
            ] if model.kind == "momentum" else []),
        },
        "costs": None if costs.zero() else costs.describe(),
        "cutoffs": [run_cutoff(panel, y, irx, caps, benchmark, rf_override, args.top, model,
                               investable, costs)
                    for y in sorted(years)],
    }

    if args.window:
        holds = [float(x) if "." in x else int(x) for x in args.window_holds.split(",")]
        result["method"]["window"] = (
            f"{args.train_years}y training window ending {args.cutoff_years_back}y back, held "
            f"forward {', '.join(str(h) for h in holds)} years -- NESTED, so one trial reported "
            "at several points rather than several trials"
        )
        result["window"] = {
            "train_years": args.train_years,
            "cutoff_years_back": args.cutoff_years_back,
            "hold_years": holds,
            "caps": caps,
            "benchmark": benchmark,
            "mu_model": model.describe(),
            "periods": run_window(panel, args.train_years, args.cutoff_years_back, holds, irx,
                                  caps, benchmark, rf_override, args.top, model, investable,
                                  costs),
        }

    if not args.no_rolling:
        result["method"]["rolling"] = (
            f"{args.train_years}y training window walked forward in non-overlapping holding "
            f"periods over the last {args.span_years}y"
        )
        result["rolling"] = [
            {
                "train_years": args.train_years,
                "hold_years": h,
                "span_years": args.span_years,
                "caps": caps,
                # `benchmark`, NOT `u.benchmark`. Every rolling printer builds its row lookup key
                # from this field (`f"benchmark:{block['benchmark']}"`), so recording the universe's
                # own benchmark while `run_rolling` measured the `--benchmark` override makes the
                # key miss: on a universe declaring `benchmark = ""` the key becomes
                # "benchmark:None" and matches nothing. `run_rolling` is passed `benchmark` two
                # lines down, which is the point -- the label and the measurement have to be the
                # same symbol, and the window block below already was.
                "benchmark": benchmark,
                "mu_model": model.describe(),
                "periods": run_rolling(panel, args.train_years, h, args.span_years, irx, caps,
                                       benchmark, rf_override, args.top, model, investable,
                                       costs),
            }
            for h in [float(x) if "." in x else int(x) for x in args.hold_years.split(",")]
        ]

    print(table(result), flush=True)
    if "window" in result:
        print(window_table(result["window"]), flush=True)
    for block in result.get("rolling", []):
        print(rolling_table(block), flush=True)
        print(matched_risk_table(block), flush=True)
        print(summary_table(block), flush=True)
        if not costs.zero():
            print(cost_table(block), flush=True)
    n = build.write_json(args.out, result)
    # `os.path.relpath`, not `Path.relative_to`: the latter RAISES on a relative --out, which is
    # the ordinary way to type one. The whole run then exits 1 after the file is safely on disk.
    print(f"\nwrote {os.path.relpath(args.out.resolve(), ROOT)} ({n / 1024:.1f} KiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
