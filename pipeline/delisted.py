"""Prices for the constituents the primary vendor will not serve, from a SECOND vendor.

WHY THIS FILE IS NOT PART OF THE STORE, AND MUST NEVER BECOME PART OF IT
-----------------------------------------------------------------------
`store.py`'s contract is one vendor, `close` plus an event log, with the reconstructed total
return checked against *that vendor's own* `adjclose` to ~1e-6. Every property the store
asserts rests on those three things arriving together. This vendor ships an adjusted level
and no event log, so there is nothing to reconstruct and nothing to check it against. Writing
it into `data/prices/` would leave the store's tests passing on symbols they cannot actually
test, which is worse than not having the prices at all.

So it lives in `data/delisted/`, a different directory with a different shape (one CSV per
symbol: date, close, adjusted, volume), and it reaches the optimiser only through
`backtest.py --extra-panel`, which records the provenance in the result JSON. Nothing here
ships to the browser.

WHY IT EXISTS
-------------
`sp500_2023.toml` excludes 17 of the 503 constituents because the primary vendor answers 404
for them: it serves *nothing* for a ticker that stopped trading rather than a series that
ends. The bias that leaves has a known direction -- takeovers are done at a premium, so the
unpriceable names are disproportionately winners -- but "bounded at +0.18 to +0.47 pp/yr on an
equal-weighted basket" is an argument, not a measurement. This file replaces the argument with
the prices, for the 12 of the 17 a second vendor does serve.

THE GATE, WHICH IS A RATIO TEST AND NOT A LEVEL TEST
---------------------------------------------------
Two total-return series for the same asset agree up to one arbitrary scale factor, so the
question is whether `second_vendor_t / store_total_return_t` is CONSTANT over the overlap, not
whether the two levels match. `crosscheck` reports the max deviation of that ratio from its own
median, over symbols both vendors cover: <=7.4e-4 on nine of ten controls, which is the
precision at which this is worth using.

The tenth is the finding worth keeping. **T (AT&T) deviates 1.0e-1, and it is not noise: it is
the 2022 WarnerMedia/Discovery spinoff.** A distribution in kind is not a cash dividend, so the
store (following the primary vendor's `adjclose`) does not credit it and this vendor does. The
two are answering different questions and this vendor is arguably the more correct one, but the
consequence for us is narrower: a symbol whose history contains a spinoff cannot be compared
across the two sources at 1e-3, so `SPINOFF_CONTROLS` is excluded from the threshold and named
rather than absorbed by loosening it. K (Kellanova, the WK Kellogg spinoff of 2023) is in the
delisted set and carries the same ambiguity in the same direction; it is recorded in the
provenance as a caveat, because there is no second series to resolve it against.

TWO THINGS ABOUT THESE SERIES THAT ARE NOT TRUE OF THE STORE'S
--------------------------------------------------------------
1. **`range=MAX` silently returns one year -- USUALLY.** The endpoint does not reject an
   unrecognised range; it serves 252 rows and a 200 response. `20Y`, `15Y`, `30Y` and `All` do
   that for every symbol tried, and `MAX` did it for every symbol tried except one, which it
   served in full three times and then stopped serving in full minutes later. So a range string
   is a request, the answer to it is not stable, and every fetch ASSERTS what it was actually
   served -- either against the date the caller needs (`needs`) or against a length
   (`min_span_years`). `fetch_longest` is the consequence: try several ranges, keep the longest.

2. **The final traded bar is dropped, uniformly, and the reason is measured.** The last bar of
   a delisted series is an index-deletion closing auction or a stub, not a normal market close:
   three of the twelve print volume 0 or 1, and six print 5-20x their trailing volume. Where an
   independent check exists it fails. CMA traded at 1.89542 x FITB (its all-stock acquirer,
   which the PRIMARY vendor serves) to within +-0.10% for eleven consecutive bars and then
   printed its last bar 5.34% below that ratio; IPG traded at 0.35386 x OMC to within +-0.21%
   *including* its last bar. So the bad final bar is real, it is not universal, and there is no
   per-name test for it in the ten cases with no listed acquirer to check against. Dropping the
   last bar of every series is the one rule that needs no case list: it is exactly right for a
   cash deal (the price is pinned to the consideration for days beforehand -- ATVI, CTLT, JNPR,
   K, SEE move by <=0.01 between the last two bars), and it costs the 2.0% IPG's clean auction
   print was worth. `provenance.json` records the per-symbol size of the choice.

WHAT HAPPENS AFTER THE TERMINAL BAR IS AN ASSUMPTION, SO THERE ARE TWO OF THEM
-----------------------------------------------------------------------------
A holder of a delisted name received something -- cash, or acquirer shares -- and then held
whatever they did with it. There is no fact of the matter, so the series is continued to the
end of the panel under two stated treatments and the backtest is run under both:

    cash  -- proceeds sit in T-bills at ^IRX as it moved (the pessimistic bound)
    spy   -- proceeds are reinvested in the index that day (the market bound)

The point of running both is not to average them. It is that a conclusion which holds under
both does not depend on the treatment, and one that flips between them was never a conclusion.
Three of the twelve were all-stock deals whose acquirer is in the panel already (ANSS/SNPS,
CMA/FITB, IPG/OMC) and could be continued exactly; that is not done here, because a per-name
continuation list is a place for a mistake to hide and the two bounds are what the argument
needs. It is a stated limitation, not an oversight.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch  # noqa: E402
import store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "data" / "delisted"
PRICE_DIR = ROOT / "pipeline" / "data" / "prices"

SOURCE = "stockanalysis.com"
API = "https://stockanalysis.com/api/symbol/s/{sym}/history?range={rng}&period=Daily"
# The endpoint answers 403 without one. Not a scrape of a rendered page -- this is the same
# JSON the site's own charts read.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

# The 12 of `sp500_2023.toml`'s 17 exclusions this vendor serves, and the ticker it serves each
# under. A ticker CHANGE is a continuation of the same company's series and is spelled out; the
# other eleven are served under the constituent's own ticker.
SERVED_AS = {
    "ANSS": "ANSS", "ATVI": "ATVI", "CDAY": "DAY", "CMA": "CMA", "CTLT": "CTLT",
    "CTRA": "CTRA", "HOLX": "HOLX", "IPG": "IPG", "JNPR": "JNPR", "K": "K",
    "SEE": "SEE", "WBA": "WBA",
}

# The five neither vendor serves at any date, so no treatment can price them. Not a list this
# module acts on -- it is here so the count in the provenance file is derived rather than typed.
UNPRICEABLE = ["DFS", "HES", "MRO", "PXD", "WRK"]

# Both vendors cover these, so the ratio gate can run on them.
CONTROLS = ["SPY", "KO", "XOM", "JNJ", "PG", "AAPL", "NVDA", "BRO", "MO"]
# ...and this one, which cannot be compared at 1e-3 and is reported outside the threshold.
# See the module docstring: a spinoff is not a cash dividend and the two vendors disagree.
SPINOFF_CONTROLS = ["T"]
GATE = 5e-3

CONTINUATIONS = ("cash", "spy")

# Range strings to try. Longest-first in intent, and NOT in guaranteed effect -- which is the
# whole reason this is a list and not a single value. See `fetch_longest`.
RANGES = ("MAX", "10Y")


# ------------------------------------------------------------------------------ the vendor

def _get(symbol: str, rng: str = "10Y", attempts: int = 4) -> list[dict]:
    url = API.format(sym=symbol.lower(), rng=rng)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if i == attempts - 1:
                raise RuntimeError(f"{symbol}: {SOURCE} gave up after {attempts}: {e}") from e
            time.sleep(1.5 * (i + 1))
    rows = payload.get("data")
    if not rows:
        raise RuntimeError(f"{symbol}: {SOURCE} served no rows for range={rng}")
    return rows


def _parse(symbol: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(
        {"close": [float(r["c"]) for r in rows],
         "adjusted": [float(r["a"]) for r in rows],
         "volume": [float(r["v"]) for r in rows]},
        index=pd.Index([pd.Timestamp(r["t"]) for r in rows], name="date"),
    ).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if not (df["adjusted"] > 0).all():
        raise RuntimeError(f"{symbol}: {(df['adjusted'] <= 0).sum()} non-positive adjusted closes")
    return df


def _span_years(df: pd.DataFrame) -> float:
    return (df.index.max() - df.index.min()).days / 365.25


def _check_served(symbol: str, df: pd.DataFrame, rng: str, needs: str | None,
                  min_span_years: float | None) -> pd.DataFrame:
    """Raise unless what the vendor served is what the caller asked for.

    THE FAILURE THIS EXISTS FOR: a range string is a request, and an unrecognised one returns one
    year of bars with a 200 rather than an error. So neither form of the check is optional and
    exactly one of them must be given -- a call with neither checks nothing.
    """
    if (needs is None) == (min_span_years is None):
        raise ValueError("give exactly one of `needs` and `min_span_years` -- a call with "
                         "neither checks nothing, which is what this guard is for")
    served = _span_years(df)
    why = (f"which does not reach {needs}" if needs is not None else
           f"a span of {served:.1f}y and not the {min_span_years}y asked for")
    if (needs is not None and df.index.min() > pd.Timestamp(needs)) or \
       (min_span_years is not None and served < min_span_years):
        raise RuntimeError(
            f"{symbol}: {SOURCE} served {df.index.min().date()} -> {df.index.max().date()} for "
            f"range={rng}, {why}. An unrecognised range returns one year with a 200, so this is "
            "what a wrong range string looks like."
        )
    return df


def fetch_series(symbol: str, needs: str | None = None, rng: str = "10Y",
                 min_span_years: float | None = None) -> pd.DataFrame:
    """`date, close, adjusted, volume` for one symbol, oldest first.

    `needs` is the earliest date the caller requires, and it is CHECKED rather than requested --
    see `_check_served`.

    `min_span_years` IS THE SAME GUARD STATED AS A LENGTH, and it exists because `10Y` is ten
    years back from the ticker's LAST traded bar, not from today. A name that stopped trading in
    2016 is served 2006-2016 and a name that stopped in 2025 is served 2015-2025, so there is no
    single absolute date that a working response must reach -- while a silent fallback to one
    year is still exactly as detectable, because what makes it detectable is the SPAN. Use this
    when the caller's earliest requirement varies per symbol; use `needs` when it does not.
    """
    return _check_served(symbol, _parse(symbol, _get(symbol, rng=rng)), rng, needs, min_span_years)


def fetch_longest(symbol: str, min_span_years: float, ranges: tuple[str, ...] = RANGES,
                  pace: float = 1.0) -> tuple[pd.DataFrame, str]:
    """The longest series any of `ranges` serves, and which range served it.

    `MAX` IS UNRELIABLE AT THIS VENDOR AND THE UNRELIABILITY IS NOT PER-SYMBOL. Measured on
    2026-09-06: AVB served 8,159 bars back to 1994 three times running, then 252 bars -- one year
    -- on every subsequent request over the following minutes, while `10Y` served 2,513 bars for
    it throughout. `20Y`, `15Y`, `30Y` and `All` fall back to one year for every symbol tried, so
    they are not alternatives. That is why the ranges are tried in order and the LONGEST span
    wins, rather than the first that clears the floor: a one-year fallback is a 200 with plausible
    data, so `MAX` can never be taken at face value, and `10Y` alone would leave the
    vendor-truncated names that are still trading starting in 2016 -- absent from exactly the
    early windows they were fetched to fill. The rule makes both outcomes safe and neither
    silent: the served span is in the provenance file per symbol, so a refetch that gets a long
    `MAX` is visible rather than a quiet change in what the panel covers.

    The floor is checked ONCE, against the winner. A range that fell back is not an error here --
    it is the expected outcome for most symbols and the reason there is more than one range.
    """
    best: tuple[pd.DataFrame, str] | None = None
    for i, rng in enumerate(ranges):
        df = _parse(symbol, _get(symbol, rng=rng))
        if best is None or _span_years(df) > _span_years(best[0]):
            best = (df, rng)
        if i < len(ranges) - 1:
            time.sleep(pace)
    return _check_served(symbol, best[0], "+".join(ranges), None, min_span_years), best[1]


def refresh(symbols: dict[str, str], needs: str | None = None, cache: Path = CACHE,
            min_span_years: float | None = None, skip_failures: bool = False,
            longest: bool = False) -> dict[str, str]:
    """Fetch and cache. Returns constituent ticker -> the file written.

    `skip_failures` collects rather than raises, for the point-in-time membership set: the vendor
    serves most of its names and the ones it does not are a fact to report, not a reason for the
    rest to go unmeasured. Off by default, because on the 2023 set every named symbol is one the
    vendor was already confirmed to serve and a failure there means something changed.

    `longest` swaps the single `10Y` request for `fetch_longest`. Off by default so the 2023 set
    refetches to the same bytes it was published from.
    """
    out: dict[str, str] = {}
    (cache / "raw").mkdir(parents=True, exist_ok=True)
    for i, (sym, served) in enumerate(symbols.items()):
        try:
            if longest:
                df, rng = fetch_longest(served, min_span_years)
            else:
                df, rng = fetch_series(served, needs=needs, min_span_years=min_span_years), "10Y"
        except Exception as exc:
            if not skip_failures:
                raise
            print(f"{sym:6s} <- {served:6s} SKIPPED  {type(exc).__name__}: {str(exc)[:110]}",
                  flush=True)
            if i < len(symbols) - 1:
                time.sleep(1.0)   # a rejected symbol still spent a request; pace the next one
            continue
        path = cache / "raw" / f"{sym}.csv"
        df.to_csv(path)
        out[sym] = path.name
        print(f"{sym:6s} <- {served:6s} {len(df):5d} bars "
              f"{df.index.min().date()} -> {df.index.max().date()}  range={rng}", flush=True)
        if i < len(symbols) - 1:
            time.sleep(1.0)
    return out


def load(symbol: str, cache: Path = CACHE) -> pd.DataFrame:
    path = cache / "raw" / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is not cached. Run `python -u pipeline/delisted.py --refresh` -- the stitch "
            "never fetches, so a rebuild cannot silently move its own inputs."
        )
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


# ------------------------------------------------------------------------------- the gate

def crosscheck(symbols: list[str], price_dir: Path = PRICE_DIR,
               needs: str = "2019-08-01") -> dict[str, float]:
    """Max deviation of `second_vendor / store_total_return` from its own median, per symbol.

    A ratio and not a level: the two series agree only up to a scale factor. See the module
    docstring for why `SPINOFF_CONTROLS` is measured and reported but not held to `GATE`.
    """
    close = store.read_close(price_dir, symbols)
    divs = store.dividends_by_symbol(price_dir)
    out: dict[str, float] = {}
    for i, sym in enumerate(symbols):
        tr = store.total_return(close[sym].dropna(), divs.get(sym))
        a = fetch_series(sym, needs=needs)["adjusted"]
        idx = tr.index.intersection(a.index)
        if len(idx) < 250:
            raise RuntimeError(f"{sym}: only {len(idx)} overlapping bars, too few to gate on")
        ratio = (a.loc[idx] / tr.loc[idx]).dropna()
        out[sym] = float((ratio / ratio.median() - 1.0).abs().max())
        if i < len(symbols) - 1:
            time.sleep(1.0)
    return out


# ------------------------------------------------------------------------------ the stitch

def terminal(df: pd.DataFrame) -> tuple[pd.Timestamp, float, dict]:
    """The last bar this module will trust, which is the PENULTIMATE traded bar.

    Returns (date, adjusted level, evidence). The evidence is what the choice cost -- the
    dropped bar's date, level, volume, and the return from the kept bar to it -- so the
    provenance file states the size of the assumption instead of asserting it is small.
    """
    if len(df) < 3:
        raise RuntimeError(f"{len(df)} bars is too few to drop one from")
    kept, dropped = df.index[-2], df.index[-1]
    lvl = float(df["adjusted"].loc[kept])
    if not df["volume"].loc[kept] > 0:
        raise RuntimeError(
            f"the kept terminal bar {kept.date()} has volume {df['volume'].loc[kept]}, so it is "
            "a stub too; this symbol needs its own look rather than the uniform rule"
        )
    return kept, lvl, {
        "terminal_date": str(kept.date()),
        "terminal_level": round(lvl, 4),
        "dropped_date": str(dropped.date()),
        "dropped_level": round(float(df["adjusted"].loc[dropped]), 4),
        "dropped_volume": int(df["volume"].loc[dropped]),
        "dropped_bar_return": round(float(df["adjusted"].loc[dropped]) / lvl - 1.0, 6),
    }


def _cash_path(rf_annual: pd.Series) -> pd.Series:
    """A T-bill value path on `rf_annual`'s bars, starting at 1.0. Same daily convention as
    `backtest.lever`: an annualised quote compounded over `TRADING_DAYS_PER_YEAR`."""
    daily = (1.0 + rf_annual) ** (1.0 / fetch.TRADING_DAYS_PER_YEAR) - 1.0
    return (1.0 + daily.fillna(0.0)).cumprod()


def stitch(symbols: list[str], continuation: str, calendar: pd.Series,
           cache: Path = CACHE) -> tuple[pd.DataFrame, dict]:
    """A total-return LEVEL panel for `symbols`, real to each one's terminal bar and continued
    to the end of `calendar` under `continuation`.

    `calendar` is the continuation asset's own value path on the trading days to fill -- SPY's
    total return, or a T-bill path -- so the trading calendar comes from the primary vendor and
    not from a vendor whose holiday handling nothing here has checked. Levels are arbitrary in
    scale (only returns are used downstream), so each column is left on the vendor's own scale.
    """
    if continuation not in CONTINUATIONS:
        raise ValueError(f"continuation must be one of {CONTINUATIONS}: {continuation!r}")
    cols: dict[str, pd.Series] = {}
    facts: dict[str, dict] = {}
    for sym in symbols:
        df = load(sym, cache)
        end, lvl, ev = terminal(df)
        real = df["adjusted"].loc[:end]
        # Reindexed onto the primary vendor's calendar, then intersected -- NOT forward filled.
        # A vendor's extra or missing holiday would otherwise become a fabricated zero return,
        # and `fetch.load_panel`'s docstring is about why that is the one thing a covariance
        # estimator cannot see through. `load_panel` runs the same intersection downstream, so
        # a real hole here shows up there as a coverage drop with a reason attached.
        real = real.reindex(calendar.index.intersection(real.index))
        after = calendar.loc[calendar.index > end]
        if len(after):
            tail = lvl * after / float(calendar.asof(end))
        else:
            tail = pd.Series(dtype=float)
        s = pd.concat([real, tail])
        if not s.index.is_monotonic_increasing or s.index.has_duplicates:
            raise RuntimeError(f"{sym}: the stitch is not a clean single series")
        cols[sym] = s
        facts[sym] = {
            # `.get`, so `stitch` works on any symbol the cache holds rather than only on the
            # twelve this module names. A function that reads a module constant for a value its
            # own argument determines cannot be tested on anything but the shipped set.
            "served_as": SERVED_AS.get(sym, sym),
            "first": str(real.index.min().date()),
            "real_bars": int(len(real)),
            "continued_bars": int(len(tail)),
            **ev,
        }
    panel = pd.DataFrame(cols)
    return panel, facts


# --------------------------------------------------- what the five nobody prices could do

def stress_bound(block: dict, panel: pd.DataFrame, spy: pd.Series,
                 unpriceable: list[str] = UNPRICEABLE,
                 extra_panel: dict | None = None) -> dict:
    """A REVERSE stress test on the constituents neither vendor serves: not "here is my estimate
    of what they did" but "what would they have had to do to overturn the answer".

    Two bounds, because the two claims in the result are bounded by different arguments and
    conflating them is how a stress test becomes decoration.

    1. THE EQUAL-WEIGHT COMPARATOR IS BOUNDED EXACTLY. It holds every candidate, so the missing
       names enter with known weight and unknown return, and terminal wealth is linear in their
       growth multiples. `all_wiped_out` is every one of them at zero; `all_at_the_best_observed`
       is every one of them matching the best outcome any priced delisted constituent achieved --
       a bound taken from their own cohort rather than from a premium assumption, because the
       premium assumption is the thing this exercise just measured to be wrong in sign.

       `unpriceable` IS THE CALLER'S TO DETERMINE and the default is only right for the 2023
       exclusion set. On the point-in-time membership the set differs per run -- the names the
       second vendor restores are missing from a baseline run and present in a restored one -- so
       `pit.py` unions each period's own count of constituents nothing priced and passes that.
       Reading a list off the membership file would bound a run that was not the one measured.

    2. THE OPTIMISED ROWS CANNOT BE BOUNDED THAT WAY, and the honest statement is a MOVE the
       unpriceable names would have had to make, not an estimate. Whether the optimiser would have bought a name it has no prices for is
       not answerable from here, so the bound is on the exposure the cap permits: five names at
       `cap` each is the most of the portfolio they could occupy, and the arithmetic is how far
       they would have had to fall short of everything else, over that exposure, to pull the
       strategy's return per unit of risk down to the benchmark's. Reported alongside the fact
       that decides how plausible that is: how often the TWELVE names that *were* restored were
       actually held by this strategy. In the shipped S&P run the answer is never, in any period,
       under either continuation -- only `min_variance` bought any of them, and only once their
       training window had gone mostly synthetic.

    IT TAKES A BLOCK AND NOT A RESULT DOCUMENT, so that a caller whose periods do not live under
    `rolling[0]` -- `pit.py`'s do not -- cannot get a vacuous answer by handing over a document
    this function digs into and finds nothing in. Same reason `extra_panel` is passed rather than
    looked up: whether a run restored anything is the fact that decides whether
    `times_a_restored_name_was_held` means anything at all.
    """
    periods = block["periods"]
    # A period carries its own `cutoff` and its test block's `end` -- the test block has no
    # `start`, because the hold begins at the cutoff. Reading `test["start"]` returns None and
    # `pd.Timestamp(None)` is NaT, which propagates into every number below without raising.
    a = pd.Timestamp(periods[0]["cutoff"])
    b = pd.Timestamp(periods[-1]["test"]["end"])
    if pd.isna(a) or pd.isna(b) or b <= a:
        raise ValueError(f"the rolling block's span reads {a} -> {b}")
    years = (b - a).days / 365.25
    n_extra = len(unpriceable)

    growth = {c: float(panel[c].dropna().asof(b) / panel[c].dropna().asof(a))
              for c in panel.columns}
    best = max(growth.values())
    best_sym = max(growth, key=growth.get)
    spy_growth = float(spy.asof(b) / spy.asof(a))

    ew = [_row(p, "equal_weight") for p in periods]
    n_held = ew[0]["n_holdings"]
    ew_growth = float(np.prod([1.0 + r["realized_monthly"]["ret"] for r in ew]))

    def annual(g: float) -> float:
        return g ** (1.0 / years) - 1.0

    # Keyed by COUNT, not by the word "five". The 2023 exclusion set has five names nobody prices
    # and the point-in-time membership has sixty-five, and a field called `five_wiped_out` holding
    # the sixty-five-name bound is the class of error this repo's label rules exist for.
    bounds = {
        "as_measured": annual(ew_growth),
        "all_wiped_out": annual((n_held * ew_growth) / (n_held + n_extra)),
        "all_at_the_best_observed": annual(
            (n_held * ew_growth + n_extra * best) / (n_held + n_extra)),
    }

    out = {
        # A DISCRIMINATOR, because this file lands in `pipeline/results/` beside the backtest
        # results and `test_results_invariants.py` globs that directory. Without it every
        # assertion in that module runs over a file with no `periods` and passes vacuously,
        # which is the one failure mode its docstring is about.
        "kind": "stress_bound",
        "unpriceable": list(unpriceable),
        "span": {"start": str(a.date()), "end": str(b.date()), "years": round(years, 3)},
        "best_observed_growth": {"symbol": best_sym, "multiple": round(best, 4)},
        "spy_growth": round(spy_growth, 4),
        "n_unpriceable": n_extra,
        "equal_weight": {
            "held": n_held,
            "annualised": {k: round(v, 5) for k, v in bounds.items()},
            "widest_swing_pp_per_year": round(
                100 * (bounds["all_at_the_best_observed"] - bounds["all_wiped_out"]), 3),
        },
        "optimised": {},
        "restored_names_in_this_result": (
            "none -- this is a baseline run, so `times_a_restored_name_was_held` is not "
            "measurable from it and is null" if not extra_panel
            else extra_panel["file"]),
    }

    bm_label = f"benchmark:{block['benchmark']}"
    bm = [(_row(p, bm_label) or {}).get("realized_monthly", {}) for p in periods]
    bm_growth = float(np.prod([1.0 + r["ret"] for r in bm]))
    bm_risk = float(np.mean([r["vol"] for r in bm]))
    target = annual(bm_growth) / bm_risk
    # ONLY MEANINGFUL ON A RESTORED RUN. Counted against a baseline result the twelve are not in,
    # every strategy holds none of them and the field reads as evidence when it is a tautology --
    # so it is None there and the caller is told which kind of file it passed.
    restored = list(panel.columns) if extra_panel else None
    for label in [s["strategy"] for s in periods[0]["strategies"]]:
        if not (label.startswith("matched_risk") or label.startswith("tangency@")):
            continue
        rows = [_row(p, label) for p in periods]
        if any(r is None for r in rows):
            continue
        cap = rows[0]["cap"]
        g = float(np.prod([1.0 + r["realized_monthly"]["ret"] for r in rows]))
        risk = float(np.mean([r["realized_monthly"]["vol"] for r in rows]))
        # The growth at which this row's return per unit of RUN risk equals the benchmark's. Risk
        # held fixed: a move concentrated in a few names would shift it too, in the direction that
        # makes the crossing harder, so ignoring that is the conservative reading.
        flip = (1.0 + target * risk) ** years
        # SIGNED, AND THE SIGN IS THE WHOLE POINT. On the 2023 single-window run the optimised rows
        # BEAT the benchmark, so the bound was "how far short of everything else would the
        # unpriceable names have had to fall to overturn this" and a positive number meant a robust
        # result. On the point-in-time run they LOSE, and the same subtraction comes out negative --
        # which does not mean the bound is small, it means the question has flipped: what the
        # unpriceable names would have had to do is OUTPERFORM. A field called
        # `terminal_shortfall_needed` holding -0.74 states the second case in the vocabulary of the
        # first, so the name is neutral and the direction is a separate boolean.
        move = flip / g - 1.0
        # `min(1.0, n * cap)` SATURATES, and at 65 names it saturates at every cap this run uses.
        # An exposure bound that permits the whole portfolio bounds nothing, so it is flagged
        # rather than printed as though 1.0 were a finding.
        exposure = min(1.0, n_extra * cap)
        out["optimised"][label] = {
            "return_per_unit_of_risk": round(annual(g) / risk, 4),
            "benchmark_return_per_unit_of_risk": round(target, 4),
            "beats_the_benchmark_per_unit_of_risk": bool(annual(g) / risk > target),
            "max_exposure_at_this_cap": round(exposure, 3),
            "exposure_bound_is_vacuous": exposure >= 1.0,
            "terminal_move_needed_to_equalise": round(move, 5),
            "direction": "the unpriceable names would have had to OUTPERFORM everything else"
                         if move > 0 else
                         "the unpriceable names would have had to UNDERPERFORM everything else",
            "move_over_that_exposure": round(move / exposure, 5) if exposure > 0 else None,
            "times_a_restored_name_was_held": None if restored is None else sum(
                1 for p in periods for s in p["strategies"]
                if s["strategy"] == label
                and any(k in restored for k in (s["top_holdings"] or {}))),
        }
    return out


def write_panels(symbols: list[str], continuations: list[str], cache: Path, price_dir: Path,
                 gate: dict[str, float] | None = None, unpriceable: list[str] = UNPRICEABLE,
                 caveats: list[str] = (), extra: dict | None = None) -> list[Path]:
    """One `panel_<treatment>.parquet` + `provenance_<treatment>.json` pair per continuation.

    Shared between the 2023 exclusion set and the point-in-time set rather than copied, because
    the provenance file is the record of what an `--extra-panel` run was fed and two copies of it
    would drift in exactly the fields a reader would trust.
    """
    irx = pd.read_parquet(ROOT / "pipeline" / "data" / "irx.parquet")["rf"]
    spy = store.read_total_return(price_dir, ["SPY"])["SPY"].dropna()
    written = []
    for treatment in continuations:
        calendar = spy if treatment == "spy" else _cash_path(irx.reindex(spy.index).ffill())
        panel, facts = stitch(symbols, treatment, calendar, cache)
        out = cache / f"panel_{treatment}.parquet"
        panel.to_parquet(out)
        prov = {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source": SOURCE,
            "continuation": treatment,
            "continuation_note": {
                "cash": "proceeds in 13-week T-bills at ^IRX as it moved",
                "spy": "proceeds reinvested in SPY total return on the terminal bar",
            }[treatment],
            "terminal_bar_rule": "the last traded bar is dropped; see delisted.py's docstring",
            "gate": {k: float(f"{v:.3e}") for k, v in (gate or {}).items()}
            or "not run this invocation",
            "gate_threshold": GATE,
            "spinoff_controls_excluded_from_gate": SPINOFF_CONTROLS,
            "caveats": list(caveats),
            "unpriceable": list(unpriceable),
            "symbols": facts,
            "panel": {"start": str(panel.index.min().date()),
                      "end": str(panel.index.max().date()),
                      "bars": int(len(panel)), "columns": int(panel.shape[1])},
            **(extra or {}),
        }
        (cache / f"provenance_{treatment}.json").write_text(json.dumps(prov, indent=2) + "\n")
        written.append(out)
        print(f"wrote {out} ({panel.shape[0]} x {panel.shape[1]}) and its provenance", flush=True)
    return written


def _row(period: dict, label: str) -> dict | None:
    """`backtest._row`, duplicated rather than imported. Importing `backtest` here would make the
    dependency circular -- it imports this module's panel through `--extra-panel` -- and a
    two-line lookup is not worth a shared module that exists only to hold it."""
    return next((s for s in period["strategies"] if s["strategy"] == label), None)


# --------------------------------------------------------------------------------- the CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="refetch the raw series from the vendor")
    ap.add_argument("--gate", action="store_true",
                    help="run the ratio cross-check against the store and print it")
    ap.add_argument("--needs", default="2019-08-01",
                    help="the earliest date every series must reach, checked not requested")
    ap.add_argument("--continuation", default=",".join(CONTINUATIONS),
                    help="post-delisting treatments to write, comma separated")
    ap.add_argument("--stress", type=Path, default=None,
                    help="a backtest result JSON: print the reverse stress bound for the "
                         "constituents neither vendor prices and write it beside the result")
    ap.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args(argv)

    if args.refresh:
        refresh(SERVED_AS, needs=args.needs, cache=args.cache)

    gate: dict[str, float] = {}
    if args.gate:
        gate = crosscheck(CONTROLS + SPINOFF_CONTROLS, args.price_dir, args.needs)
        for sym, dev in gate.items():
            flag = "  (spinoff, outside the gate)" if sym in SPINOFF_CONTROLS else ""
            print(f"{sym:6s} max ratio deviation {dev:.2e}{flag}", flush=True)
        bad = {s: d for s, d in gate.items() if s not in SPINOFF_CONTROLS and d > GATE}
        if bad:
            raise SystemExit(f"the second source failed the ratio gate on {bad}")

    written = write_panels(
        list(SERVED_AS), args.continuation.split(","), args.cache, args.price_dir, gate=gate,
        caveats=[
            "K (Kellanova) spun off WK Kellogg in 2023 and this vendor credits a "
            "distribution in kind that the store does not, so K's level carries the same "
            "disagreement the T control measures at 1.0e-1 and there is no second series "
            "to resolve it against",
            "CTRA's last traded bar is 2026-05-06 and the corporate event behind the "
            "delisting is not known here; the terminal level is what the market paid, "
            "which is the number the measurement needs, but the event is unstated because "
            "it was not verified",
            f"{len(UNPRICEABLE)} constituents ({', '.join(UNPRICEABLE)}) are served by "
            "neither vendor at any date and are not in this panel at all",
        ])

    if args.stress is not None:
        irx = pd.read_parquet(ROOT / "pipeline" / "data" / "irx.parquet")["rf"]
        spy = store.read_total_return(args.price_dir, ["SPY"])["SPY"].dropna()
        result = json.loads(args.stress.read_text())
        panel, _ = stitch(list(SERVED_AS), "cash", _cash_path(irx.reindex(spy.index).ffill()),
                          args.cache)
        bound = stress_bound(result["rolling"][0], panel, spy,
                             extra_panel=result["universe"].get("extra_panel"))
        bound["result"] = args.stress.name
        out = args.stress.with_name(args.stress.stem + "_stress.json")
        out.write_text(json.dumps(bound, indent=2) + "\n")
        print(json.dumps(bound, indent=2), flush=True)
        print(f"wrote {out}", flush=True)

    if not written and not args.gate and args.stress is None:
        print("nothing to do -- pass --refresh, --gate and/or --stress", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
