"""Daily total-return bars from Yahoo's v8 chart endpoint. Standard library only.

Why not yfinance
----------------
This collector runs unattended in a GitHub Actions container. `yfinance` is a scraper
whose working state depends on Yahoo's cookie/crumb flow and which has broken on minor
releases several times; pinning it means pinning to whatever was working on the day the
pin was written, and not pinning means the cron silently starts failing. The v8 chart
endpoint used here needs no cookie, no crumb and no API key, returns the entire history
in ONE response at `range=100y`, and is 60 lines of `urllib`. There is no pagination to
get wrong and no third-party release to track.

THE PRICE IS A TOTAL RETURN, AND THE RAW CLOSE WOULD BE A DIFFERENT ANSWER
--------------------------------------------------------------------------
Every number the optimiser sees is a dividend-adjusted price, never a raw close. Two
reasons, and the second is the one that actually bites:

1. Dividends. SPY's raw close compounds ~1.8%/yr below its dividend-adjusted close.
   That is a fifth of the equity risk premium. But the real damage is CROSS-SECTIONAL:
   the yield spread across this universe runs from ~0% (UNG, gold) to ~4% (PFF, HYG,
   VNQ, VYM), so using raw closes does not shift the frontier uniformly -- it
   systematically understates exactly the income assets whose whole return IS the
   dividend, and the optimiser would conclude that high-yield credit is a dominated
   asset. On a total-return basis it is not.
2. Splits. Several of these funds have reverse-split (USO 1:8 in 2020, UNG 1:4 in 2020 and
   again later, GDXJ, DBC). An unadjusted 1:8 reverse split is a +700% single-day return;
   one such row in a 3,900-row sample dominates that asset's variance AND every entry in
   its covariance row, so a single unhandled split corrupts the matrix rather than one
   column.

   MEASURED, NOT ASSUMED: the v8 endpoint's `close` array is ALREADY split-adjusted, and
   `adjclose` adds only the dividend adjustment on top. USO and UNG -- the two heaviest
   reverse-splitters here and payers of no distributions -- come back with `close` and
   `adjclose` identical to the cent across their full histories, and neither contains a
   split-sized move. So splits are handled whichever column is read, and reason (1) is the
   only one that requires an adjustment of our own. Worth stating explicitly, because "the
   raw close contains the splits" is the natural assumption and it is false here.

WHERE THE DIVIDEND ADJUSTMENT HAPPENS, AND WHY NOT HERE
------------------------------------------------------
`parse` returns BOTH columns and the event log, but the panel handed to the optimiser is
built by `store.total_return()` from `close` plus the dividend log -- not from the vendor's
`adjclose`. That is a storage decision, not a modelling one, and `store.py`'s docstring is
where it is argued: the adjustment is retroactive, so an `adjclose` column can never be
stored in an append-only way, while `close` and a dividend log can. The reconstruction is
checked against the vendor's own field to ~1e-6 relative, which is the precision at which
that field ships (float32).

The adjustment is applied BACKWARD: the most recent adjusted price equals the most recent
`close` by construction, and the divergence accumulates going back in time (SPY's first
1993 bar has close/adjclose = 1.82). Any check that the adjustment happened has to look at
the START of a series -- at the end the ratio is 1.0000 for every symbol, dividend payer or
not.

`close` is stored alongside the events so the two can be compared rather than trusted.
`tests/test_fetch_invariants.py` asserts the reconstructed total-return series compounds
strictly faster than the price series for every asset (dividends can only add, so the
reverse ordering is a swapped column or a sign error) and bounds the implied yield gap at a
plausible level.

THE INDEX IS THE TRADING DAY, AND NOTHING IS REINDEXED
------------------------------------------------------
The gaps in these series are weekends and holidays: they are the absence of trading, not
missing data. Reindexing onto a complete daily calendar would fabricate ~40% more rows,
forward-fill a price into each, and hand the covariance estimator a zero-return Saturday
as a real observation -- which deflates every variance by ~40% and, worse, deflates them
unequally, since it also injects a perfect common zero into every pair and pulls all
correlations toward zero. Annualisation therefore uses 252 TRADING bars per year, and
the panel is aligned by INTERSECTION of trading days across assets (see `load_panel`),
never by union-and-fill.

A partial response RAISES rather than returning what arrived. There is no pagination to
resume from, so a truncated history is indistinguishable from a young fund, and a young
fund gets dropped by the window filter -- i.e. the failure mode of returning partial
data is silent exclusion, which is the worst kind.
"""
from __future__ import annotations

import http.client
import json
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import store

BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

# The endpoint returns 429 to urllib's default agent.
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

TRADING_DAYS_PER_YEAR = 252.0

RETRIES = 6
RETRYABLE = (
    urllib.error.URLError,
    urllib.error.HTTPError,
    socket.timeout,
    json.JSONDecodeError,
    http.client.IncompleteRead,
    http.client.HTTPException,
    ConnectionError,
    TimeoutError,
)


def _get(symbol: str, rng: str = "100y", timeout: float = 30.0) -> dict:
    # `^IRX` and friends carry a caret, which is not a legal path character; unquoted it
    # reaches Yahoo as a malformed request rather than as a 404, so the failure looks like
    # a network problem.
    url = (
        f"{BASE}/{urllib.parse.quote(symbol, safe='')}?range={rng}&interval=1d"
        "&events=div%2Csplit&includeAdjustedClose=true"
    )
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except RETRYABLE as exc:  # noqa: PERF203 - retry is the point
            last = exc
            if attempt < RETRIES - 1:
                time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"{symbol}: giving up after {RETRIES} attempts: {last!r}")


def parse(payload: dict, symbol: str) -> pd.DataFrame:
    """Chart JSON -> DataFrame indexed by trading date with close and adjclose.

    Yahoo returns the session OPEN timestamp in exchange-local time, so the intraday
    component shifts by an hour across daylight saving. It is normalised to the New York
    DATE because the trading DAY is the actual index and the time-of-day is an artifact
    of the endpoint.
    """
    result = (payload.get("chart") or {}).get("result")
    if not result:
        err = (payload.get("chart") or {}).get("error")
        raise ValueError(f"{symbol}: no result ({err})")
    r = result[0]
    ts = r.get("timestamp")
    if not ts:
        raise ValueError(f"{symbol}: no timestamps")
    quote = r["indicators"]["quote"][0]
    adj = r["indicators"].get("adjclose")
    if not adj:
        raise ValueError(f"{symbol}: no adjclose (includeAdjustedClose ignored)")

    idx = (
        pd.to_datetime(pd.Series(ts), unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    df = pd.DataFrame(
        {"close": quote["close"], "adjclose": adj[0]["adjclose"]}, index=pd.Index(idx, name="date")
    )
    df = df.dropna(subset=["adjclose"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        raise ValueError(f"{symbol}: empty after cleaning")
    if (df["adjclose"] <= 0).any():
        raise ValueError(f"{symbol}: non-positive adjusted close")
    return df


def parse_events(payload: dict, symbol: str) -> pd.DataFrame:
    """Chart JSON -> the dividend and split log: columns symbol, date, kind, amount.

    Requested with `&events=div,split`. The dividends are what `store.total_return` needs;
    the splits are recorded because `close` arrives already split-adjusted, so having the
    split dates stored is what lets a test assert that -- a split-sized jump surviving in
    `close` on a date the log knows about is a specific, diagnosable failure, whereas the
    same jump with no event log is just an outlier.

    Ex-dates are normalised to the New York DATE, exactly as the bars are in `parse`; the
    reconstruction indexes into the bar series by date, so any other convention would put a
    factor on the wrong side of a boundary once or twice a year.
    """
    r = ((payload.get("chart") or {}).get("result") or [{}])[0]
    events = r.get("events") or {}
    rows = []
    for key, kind, field in (("dividends", "dividend", "amount"), ("splits", "split", None)):
        for rec in (events.get(key) or {}).values():
            if "date" not in rec:
                continue
            if kind == "dividend":
                amount = rec.get("amount")
            else:
                # Yahoo gives numerator/denominator; store the ratio a 1:8 reverse split is
                # (0.125), so the number means the same thing in both directions.
                num, den = rec.get("numerator"), rec.get("denominator")
                amount = (float(num) / float(den)) if num and den else None
            if amount is None:
                continue
            rows.append({"symbol": symbol, "date": _ny_date(rec["date"]), "kind": kind, "amount": float(amount)})
    if not rows:
        return pd.DataFrame(
            {"symbol": pd.Series(dtype="object"), "date": pd.Series(dtype="datetime64[ns]"),
             "kind": pd.Series(dtype="object"), "amount": pd.Series(dtype=float)}
        )
    return pd.DataFrame(rows).sort_values(["date", "kind"], ignore_index=True)


def _ny_date(epoch_seconds) -> pd.Timestamp:
    return (
        pd.Timestamp(int(epoch_seconds), unit="s", tz="UTC")
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )


def download(
    symbols,
    out_dir: Path,
    max_age_days: int = 1,
    pause: float = 0.4,
    verbose: bool = True,
    flush_every: int = 25,
) -> tuple[list[str], dict[str, str]]:
    """Fetch each symbol into the year-partitioned store at `out_dir`. Returns (ok, failures).

    `max_age_days` skips a symbol whose stored history already reaches within that many
    days of now, so a local re-run is instant. In CI the directory starts empty, so
    everything is fetched -- which is what makes the committed output reproducible from
    the workflow alone rather than from whatever happened to be on a laptop.

    A per-symbol failure is COLLECTED, not raised: one delisted ticker must not take down
    a 130-symbol run. It is returned in the failure map and ends up in the output
    manifest, because a symbol that vanished from the universe without a recorded reason
    is the same bug as a wrong price.

    WRITES ARE BATCHED, and that is forced by the layout rather than an optimisation. Each
    year partition holds every symbol, so writing one symbol at a time would rewrite all ~34
    files 132 times per run. Instead symbols accumulate and flush every `flush_every` of
    them, which bounds what a crashed or rate-limited run loses to one batch while keeping
    the file rewrites in the hundreds rather than the thousands.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_seen = store.last_dates(out_dir)
    today = pd.Timestamp.today().normalize()

    ok: list[str] = []
    failed: dict[str, str] = {}
    pending_frames: dict[str, pd.DataFrame] = {}
    pending_events: dict[str, pd.DataFrame] = {}

    def flush() -> None:
        if not pending_frames:
            return
        years = store.write(out_dir, pending_frames, pending_events)
        if verbose:
            # "no year changed" is the expected result of re-fetching a symbol whose history has
            # not moved, not an anomaly -- see `store._write_if_changed`. Printed either way,
            # because it is also the number to watch if the churn ever grows.
            where = f"years {years[0]}..{years[-1]}" if years else "no year changed"
            print(f"      flushed {len(pending_frames)} symbols -> {where}", flush=True)
        pending_frames.clear()
        pending_events.clear()

    for i, sym in enumerate(symbols, 1):
        last = last_seen.get(sym)
        if last is not None and (today - last).days <= max_age_days:
            ok.append(sym)
            if verbose:
                print(f"[{i:3d}/{len(symbols)}] {sym:<6} cached", flush=True)
            continue
        try:
            payload = _get(sym)
            df = parse(payload, sym)
            pending_frames[sym] = df
            pending_events[sym] = parse_events(payload, sym)
            ok.append(sym)
            if verbose:
                print(
                    f"[{i:3d}/{len(symbols)}] {sym:<6} {len(df):>5} bars  "
                    f"{df.index.min().date()} -> {df.index.max().date()}  "
                    f"{len(pending_events[sym]):>3} events",
                    flush=True,
                )
        except Exception as exc:
            failed[sym] = f"{type(exc).__name__}: {exc}"
            print(f"[{i:3d}/{len(symbols)}] {sym:<6} FAILED  {failed[sym]}", file=sys.stderr, flush=True)
        if len(pending_frames) >= flush_every:
            flush()
        time.sleep(pause)
    flush()
    return ok, failed


def load_panel(
    symbols,
    price_dir: Path,
    start: str | None = None,
    end: str | None = None,
    min_coverage: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Wide TOTAL-RETURN panel on the INTERSECTION of trading days. Returns (panel, dropped).

    The prices come from `store.read_total_return`, i.e. stored closes with the stored
    dividend log applied over each symbol's FULL history. The adjustment therefore happens
    before any windowing here, which is required rather than incidental: the factor at a bar
    depends on every dividend after it, so windowing first would leave the end of the window
    under-adjusted. See `store.total_return`.

    The two filters, in order, and why the order matters:

    1. START DATE. A symbol whose first bar is after `start` is dropped. Doing this first
       means the window is a stated parameter rather than an emergent property of the
       youngest survivor -- the alternative (keep everything, start the panel where the
       last arrival starts) lets one 2018 fund silently discard seven years of every
       other asset's history.
    2. COVERAGE. Of the trading days remaining, a symbol must have a price on at least
       `min_coverage` of them. This catches the fund that exists across the whole window
       but stopped trading for a month, which an intersection would otherwise handle by
       deleting that month from all 130 assets.

    Only then is the intersection taken, and it is an intersection (`dropna`) rather than
    a forward fill: a forward-filled hole is a fabricated zero return, and zero returns
    are the one thing a covariance estimator cannot see through.
    """
    total = store.read_total_return(price_dir, symbols)
    frames: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}
    for sym in symbols:
        if sym not in total.columns:
            dropped[sym] = "not in the price store"
            continue
        s = total[sym].dropna()
        if start is not None and s.index.min() > pd.Timestamp(start):
            dropped[sym] = f"history starts {s.index.min().date()}, after {start}"
            continue
        frames[sym] = s

    if not frames:
        raise RuntimeError("no symbols survived the start-date filter")

    wide = pd.DataFrame(frames)
    if start is not None:
        wide = wide.loc[wide.index >= pd.Timestamp(start)]
    if end is not None:
        wide = wide.loc[wide.index <= pd.Timestamp(end)]

    coverage = wide.notna().mean()
    for sym in coverage.index[coverage < min_coverage]:
        dropped[sym] = f"coverage {coverage[sym]:.3f} < {min_coverage}"
    wide = wide.loc[:, coverage >= min_coverage]

    wide = wide.dropna(how="any")
    if len(wide) < 2:
        raise RuntimeError(f"intersection left {len(wide)} rows")
    return wide, dropped


def monthly_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns ANCHORED AT THE PANEL'S FIRST BAR, for the SPA's growth curve.

    Monthly rather than daily because this is the only artifact whose size scales with both
    assets and time: 116 assets x 3,939 daily bars is 457,000 floats to ship to a browser,
    and 116 x 189 monthly bars is 22,000 -- a 20x cut for a curve drawn a few hundred pixels
    wide that cannot resolve a daily wiggle anyway. No optimiser input is computed from
    these; risk and return come from the daily panel.

    THE ANCHOR IS THE POINT OF THIS FUNCTION. The obvious implementation --
    `panel.resample("ME").last().pct_change()` -- drops the first month-end row, so the
    series starts at the END of the first month and silently covers a SHORTER window than
    the daily panel. Cumulative growth from it then disagrees with the `mu` the optimiser
    used, and the size of the disagreement scales with volatility: GDXJ (~50% vol, and
    falling hard through January 2011) came out 1.0 percentage point of annualised return
    too high, while low-vol assets came out near-identical. That is the worst shape of
    error -- large exactly where the asset is most interesting, small enough elsewhere to
    look like rounding.

    Prepending the panel's first row as the anchor fixes it exactly rather than
    approximately: chained month-end price ratios telescope, so the product of (1 + monthly
    return) equals P_last / P_first for the same first and last bar the daily panel uses.
    The two artifacts then agree to float precision, which is what makes
    `test_monthly_compounding_reproduces_the_annualised_mean_return` a real cross-check
    instead of a tolerance wide enough to hide a wrong resample rule.

    The first and last entries are PARTIAL months (the anchor is a mid-month trading day,
    and the final bucket ends at the last bar rather than at a month end). That is recorded
    in the output's `anchor` field so a reader is not told a partial month is a full one.
    """
    m = panel.resample("ME").last()
    anchor = panel.iloc[[0]]
    m = pd.concat([anchor, m.loc[m.index > anchor.index[0]]])
    return m.pct_change().dropna(how="all")
