"""The price store: a symbol-keyed, year-partitioned, append-mostly parquet layout.

    <price_dir>/close/<year>.parquet     symbol, date, close      -- one file per calendar year
    <price_dir>/events.parquet           symbol, date, kind, amount
    <price_dir>/adjclose_sample.parquet  symbol, date, adjclose   -- the vendor's own answer, sampled

WHY THIS IS NOT ONE FILE PER SYMBOL
-----------------------------------
The previous layout was `<price_dir>/<SYMBOL>.parquet`, holding that symbol's full history.
It is the obvious layout and it has one fatal property: every refresh rewrites every file.
Parquet is compressed binary, so a rewritten file is a whole new blob -- git cannot delta it
-- and a weekly refresh of this universe would therefore add its full 14 MB to the history
every week, forever, to record 250 new bars. That is the thing standing between "the repo is
the database" and a repo that has to be a database somewhere else.

Partitioning by YEAR fixes it: 2011..2025 are written once and never touched again, and only
the current year's file changes on a weekly run. Measured on this universe, the churn goes
from 14 MB/week to ~0.4 MB/week.

WHY THE STORE HOLDS `close` AND NOT `adjclose` -- THIS IS THE WHOLE REASON THE PARTITION WORKS
----------------------------------------------------------------------------------------------
Partitioning by year is worthless if the stored column is `adjclose`, and that is not obvious
until you write it down. The dividend adjustment is applied BACKWARD: a distribution with an
ex-date this week multiplies every earlier `adjclose` for that symbol by (1 - D/P). So one
dividend rewrites that symbol's entire history, a universe of ~90 payers has some ex-date
almost every week, and all 34 year files churn anyway. Time-partitioning an
already-retroactive column buys nothing.

So the store holds the two pieces the adjustment is MADE of, each of which is genuinely
append-only:

  close   split-adjusted, from the vendor. Immutable except on a split -- a handful of
          events per year across 130 ETFs, and one that touches a single symbol.
  events  the dividend and split log, keyed by ex-date. New rows only.

and `total_return()` reconstructs `adjclose` from them at load time. Total returns are still
what the optimiser sees -- nothing about the method changed, only where the adjustment is
performed.

MEASURED, NOT ASSUMED, and the test is `test_reconstruction_matches_the_vendors_adjclose`:
the reconstruction reproduces Yahoo's own `adjclose` to a relative error of ~1e-6 across the
universe. That is the precision of Yahoo's field itself, which ships as float32 (its values
are exactly representable at 24 bits and at no decimal count -- 761.78 arrives as
761.7800292969), so the residual is transport noise in the vendor's number rather than a
disagreement about the adjustment.

`adjclose_sample.parquet` is what keeps that claim checkable offline. Storing the full
`adjclose` column would reintroduce exactly the churn this layout exists to avoid, so what is
stored is 24 bars per symbol -- ~50 KB, including the FIRST bar, where the accumulated
adjustment is largest and a wrong factor is most visible. A reconstruction error is
systematic by nature (a bad factor propagates to every earlier bar), which is why a sample
has power here and would not if the errors were independent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLOSE_DIR = "close"
EVENTS_FILE = "events.parquet"
ADJCLOSE_SAMPLE_FILE = "adjclose_sample.parquet"

# Bars per symbol kept in adjclose_sample.parquet. 24 over a 15-year history is roughly one
# every 8 months; the first and last bar are always among them.
SAMPLE_BARS = 24

EVENT_KINDS = ("dividend", "split")


def close_dir(price_dir: Path) -> Path:
    return Path(price_dir) / CLOSE_DIR


def _year_paths(price_dir: Path) -> list[Path]:
    """Year partitions, oldest first. Sorted by the YEAR, not lexically."""
    d = close_dir(price_dir)
    if not d.exists():
        return []
    out = []
    for p in d.glob("*.parquet"):
        try:
            out.append((int(p.stem), p))
        except ValueError:
            continue
    return [p for _, p in sorted(out)]


def exists(price_dir: Path) -> bool:
    return bool(_year_paths(price_dir))


# ----------------------------------------------------------------------------- writing


def _split_by_year(long: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(y): chunk for y, chunk in long.groupby(long["date"].dt.year, sort=True)}


def _write_if_changed(path: Path, merged: pd.DataFrame) -> bool:
    """Write `merged` to `path` only if it differs from what is there. Returns whether it wrote.

    THIS IS WHAT MAKES THE PARTITION WORTH ANYTHING, and skipping it would be the quiet way to
    lose the whole benefit. The collector fetches each symbol's FULL history every time
    (`range=100y` -- there is no incremental endpoint), so a weekly refresh hands every year
    partition a chunk that is identical to what it already holds. Writing them all back would
    touch 34 files a week and, since parquet is compressed binary that git stores whole, add the
    entire store to the history every week -- exactly the behaviour the layout exists to avoid.

    The comparison is on the DataFrame, not on the file bytes. Parquet output happens to be
    reproducible for identical input today, but that is a property of the writer's version, not
    a guarantee, and "the historical years are frozen" must not rest on it.
    """
    if path.exists():
        old = pd.read_parquet(path)
        if old.shape == merged.shape and old.equals(merged):
            return False
    merged.to_parquet(path, index=False)
    return True


def write_close(price_dir: Path, frames: dict[str, pd.DataFrame]) -> list[int]:
    """Upsert each symbol's `close` series into the year partitions. Returns years CHANGED.

    Upsert, not append: a symbol already present in a year is REPLACED for that year, so
    re-fetching a symbol corrects a bad print rather than duplicating it. Only the years the
    incoming data actually covers are opened, and of those only the ones whose content actually
    moved are written -- see `_write_if_changed`. On a weekly refresh that is one file.
    """
    if not frames:
        return []
    d = close_dir(price_dir)
    d.mkdir(parents=True, exist_ok=True)

    incoming = pd.concat(
        [
            pd.DataFrame({"symbol": sym, "date": df.index, "close": df["close"].to_numpy(dtype=float)})
            for sym, df in frames.items()
        ],
        ignore_index=True,
    )
    syms = set(frames)
    changed = []
    for year, chunk in _split_by_year(incoming).items():
        path = d / f"{year}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            chunk = pd.concat([old[~old["symbol"].isin(syms)], chunk], ignore_index=True)
        # Sorted by symbol then date: it is the order every reader wants, it is what lets
        # parquet dictionary-encode the symbol column into almost nothing, and it is what makes
        # the unchanged-content comparison below a comparison of content rather than of order.
        chunk = chunk.sort_values(["symbol", "date"], ignore_index=True)
        if _write_if_changed(path, chunk):
            changed.append(year)
    return sorted(changed)


def _upsert_by_symbol(path: Path, incoming: pd.DataFrame, symbols: set[str], sort_on: list[str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        incoming = pd.concat([old[~old["symbol"].isin(symbols)], incoming], ignore_index=True)
    return _write_if_changed(path, incoming.sort_values(sort_on, ignore_index=True))


def write_events(price_dir: Path, events: dict[str, pd.DataFrame], symbols: set[str]) -> None:
    """Upsert the dividend/split log.

    `symbols` is passed separately from `events.keys()` on purpose: a symbol that pays nothing
    (GLD, UNG) has an EMPTY frame, and it still has to clear that symbol's old rows -- otherwise
    a record that disappears from the vendor's feed would survive forever in the store.
    """
    if not symbols:
        return
    parts = [df for df in events.values() if not df.empty]
    incoming = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame({"symbol": pd.Series(dtype="object"), "date": pd.Series(dtype="datetime64[ns]"),
                           "kind": pd.Series(dtype="object"), "amount": pd.Series(dtype=float)})
    )
    _upsert_by_symbol(Path(price_dir) / EVENTS_FILE, incoming, symbols, ["symbol", "date", "kind"])


def write_adjclose_sample(price_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    """Upsert `SAMPLE_BARS` (date, adjclose) pairs per symbol -- the vendor's own total return.

    The sample is taken at evenly spaced POSITIONS including both ends, so it is deterministic
    from the series alone and always contains the first bar, where the accumulated dividend
    adjustment is largest and therefore where the reconstruction check has the most power.
    """
    if not frames:
        return
    rows = []
    for sym, df in frames.items():
        n = len(df)
        pos = np.unique(np.linspace(0, n - 1, min(SAMPLE_BARS, n)).round().astype(int))
        s = df["adjclose"].iloc[pos]
        rows.append(pd.DataFrame({"symbol": sym, "date": s.index, "adjclose": s.to_numpy(dtype=float)}))
    _upsert_by_symbol(
        Path(price_dir) / ADJCLOSE_SAMPLE_FILE,
        pd.concat(rows, ignore_index=True),
        set(frames),
        ["symbol", "date"],
    )


def write(price_dir: Path, frames: dict[str, pd.DataFrame], events: dict[str, pd.DataFrame]) -> list[int]:
    """One flush: close partitions, the event log, and the audit sample. Returns years CHANGED."""
    years = write_close(price_dir, frames)
    write_events(price_dir, events, set(frames))
    write_adjclose_sample(price_dir, frames)
    return years


# ----------------------------------------------------------------------------- reading


def read_close(price_dir: Path, symbols=None) -> pd.DataFrame:
    """Wide `close` panel, date-indexed, one column per stored symbol. NaN where not traded.

    Every year partition is read, because a symbol's history spans all of them. That is ~740k
    rows for this universe and takes well under a second -- fewer file opens than the 132 the
    per-symbol layout needed.
    """
    paths = _year_paths(price_dir)
    if not paths:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    want = None if symbols is None else set(symbols)
    parts = []
    for p in paths:
        chunk = pd.read_parquet(p)
        if want is not None:
            chunk = chunk[chunk["symbol"].isin(want)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    wide.index.name = "date"
    wide.columns.name = None
    return wide.sort_index()


def read_events(price_dir: Path, kind: str | None = None) -> pd.DataFrame:
    path = Path(price_dir) / EVENTS_FILE
    cols = {"symbol": "object", "date": "datetime64[ns]", "kind": "object", "amount": float}
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})
    ev = pd.read_parquet(path)
    return ev[ev["kind"] == kind] if kind is not None else ev


def dividends_by_symbol(price_dir: Path) -> dict[str, pd.Series]:
    """{symbol: Series of cash amount indexed by ex-date}, sorted, ready for `total_return`."""
    ev = read_events(price_dir, kind="dividend")
    out = {}
    for sym, chunk in ev.groupby("symbol"):
        s = chunk.set_index("date")["amount"].sort_index()
        out[str(sym)] = s.groupby(level=0).sum()
    return out


def read_adjclose_sample(price_dir: Path) -> pd.DataFrame:
    path = Path(price_dir) / ADJCLOSE_SAMPLE_FILE
    if not path.exists():
        return pd.DataFrame(
            {"symbol": pd.Series(dtype="object"), "date": pd.Series(dtype="datetime64[ns]"),
             "adjclose": pd.Series(dtype=float)}
        )
    return pd.read_parquet(path)


def last_dates(price_dir: Path) -> dict[str, pd.Timestamp]:
    """{symbol: most recent stored bar}, for the fetch freshness check.

    Only the two newest partitions are read, not all of them. Two rather than one because on
    the first trading days of January the current year's file does not exist yet, and reading
    only it would report every symbol as having no data and refetch the entire universe. A
    symbol whose last bar is older than that reports nothing and is treated as stale -- which
    is the right answer, since a symbol that stopped printing is either delisted or broken and
    either way wants another look.
    """
    paths = _year_paths(price_dir)[-2:]
    if not paths:
        return {}
    long = pd.concat([pd.read_parquet(p, columns=["symbol", "date"]) for p in paths], ignore_index=True)
    return {str(k): v for k, v in long.groupby("symbol")["date"].max().items()}


def stored_symbols(price_dir: Path) -> set[str]:
    out: set[str] = set()
    for p in _year_paths(price_dir):
        out |= set(pd.read_parquet(p, columns=["symbol"])["symbol"].astype(str).unique())
    return out


# --------------------------------------------------------------------- the adjustment


def total_return(close: pd.Series, dividends: pd.Series | None) -> pd.Series:
    """Rebuild the dividend-adjusted (total-return) series from split-adjusted closes.

    Backward adjustment, which is the convention every vendor uses and the reason the stored
    `close` can be immutable while its adjusted form cannot: a distribution D with ex-date t_d
    scales every bar BEFORE t_d by (1 - D / P), where P is the close on the last trading day
    before the ex-date. The most recent bar therefore always equals its own close, and the
    divergence accumulates going back -- SPY's first 1993 bar comes out at 0.549x its close.

    `close` must be the symbol's FULL stored history, not a window. The factor at any bar
    depends on every dividend after it, so windowing first and adjusting second silently drops
    the adjustment for the tail of the window. Adjust, then window.
    """
    if dividends is None or len(dividends) == 0:
        return close.astype(float)
    idx = close.index
    factor = np.ones(len(close))
    px = close.to_numpy(dtype=float)
    for ex, amount in dividends.items():
        # First stored bar at or after the ex-date; everything strictly before it is adjusted.
        pos = int(idx.searchsorted(pd.Timestamp(ex), side="left"))
        if pos == 0:
            # Ex-date at or before the first stored bar: there is no prior close to scale by,
            # and no earlier bar to scale. Nothing to do -- not an error.
            continue
        prior = px[pos - 1]
        if not prior > 0:
            raise ValueError(f"non-positive close {prior} before ex-date {ex}")
        if amount >= prior:
            # A distribution at or above the price is not a distribution; it is a bad record,
            # and applying it would produce a non-positive or sign-flipped price history.
            raise ValueError(f"dividend {amount} >= prior close {prior} at ex-date {ex}")
        factor[:pos] *= 1.0 - amount / prior
    return close.astype(float) * factor


def read_total_return(price_dir: Path, symbols=None) -> pd.DataFrame:
    """Wide TOTAL-RETURN panel: `read_close` with the dividend adjustment applied per symbol.

    This is what the optimiser is estimated from. Each column is adjusted over its own full
    history before the columns are aligned, per `total_return`'s contract.
    """
    close = read_close(price_dir, symbols)
    if close.empty:
        return close
    divs = dividends_by_symbol(price_dir)
    return pd.DataFrame(
        {sym: total_return(close[sym].dropna(), divs.get(sym)) for sym in close.columns}
    ).sort_index()
