"""Fetch, estimate, solve, write. The whole pipeline, one entry point.

    python -u pipeline/build.py --out web/public/data

Run from the repo root. Designed to be identical on a laptop and on a GitHub Actions
runner: the only difference is that the runner starts with an empty `pipeline/data/prices`
and therefore fetches everything, which is what makes the committed JSON reproducible from
the workflow rather than from local state.

THE RISK-FREE RATE IS NOT BAKED INTO THE FRONTIER, AND THAT IS THE KEY STRUCTURAL CHOICE
----------------------------------------------------------------------------------------
The efficient frontier does not depend on the risk-free rate at all -- it is the solution
to "minimise variance subject to a return target", and `rf` appears nowhere in that
problem. Only the TANGENCY portfolio and the capital market line depend on `rf`, and both
are recoverable from the frontier: the tangency point is the frontier point maximising
(ret - rf)/vol, and the CML is the ray from (0, rf) through it.

So `rf` is a browser-side slider, not a pipeline parameter, and moving it re-derives the
tangency portfolio and the CML instantly with no re-solve. The alternative -- one frontier
file per rf value -- would multiply the output by ten to express something the browser can
compute exactly. `rf_default` is written to the manifest only to seed the slider and to
fill the `sharpe` column of the asset table; nothing downstream depends on it, and the SPA
recomputes every Sharpe it displays from its own slider value.

The weight cap DOES require a re-solve (it is a constraint on the QP), so it is a pipeline
parameter and each cap gets its own file. Three caps ship by default -- 100%, 20%, 10% --
because the comparison between them is the honest summary of how much of the "optimal"
portfolio is estimation error rather than signal. See `frontier.py`.

ONE UNIVERSE PER OUTPUT DIRECTORY, STATED RATHER THAN ARRANGED
-------------------------------------------------------------
`--universe` selects a file from `pipeline/universes/`; `--out` says where its artifacts go.
They are separate flags with no relationship inferred between them, so building a second
universe is `--universe sp500 --out web/public/data/sp500` and the default invocation
reproduces exactly what ships today. The manifest records which universe file it came from,
which is what makes an output directory self-describing -- the alternative (deriving the
output path from the universe key) invents a layout convention before there is a second
universe to serve, and quietly overwrites one universe's artifacts with another's when the
derivation and the flag disagree.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch  # noqa: E402
import frontier as fr  # noqa: E402
import store  # noqa: E402
import universe as uni  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRICE_DIR = ROOT / "pipeline" / "data" / "prices"
DEFAULT_OUT = ROOT / "web" / "public" / "data"

# 13-week T-bill yield. Only seeds the SPA's slider; see the module docstring.
RF_TICKER = "^IRX"
RF_FALLBACK = 0.04


def fetch_rf() -> tuple[float, str]:
    try:
        df = fetch.parse(fetch._get(RF_TICKER, rng="1mo"), RF_TICKER)
        last = float(df["adjclose"].iloc[-1]) / 100.0
        if not 0.0 <= last < 0.25:
            raise ValueError(f"implausible {RF_TICKER} value {last}")
        # FOUR DECIMALS = ONE BASIS POINT, which is the SPA's rf slider step and the precision
        # its readout prints. At five the seeded default lands between two slider positions, so
        # the reader who touches the slider can never get back to it and the URL fragment then
        # carries an `rf=` forever. `web/src/config.test.ts` asserts the two agree.
        return round(last, 4), f"{RF_TICKER} close {df.index[-1].date()}"
    except Exception as exc:
        print(f"rf: falling back to {RF_FALLBACK} ({exc})", file=sys.stderr, flush=True)
        return RF_FALLBACK, f"fallback ({type(exc).__name__})"


def write_json(path: Path, payload) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    path.write_text(text)
    return len(text)


def cap_slug(cap: float) -> str:
    return f"cap{round(cap * 100):d}"


def cap_slugs(caps: list[float]) -> list[str]:
    """One slug per cap, ASSERTED DISTINCT.

    `cap_slug` rounds to whole percent, so 0.125 and 0.124 both come out `cap12`. Left
    unchecked that is silent and asymmetric: `frontiers[slug]` keeps only the second solve,
    while the manifest lists two caps pointing at the one file -- so the SPA offers two
    selectable caps that are the same frontier, and every cross-cap invariant compares a
    file with itself and passes. Raise instead; the caller wanted two frontiers.
    """
    slugs = [cap_slug(c) for c in caps]
    if len(set(slugs)) != len(slugs):
        clashes = sorted({s for s in slugs if slugs.count(s) > 1})
        raise SystemExit(
            f"--caps {caps} collide on the slug(s) {clashes}: caps are named to the nearest "
            "whole percent, so two caps within 1% of each other cannot both be built"
        )
    return slugs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    ap.add_argument(
        "--universe",
        default=uni.DEFAULT,
        help=f"a file in pipeline/universes/ ({', '.join(uni.available())})",
    )
    ap.add_argument("--start", default="2011-01-03", help="first day of the common window")
    ap.add_argument("--end", default=None)
    ap.add_argument("--caps", default="1.0,0.2,0.1", help="max weight per asset, comma separated")
    ap.add_argument("--points", type=int, default=60)
    ap.add_argument("--rf", default="auto", help="'auto' (^IRX) or a decimal like 0.042")
    ap.add_argument("--max-age-days", type=int, default=1)
    ap.add_argument("--min-coverage", type=float, default=0.98)
    ap.add_argument("--skip-fetch", action="store_true", help="use whatever is already on disk")
    # A subset build. `tests/_mutate.py` needs to rebuild the whole artifact set per mutant,
    # and 116 assets x 3 caps x 62 QPs per rebuild x 10 mutants is minutes of solving for a
    # question a 25-asset universe answers identically.
    ap.add_argument("--symbols", default=None, help="comma-separated subset of the universe")
    args = ap.parse_args(argv)

    caps = [float(c) for c in args.caps.split(",")]
    slugs = cap_slugs(caps)
    u = uni.load(args.universe)
    print(f"universe {u.key} ({u.name}): {len(u.symbols)} symbols from {u.path.name}", flush=True)

    symbols_wanted = tuple(args.symbols.split(",")) if args.symbols else u.symbols
    unknown = set(symbols_wanted) - set(u.symbols)
    if unknown:
        raise SystemExit(f"--symbols not in universe {u.key}: {sorted(unknown)}")

    if args.skip_fetch:
        on_disk = store.stored_symbols(args.price_dir)
        ok, failed = [s for s in symbols_wanted if s in on_disk], {}
        print(f"skip-fetch: {len(ok)}/{len(symbols_wanted)} symbols on disk", flush=True)
    else:
        print(f"fetching {len(symbols_wanted)} symbols -> {args.price_dir}", flush=True)
        ok, failed = fetch.download(symbols_wanted, args.price_dir, max_age_days=args.max_age_days)

    # Rounded on both paths, `auto` and explicit, for the reason in `fetch_rf`: this number seeds
    # a 1bp slider, and one that cannot be returned to is worse than one that is 1bp off.
    rf = (fetch_rf() if args.rf == "auto" else (round(float(args.rf), 4), "explicit"))
    rf_value, rf_source = rf
    print(f"rf_default = {rf_value:.4f} ({rf_source})", flush=True)

    panel, dropped = fetch.load_panel(
        ok, args.price_dir, start=args.start, end=args.end, min_coverage=args.min_coverage
    )
    print(
        f"panel: {panel.shape[1]} assets x {len(panel)} trading days "
        f"({panel.index.min().date()} -> {panel.index.max().date()}); "
        f"{len(dropped)} dropped, {len(failed)} failed to fetch",
        flush=True,
    )
    for sym, why in sorted(dropped.items()):
        print(f"  dropped {sym:<6} {why}", flush=True)

    est = fr.estimate(panel)
    print(f"ledoit-wolf shrinkage delta = {est.shrinkage:.4f} over {est.n_obs} returns", flush=True)

    assets = fr.asset_table(est, panel, rf_value, u.meta)
    symbols = list(est.mu.index)

    # ONE STAMP FOR THE WHOLE RUN, taken before anything is written, and carried by every one of
    # the six files. It is what lets the browser refuse a MIXED BUNDLE -- last week's cached
    # frontier against this week's covariance -- which is the failure mode of a weekly cron plus
    # HTTP caching, and which no other field can detect when the symbol set has not changed.
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    frontiers = {}
    for cap, slug in zip(caps, slugs):
        payload = {"generated_at": generated_at, **fr.build(est, rf=rf_value, cap=cap, n_points=args.points)}
        frontiers[slug] = payload
        n = payload["n_points"]
        tan = payload["frontier"][payload["max_sharpe_index"]]
        print(
            f"cap {cap:<4} -> {n:>3} frontier points "
            f"(pruned {payload['pruned']}, infeasible {len(payload['failed_targets'])}); "
            f"tangency ret {tan['ret']:.4f} vol {tan['vol']:.4f} sharpe {tan['sharpe']:.3f} "
            f"in {len(tan['w'])} holdings",
            flush=True,
        )

    monthly = fetch.monthly_returns(panel)

    # THE DENOMINATOR IS RETURNS, NOT BARS. `mean_historical_return` annualises with the
    # exponent 252/len(returns), and 3,939 price bars carry 3,938 returns. Using the bar
    # count here instead put `years` 0.025% high, which moved the annualised return of a
    # 19%/yr asset by 5e-5 -- far too small to see in the chart and far too big for the
    # `history` cross-check, which is how it was caught. Anything that annualises a
    # compounded growth from these artifacts must use this number.
    years = (len(panel) - 1) / fetch.TRADING_DAYS_PER_YEAR

    manifest = {
        "generated_at": generated_at,
        "window": {
            "start": str(panel.index.min().date()),
            "end": str(panel.index.max().date()),
            "requested_start": args.start,
            "trading_days": len(panel),
            "n_returns": len(panel) - 1,
            "years": round(years, 6),
        },
        "estimation": {
            "expected_returns": "compounded mean of daily simple returns, annualised at 252",
            "covariance": "Ledoit-Wolf shrinkage toward constant variance, annualised at 252",
            "shrinkage_delta": round(est.shrinkage, 6),
            "n_returns": est.n_obs,
        },
        "rf_default": rf_value,
        "rf_source": rf_source,
        # Which universe file this directory of artifacts is a statement about. Recorded so an
        # output directory is self-describing; see the module docstring on why the path is not
        # derived from it. `description` rides along because it is the ARGUMENT for the
        # selection, and an argument that stays in the TOML is an argument no reader ever sees.
        "universe": {
            "key": u.key,
            "name": u.name,
            "file": u.path.name,
            "description": u.description,
        },
        "benchmark": u.benchmark if u.benchmark in symbols else None,
        "groups": list(u.groups),
        # The label per group, so no consumer has to hold its own copy. A group map hardcoded in
        # the SPA is what made "a universe is a file" false everywhere the reader could see.
        "group_labels": dict(u.group_labels),
        "n_universe": len(symbols_wanted),
        "n_assets": len(symbols),
        "caps": [{"cap": c, "file": f"frontier_{s}.json", "slug": s} for c, s in zip(caps, slugs)],
        # THREE LISTS BECAUSE THERE ARE THREE DIFFERENT CLAIMS, and collapsing them would lose
        # the only one that carries an argument. `deliberate` is a judgement recorded in the
        # universe file with its evidence and is NOT part of `n_universe`; the other two are
        # measured this run, and both are failures of a symbol the universe did ask for.
        "excluded": {
            "fetch_failed": failed,
            "window_or_coverage": dropped,
            "deliberate": dict(u.excluded),
        },
        "assets": assets,
    }

    sizes = {}
    sizes["manifest.json"] = write_json(args.out / "manifest.json", manifest)
    sizes["stats.json"] = write_json(
        args.out / "stats.json",
        {
            "generated_at": generated_at,
            "symbols": symbols,
            "mu": [round(float(x), 8) for x in est.mu.to_numpy()],
            "cov": [[round(float(x), 9) for x in row] for row in est.cov.to_numpy()],
        },
    )
    sizes["history.json"] = write_json(
        args.out / "history.json",
        {
            "generated_at": generated_at,
            "freq": "monthly",
            # The chained product of these returns equals the daily panel's total growth
            # exactly, because the series is anchored at `anchor` rather than at the first
            # month end -- see fetch.monthly_returns. `years` is the DAILY window's length
            # in 252-day years, and is the exponent to annualise a compounded growth from
            # this series so that it reproduces the optimiser's `mu`.
            "anchor": str(panel.index.min().date()),
            "years": round(years, 6),
            "partial_months": ["first", "last"],
            "dates": [d.strftime("%Y-%m") for d in monthly.index],
            "symbols": symbols,
            "returns": {s: [round(float(x), 6) for x in monthly[s].to_numpy()] for s in symbols},
        },
    )
    for slug, payload in frontiers.items():
        name = f"frontier_{slug}.json"
        sizes[name] = write_json(args.out / name, payload)

    total = sum(sizes.values())
    print(f"\nwrote {len(sizes)} files to {args.out} ({total / 1024:.0f} KiB uncompressed)", flush=True)
    for name, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {size / 1024:>8.1f} KiB  {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
