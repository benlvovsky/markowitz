"""Universe loading: which instruments a build is about, read from `pipeline/universes/*.toml`.

A UNIVERSE IS A SELECTION, NOT A DATASET -- THAT SEPARATION IS THE POINT OF THIS MODULE
---------------------------------------------------------------------------------------
The price store (`store.py`) is keyed by SYMBOL and knows nothing about universes. A universe
file is a list of symbols with labels. So two universes that overlap -- an S&P 500 list and a
US sector-ETF list both wanting SPY, a European list and a global list both wanting VGK --
fetch and store that symbol ONCE, and adding a universe costs only the symbols it introduces.

The alternative, which this replaced, was a single `UNIVERSE` dict in this file. It works
until there is a second universe, at which point the dict is either duplicated or grown into
a dict of dicts, and either way the reasoning about WHY those instruments (which is most of
what a universe file contains) has nowhere to live per-universe. Now adding a universe is
adding one file and nothing else, and `git log pipeline/universes/` reads as a record of which
instruments were considered when.

Universe files are TOML read with the standard library's `tomllib` -- no dependency, comments
allowed, and inert. Inert matters: a universe is data, and the person adding one should not be
able to execute anything by doing so. TOML's inline tables make it one line per symbol, which
keeps a 132-symbol file legible.

WHAT IS NOT IN A UNIVERSE FILE
------------------------------
Start dates. History depth decides which of these instruments survives, but it is measured
from the prices (`fetch.load_panel`) and reported in the manifest with the reason for every
drop -- never asserted in the universe. A hand-recorded inception date is a fact that goes
stale silently; a measured one cannot.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

UNIVERSE_DIR = Path(__file__).resolve().parent / "universes"

DEFAULT = "etf_global"

# Every asset must carry all three, because all three are consumed downstream: `name` in the
# tooltip and table, `asset_class` in the filter row, `group` as the chart's colour channel.
# A missing one would surface as a KeyError deep in `frontier.asset_table` after the solve.
REQUIRED_ASSET_FIELDS = ("name", "asset_class", "group")


@dataclass(frozen=True)
class Universe:
    key: str
    name: str
    benchmark: str | None
    groups: tuple[str, ...]
    symbols: tuple[str, ...]
    # symbol -> (name, asset_class, group). The tuple form is what `frontier.asset_table`
    # takes, and it is the whole per-asset payload, so it is stored once rather than rebuilt.
    meta: dict[str, tuple[str, str, str]]
    description: str
    excluded: dict[str, str]
    path: Path

    def group_of(self, symbol: str) -> str:
        return self.meta[symbol][2]

    def name_of(self, symbol: str) -> str:
        return self.meta[symbol][0]

    def asset_class_of(self, symbol: str) -> str:
        return self.meta[symbol][1]


def available() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in UNIVERSE_DIR.glob("*.toml") if not p.stem.startswith("_")))


def load(key: str = DEFAULT) -> Universe:
    """Read and VALIDATE a universe file. Every failure below is raised, not warned about.

    The validation is not defensive boilerplate: this file decides what the frontier is a
    statement about, and each of these mistakes has a silent failure mode. An asset whose
    `group` is not in `groups` gets no colour and vanishes from the chart's filter row while
    still being held by the optimiser. A benchmark that is not in `assets` makes the SPA's
    growth comparison fetch a symbol nobody downloaded. A duplicate symbol would be counted
    twice in `n_universe` and once everywhere else, which is the kind of off-by-one that gets
    diagnosed as a bug in the drop accounting.
    """
    path = UNIVERSE_DIR / f"{key}.toml"
    if not path.exists():
        raise SystemExit(f"no universe {key!r} in {UNIVERSE_DIR} (have: {', '.join(available())})")
    doc = tomllib.loads(path.read_text())

    assets = doc.get("assets") or {}
    if not assets:
        raise ValueError(f"{path}: no [assets]")
    groups = tuple(doc.get("groups") or ())
    if not groups:
        raise ValueError(f"{path}: no `groups`")

    meta: dict[str, tuple[str, str, str]] = {}
    for symbol, row in assets.items():
        missing = [f for f in REQUIRED_ASSET_FIELDS if not row.get(f)]
        if missing:
            raise ValueError(f"{path}: asset {symbol} is missing {', '.join(missing)}")
        if row["group"] not in groups:
            raise ValueError(f"{path}: asset {symbol} has group {row['group']!r}, not in {list(groups)}")
        meta[symbol] = (row["name"], row["asset_class"], row["group"])

    # tomllib raises on a duplicate key, so this can only fire if `assets` was built some other
    # way -- but the assertion is cheap and the failure it guards is a miscounted universe.
    if len(meta) != len(assets):
        raise ValueError(f"{path}: duplicate symbols in [assets]")

    benchmark = doc.get("benchmark")
    if benchmark is not None and benchmark not in meta:
        raise ValueError(f"{path}: benchmark {benchmark!r} is not in [assets]")

    overlap = set(doc.get("excluded") or {}) & set(meta)
    if overlap:
        raise ValueError(f"{path}: {sorted(overlap)} are both in [assets] and [excluded]")

    return Universe(
        key=doc.get("key", key),
        name=doc.get("name", key),
        benchmark=benchmark,
        groups=groups,
        symbols=tuple(meta),
        meta=meta,
        description=(doc.get("description") or "").strip(),
        excluded={k: str(v).strip() for k, v in (doc.get("excluded") or {}).items()},
        path=path,
    )
