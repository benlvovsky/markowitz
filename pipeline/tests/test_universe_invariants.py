"""The universe loader's validation, which is the only thing in the pipeline whose job is to FAIL.

`universe.load` has one purpose beyond reading TOML: every mistake it can catch has a silent
failure mode further down. An asset whose `group` is not declared gets no filter button and no
label but is still held by the optimiser. A benchmark that is not in `[assets]` makes the SPA's
growth comparison fetch a symbol nobody downloaded. An exclusion with no reason removes an
instrument and leaves no way to tell whether that was a data problem or an oversight.

So the tests here are almost all NEGATIVE: they write a deliberately broken universe file and
require the raise. That is the opposite shape from the rest of this suite, which reads shipped
artifacts -- and it has to be, because a validator can only be tested with input the shipped
data by definition does not contain. The one positive test reads the real file, so the loader
and `etf_global.toml` cannot drift apart.

Written to `tmp_path` and loaded by path rather than by key, via `_load_text`, so nothing here
can touch `pipeline/universes/`.
"""
from __future__ import annotations

import pytest

import universe as uni

MINIMAL = """
key = "t"
name = "Test"
benchmark = "AAA"
groups = ["alpha", "beta"]
group_labels = { alpha = "Alpha", beta = "Beta" }

[assets]
AAA = { name = "A fund", asset_class = "equity", group = "alpha" }
BBB = { name = "B fund", asset_class = "bond", group = "beta" }
"""


def _load_text(tmp_path, text: str, monkeypatch) -> uni.Universe:
    """Load `text` as a universe, with UNIVERSE_DIR pointed at a throwaway directory."""
    (tmp_path / "t.toml").write_text(text)
    monkeypatch.setattr(uni, "UNIVERSE_DIR", tmp_path)
    return uni.load("t")


# ------------------------------------------------------------------ the shipped file loads


def test_the_shipped_universe_loads_and_is_internally_consistent():
    """Read the real file, not a fixture: the loader and the universe it exists to load must
    not drift apart, and every fixture below is only evidence about the fixture."""
    u = uni.load()
    assert u.symbols, "no symbols"
    assert len(set(u.symbols)) == len(u.symbols)
    assert set(u.meta) == set(u.symbols)
    for s in u.symbols:
        assert u.group_of(s) in u.groups, s
        assert u.name_of(s) and u.asset_class_of(s), s
    assert u.benchmark in u.meta
    assert set(u.excluded).isdisjoint(u.symbols)
    assert u.description, "the argument for the selection is empty"


def test_every_declared_group_gets_a_label_and_no_group_is_left_unnamed():
    """`group_labels` is complete by construction, so no consumer needs a fallback. The SPA's
    filter buttons read it directly; a missing key would render an empty button."""
    u = uni.load()
    assert set(u.group_labels) == set(u.groups)
    assert all(v.strip() for v in u.group_labels.values())


def test_a_group_without_a_label_in_the_file_gets_a_derived_one(tmp_path, monkeypatch):
    """Adding a universe must not require adding labels: the derived label is legible enough to
    ship. It is also plainly not hand-written, so an unlabelled group looks unfinished."""
    u = _load_text(tmp_path, MINIMAL.replace('group_labels = { alpha = "Alpha", beta = "Beta" }', ""), monkeypatch)
    assert u.group_labels == {"alpha": "Alpha", "beta": "Beta"}
    assert uni.derived_label("real_asset_fx") == "Real asset fx"


# ------------------------------------------------------------------ every raise, exercised


def test_an_asset_whose_group_is_not_declared_is_rejected(tmp_path, monkeypatch):
    """The silent version: the asset is optimised over and held, but has no colour, no filter
    button and no label -- it simply is not in the chart's legend while being in the portfolio."""
    bad = MINIMAL.replace('group = "beta" }', 'group = "gamma" }')
    with pytest.raises(ValueError, match="not in"):
        _load_text(tmp_path, bad, monkeypatch)


def test_an_asset_missing_a_required_field_is_rejected(tmp_path, monkeypatch):
    for field in uni.REQUIRED_ASSET_FIELDS:
        bad = MINIMAL.replace(f'{field} = "bond", ', "").replace(f', {field} = "beta"', "")
        bad = bad.replace(f'{field} = "B fund", ', "")
        assert bad != MINIMAL, field
        with pytest.raises(ValueError, match=field):
            _load_text(tmp_path, bad, monkeypatch)


def test_a_benchmark_outside_the_assets_is_rejected(tmp_path, monkeypatch):
    """The SPA draws the selected portfolio against this symbol's growth curve, which it reads
    from history.json -- so a benchmark nobody downloaded is a missing series in the browser."""
    with pytest.raises(ValueError, match="benchmark"):
        _load_text(tmp_path, MINIMAL.replace('benchmark = "AAA"', 'benchmark = "ZZZ"'), monkeypatch)


def test_a_symbol_both_included_and_excluded_is_rejected(tmp_path, monkeypatch):
    """Contradictory intent. It would also make the drop accounting wrong in a way that reads
    as an off-by-one somewhere else entirely."""
    bad = MINIMAL + '\n[excluded]\nBBB = "some reason"\n'
    with pytest.raises(ValueError, match="both in"):
        _load_text(tmp_path, bad, monkeypatch)


def test_an_exclusion_with_no_reason_is_rejected(tmp_path, monkeypatch):
    """The evidence IS the record. An exclusion with an empty reason removes an instrument from
    the universe and leaves nothing to distinguish a measured data fault from an oversight."""
    with pytest.raises(ValueError, match="no reason"):
        _load_text(tmp_path, MINIMAL + '\n[excluded]\nCCC = "   "\n', monkeypatch)


def test_a_label_for_an_undeclared_group_is_rejected(tmp_path, monkeypatch):
    """A typo in a label key is silent otherwise: the group it was meant for falls back to the
    derived label and the typo'd entry is simply ignored."""
    bad = MINIMAL.replace('beta = "Beta" }', 'betta = "Beta" }')
    with pytest.raises(ValueError, match="group_labels"):
        _load_text(tmp_path, bad, monkeypatch)


def test_a_file_with_no_assets_or_no_groups_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no .assets."):
        _load_text(tmp_path, 'key = "t"\ngroups = ["alpha"]\n', monkeypatch)
    with pytest.raises(ValueError, match="no .groups."):
        _load_text(tmp_path, MINIMAL.replace('groups = ["alpha", "beta"]', ""), monkeypatch)


def test_a_missing_universe_names_the_ones_that_exist(tmp_path, monkeypatch):
    """A typo'd `--universe` is the most likely operator error, and the fix is the list."""
    monkeypatch.setattr(uni, "UNIVERSE_DIR", tmp_path)
    (tmp_path / "real.toml").write_text(MINIMAL)
    with pytest.raises(SystemExit, match="real"):
        uni.load("nope")
