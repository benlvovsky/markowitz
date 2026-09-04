import json
import sys
from pathlib import Path

import numpy as np
import pytest

import os

PIPELINE = Path(__file__).resolve().parents[1]
ROOT = PIPELINE.parent

# MARKOWITZ_PRICE_DIR lets the suite read a price cache outside its own tree. `_mutate.py`
# runs a COPY of this package with one line changed, and re-downloading 132 symbols per
# mutant would make the harness a network test.
PRICES = Path(os.environ.get("MARKOWITZ_PRICE_DIR") or (PIPELINE / "data" / "prices"))

# MARKOWITZ_DATA_DIR points the artifact tests at a different output directory, and it
# exists for `tests/_mutate.py`: the mutation harness rebuilds the whole JSON set per
# mutant from a 25-symbol subset and needs the suite to read THAT, not the committed
# production artifacts. Without the redirect the harness would have to overwrite
# web/public/data to run, i.e. a test tool whose failure mode is corrupting the thing it
# is testing.
DATA = Path(os.environ.get("MARKOWITZ_DATA_DIR") or (ROOT / "web" / "public" / "data"))

sys.path.insert(0, str(PIPELINE))


def _load(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{path} missing -- run `python -u pipeline/build.py` first")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def manifest():
    return _load("manifest.json")


@pytest.fixture(scope="session")
def stats():
    s = _load("stats.json")
    s["mu_v"] = np.asarray(s["mu"], dtype=float)
    s["cov_v"] = np.asarray(s["cov"], dtype=float)
    s["index"] = {sym: i for i, sym in enumerate(s["symbols"])}
    return s


@pytest.fixture(scope="session")
def history():
    return _load("history.json")


@pytest.fixture(scope="session")
def frontiers(manifest):
    return {c["slug"]: _load(c["file"]) for c in manifest["caps"]}


def pytest_generate_tests(metafunc):
    """Parametrise over the shipped cap files by NAME, read from the manifest.

    Hardcoding ("cap100", "cap20", "cap10") would let `--caps` change in `build.py` while
    the suite kept testing three files that no longer exist -- and a skipped-because-missing
    test is indistinguishable from a passing one in a summary line.
    """
    if "cap_slug" not in metafunc.fixturenames:
        return
    path = DATA / "manifest.json"
    slugs = [c["slug"] for c in json.loads(path.read_text())["caps"]] if path.exists() else []
    metafunc.parametrize("cap_slug", slugs or ["__missing__"])
