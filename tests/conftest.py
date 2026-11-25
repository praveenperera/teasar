"""Device fixtures: the CPU reference always, the GPU backend only with TEASAR_GPU=1.

GPU runs are opt-in, so a plain local ``pytest`` never opens the GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures  # noqa: E402
import teasar as tm  # noqa: E402

GPU = os.environ.get("TEASAR_GPU") == "1"
# TEASAR_DEVICES=gpu keeps CPU-only tests out of a GPU queue job
DEVICES = os.environ.get("TEASAR_DEVICES", "cpu,gpu" if GPU else "cpu").split(",")

_open: dict[str, object] = {}


def open_device(kind: str):
    """A fresh device of `kind`."""
    return fixtures.open_device(kind)


@pytest.fixture(params=DEVICES, scope="session")
def dev(request):
    """One shared device per kind for the session."""
    if request.param not in _open:
        _open[request.param] = open_device(request.param)
    return _open[request.param]


@pytest.fixture(params=DEVICES)
def fresh(request):
    """A device of each kind that the test may close or reconfigure."""
    d = open_device(request.param)
    yield d
    d.close()


@pytest.fixture(scope="session")
def block():
    import fixtures

    return fixtures.block()


# tests that read the fixture set skip, naming the variable, when TEASAR_DATA is unset
needs_data = pytest.mark.skipif(fixtures.DATA is None, reason=fixtures.NO_DATA)


def needs_testing_build():
    return pytest.mark.skipif(not getattr(tm._core, "TESTING", False), reason="needs the `testing` feature build")
