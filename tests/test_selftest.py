"""GPU self-tests of part B's building blocks: the 64-bit atomic minimum of spec part B §3.2 (unverified on this Mac
until this runs) and the float rules (correctly rounded square roots and divisions)."""

from __future__ import annotations

import os

import pytest

import teasar as tm

pytestmark = pytest.mark.skipif(os.environ.get("TEASAR_GPU") != "1", reason="needs the GPU")


@pytest.mark.parametrize("name", ["atomic64", "float_rules"])
def test_selftest(name):
    report = tm.Device()._selftest(name)
    print(report)
    assert report.startswith(name)
