#!/usr/bin/env python3
"""Checks an installed teasar without the fixture set, as CI does after installing each wheel and the sdist.

It checks the version, that the build has the backend its platform should have, and that `extract` on a small synthetic
block gives the same non-empty skeletons on repeated runs in both target orders. On a Mac whose GPU opens, the Metal
backend must equal the CPU reference bit for bit.

usage: install_smoke.py [VERSION] [metal|cpu]
"""

import sys

import numpy as np

import teasar

SETTINGS = dict(settings="published_0211", downsample=2, min_count=50, max_count=80000, anisotropy=(2, 2, 2),
                dust_threshold=49)


def block() -> tuple[np.ndarray, int]:
    """A bit-packed (Z, Y, ceil(X / 8)) block holding a cross in the z-x plane that runs through every y"""
    z, y, x = 24, 16, 64
    b = np.zeros((z, y, x), bool)
    b[2:22, :, 28:36] = True
    b[9:15, :, 4:60] = True
    return np.packbits(b, axis=-1, bitorder="big"), x


def arrays(packed) -> list[np.ndarray]:
    return [np.asarray(getattr(packed, f)) for f in ("ids", "v_off", "e_off", "vertices", "edges", "radii")]


def run(dev, order: str) -> list[list[np.ndarray]]:
    data, x_size = block()
    slabs = [("vy", 0, 4), ("vy", 4, 8), ("vy", 8, 12)]
    return [arrays(p) for p in dev.extract(data, x_size, slabs, target_order=order, **SETTINGS)]


def same(a: list[list[np.ndarray]], b: list[list[np.ndarray]]) -> bool:
    return len(a) == len(b) and all(
        len(x) == len(y) and all(u.dtype == v.dtype and np.array_equal(u, v) for u, v in zip(x, y, strict=True))
        for x, y in zip(a, b, strict=True)
    )


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else None
    backend = sys.argv[2] if len(sys.argv) > 2 else None
    print("teasar", teasar.__version__, "from", teasar.__file__)
    if version is not None:
        assert teasar.__version__ == version, (teasar.__version__, version)

    ok, msg = teasar.check_available()
    print("check_available:", ok, msg)
    # a runner may have no usable GPU, but a Metal build always tries one, so it never reports a build without one
    if backend == "metal":
        assert "no GPU backend" not in msg, f"this macOS build lacks the metal feature: {msg}"
    elif backend == "cpu":
        assert not ok and "no GPU backend" in msg, msg

    cpu = teasar.CpuDevice(threads=2)
    for order in ("best", "kimimaro"):
        first = run(cpu, order)
        vertices = sum(len(s[3]) for s in first)
        assert vertices > 0, f"no skeleton vertices in target_order={order!r}"
        assert same(first, run(cpu, order)), f"two CPU runs differ in target_order={order!r}"
        print(f"cpu {order}: {len(first)} slabs, {vertices} vertices, repeatable")
        if ok:
            assert same(first, run(teasar.Device(), order)), f"Metal differs from the CPU reference in {order!r}"
            print(f"metal {order}: equal to the CPU reference")


if __name__ == "__main__":
    main()
