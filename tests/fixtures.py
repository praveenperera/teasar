"""B1 fixture access for the acceptance tests (spec part A §6).

The packed block for z 8068-9600 is built once from the zarr box and cached under the repo's ``_scratch``, so tests
never hold the 3.6 GB unpacked box in memory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def env_dir(name: str) -> Path | None:
    """The directory an environment variable names, or None when it is unset"""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def need(name: str, what: str) -> Path:
    """The directory `name` names. When it is unset this skips the calling test under pytest and stops a script"""
    path = env_dir(name)
    if path is not None:
        return path

    message = f"set {name} to {what}"
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(message)
    sys.exit(message)


# spec part A §6's data/ directory, with the inputs and per-slab fixtures; the fixture set is not part of this repo, so
# there is no default, and tests that need it skip when TEASAR_DATA is unset
DATA = env_dir("TEASAR_DATA")
DATA_WHAT = "the fixture set's data/ directory"
NO_DATA = f"set TEASAR_DATA to {DATA_WHAT}"
# fixture subdirectories of data/
B1 = "fixtures/b1"
B1_DS4 = "fixtures/b1_ds4"


def data(*parts: str) -> Path:
    """A path under the fixture set's data/ directory (TEASAR_DATA)"""
    return need("TEASAR_DATA", DATA_WHAT).joinpath(*parts)


CACHE = Path(__file__).resolve().parent.parent / "_scratch/teasar-impl"
Z0, Z1 = 8068, 9600
YX = 1536
SHAPE = (Z1 - Z0, YX, YX)

# keyword arguments for each settings set (spec part A §2.4): PUBLISHED and VILLA for prepare_from_block,
# SKEL_* for skeletonize, EXTRACT_* for extract, and PREP_ARGS for prepare_labels' positional values
PUBLISHED = dict(settings="published_0211", downsample=2, min_count=50, max_count=80000)
VILLA = dict(settings="villa_default", downsample=4, min_count=6, max_count=10000)
SKEL_PUBLISHED = dict(settings="published_0211", anisotropy=(2, 2, 2), dust_threshold=49)
SKEL_VILLA = dict(settings="villa_default", anisotropy=(4, 4, 4), dust_threshold=5)
EXTRACT_PUBLISHED = {**PUBLISHED, **SKEL_PUBLISHED}
EXTRACT_VILLA = {**VILLA, **SKEL_VILLA}
PREP_ARGS = {"published_0211": (2, 50, 80000), "villa_default": (4, 6, 10000)}
# villa's h ribbon depth for each set; vertical slabs are 4 thick in both
H_DEPTH = {"published_0211": 2, "villa_default": 4}

# spec part A §2.5: a call's pooled voxels, summed over its slabs
CALL_POOLED = 1 << 26


def box():
    import zarr

    return zarr.open_array(str(data("b1box.zarr/0")), mode="r")


def block() -> np.ndarray:
    """The B1 block: (1532, 1536, 192) uint8, bit-packed along x, most significant bit first."""
    # the cache is built from TEASAR_DATA, so a run without it skips even when the cache exists
    data()
    path = CACHE / "b1_block.npy"
    if path.exists():
        return np.load(path)
    arr = box()
    out = np.empty((Z1 - Z0, YX, YX // 8), np.uint8)
    for z in range(Z0, Z1, 128):
        hi = min(z + 128, Z1)
        out[z - Z0 : hi - Z0] = np.packbits(arr[z:hi, :YX, :YX] > 0, axis=-1, bitorder="big")
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, out)
    tmp.replace(path)
    return out


def slab_keys(root: str = B1) -> list[str]:
    """The B1 slab keys in index order: 384 at published_0211, or 144 with root=B1_DS4 (villa_default)."""
    return list(json.loads(data(root, "index.json").read_text())["slabs"])


def listed_slab_keys(root: str = B1) -> list[str]:
    """slab_keys for collection-time parameter lists: empty when TEASAR_DATA is unset, where the tests skip"""
    return slab_keys(root) if DATA is not None else []


def slab_spec(key: str, settings: str = "published_0211") -> tuple[str, int, int]:
    """A fixture key as a block-local `(kind, lo, hi)` slab, with the named set's slab depth."""
    kind, pos = key.split(":")
    pos = int(pos)
    if kind == "h":
        return ("h", pos - Z0, pos - Z0 + H_DEPTH[settings])
    return (kind, pos, pos + 4)


def pooled_voxels(spec: tuple[str, int, int], shape=SHAPE, ds: int = 2) -> int:
    """Pooled voxels of one slab of a block with full-resolution shape (Z, Y, X); 0 for a thin slab."""
    kind, lo, hi = spec
    z, y, x = shape
    dims = {"h": (hi - lo, y, x), "vy": (z, hi - lo, x), "vx": (z, y, hi - lo)}[kind]
    n = 1
    for d in dims:
        n *= d // ds
    return n


def baseline_partition(items: list, pooled, limit: int = CALL_POOLED) -> list[list]:
    """Spec part A §2.5's baseline partition: items in list order, packed greedily into calls, starting a new call
    whenever the next item would take the call's pooled total (`pooled(item)`) over `limit`."""
    calls, cur, total = [], [], 0
    for item in items:
        n = pooled(item)
        if cur and total + n > limit:
            calls.append(cur)
            cur, total = [], 0
        cur.append(item)
        total += n
    return calls + [cur] if cur else calls


def slab_calls(specs: list, shape=SHAPE, ds: int = 2) -> list[list]:
    """The baseline partition of block slabs `(kind, lo, hi)`, as extract and prepare_from_block take them."""
    return baseline_partition(specs, lambda s: pooled_voxels(s, shape, ds))


def label_calls(labels: list) -> list[list]:
    """The baseline partition of pooled label arrays, as skeletonize takes them."""
    return baseline_partition(labels, lambda a: a.size)


def unpack_slab(blk: np.ndarray, spec: tuple[str, int, int], x_size: int = YX) -> np.ndarray:
    """The slab's dense uint8 array, unpacked from the block."""
    kind, lo, hi = spec
    if kind == "h":
        return np.unpackbits(blk[lo:hi], axis=-1, bitorder="big", count=x_size)
    if kind == "vy":
        return np.unpackbits(blk[:, lo:hi], axis=-1, bitorder="big", count=x_size)
    b0, b1 = lo // 8, (hi + 7) // 8
    bits = np.unpackbits(blk[:, :, b0:b1], axis=-1, bitorder="big")
    return np.ascontiguousarray(bits[:, :, lo - 8 * b0 : hi - 8 * b0])


def fixture(key: str, root: str = B1):
    return np.load(data(root, f"{key}.npz"))


def fixture_labels(key: str, root: str = B1) -> np.ndarray:
    return fixture(key, root)["labels"]


def size_filter_fixture(settings: str):
    """A3's size-filter boundary slab for one set: `binary`, villa's `labels` and the four pooled `counts`."""
    return np.load(data("fixtures/a3", f"size_filter_{settings}.npz"))


def reference_labels(binary: np.ndarray, ds: int, min_count: int, max_count: int) -> np.ndarray:
    """villa's prepare_cc_labels recomputed independently with scipy and numpy."""
    from scipy import ndimage

    lab, _ = ndimage.label(binary != 0)
    lab = lab.astype(np.uint32)
    lab = lab[tuple(slice(0, s - s % ds) for s in lab.shape)]
    z, y, x = (s // ds for s in lab.shape)
    pooled = lab.reshape(z, ds, y, ds, x, ds).max(axis=(1, 3, 5))
    counts = np.bincount(pooled.ravel())
    keep = (counts >= min_count) & (counts < max_count)
    keep[0] = False
    return np.where(keep[pooled], pooled, 0).astype(np.uint32)


# the host threads timed runs use: a Device reads TEASAR_CPU_THREADS itself (default: the performance cores), and a
# CpuDevice gets the same count (0, all cores, when unset); part B B9 compares N host threads with N Kimimaro
# workers, so every timing prints last_stats() with the count it ran on
CPU_THREADS = int(os.environ.get("TEASAR_CPU_THREADS", "0"))


def open_device(kind: str):
    """A fresh device of `kind` ("gpu" or "cpu")."""
    import teasar as tm

    return tm.Device() if kind == "gpu" else tm.CpuDevice(threads=CPU_THREADS)


class StatsSum:
    """`last_stats()` summed over the calls of one pass; `host_threads` is the largest any call used."""

    def __init__(self):
        self.s = dict.fromkeys(
            ("wall_s", "gpu_busy_s", "host_threads", "cpu_pieces", "cpu_voxels", "gpu_pieces", "gpu_voxels",
             "cpu_busy_s"), 0)

    def add(self, dev):
        for k, v in dev.last_stats().items():
            self.s[k] = max(self.s[k], v) if k == "host_threads" else self.s[k] + v

    def __str__(self):
        s = self.s
        return (
            f"wall {s['wall_s']:.3f} s, gpu busy {s['gpu_busy_s']:.3f} s, {s['host_threads']} host threads, "
            f"cpu {s['cpu_pieces']} pieces / {s['cpu_voxels']} voxels / {s['cpu_busy_s']:.3f} thread-s, "
            f"gpu {s['gpu_pieces']} pieces / {s['gpu_voxels']} voxels"
        )
