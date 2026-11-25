"""Acceptance tests A1-A4 and A6 (spec part A §7), on every device the run enables."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys

import numpy as np
import pytest

import fixtures as F
import teasar as tm
from conftest import needs_data, needs_testing_build

# empty without TEASAR_DATA, where every test that uses them skips
KEYS = F.listed_slab_keys()
SPECS = [F.slab_spec(k) for k in KEYS]


P = "published_0211"
V = "villa_default"


def prep(dev, binary, settings=P):
    """prepare_labels with the named set's values."""
    return dev.prepare_labels(binary, *F.PREP_ARGS[settings], settings=settings)


def from_block(dev, blk, specs, kw=F.PUBLISHED, x_size=F.YX):
    """prepare_from_block over the baseline partition of `specs`."""
    shape = (blk.shape[0], blk.shape[1], x_size)
    out = []
    for part in F.slab_calls(specs, shape, kw["downsample"]):
        out += dev.prepare_from_block(blk, x_size, part, **kw)
    return out


def labels_digest(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def packed_nbytes(p) -> int:
    return sum(getattr(p, k).nbytes for k in ("ids", "v_off", "e_off", "vertices", "edges", "radii"))


# ---- packaging ----


@pytest.mark.skipif(sys.platform != "darwin", reason="the Metal backend is built on macOS only")
def test_macos_build_has_metal():
    """A macOS build has the Metal backend: pyproject.toml enables the metal feature, so a plain pip install gets it"""
    ok, msg = tm.check_available()
    assert "no GPU backend" not in msg, f"this macOS build lacks the metal feature: {msg}"
    assert ok, msg


# ---- A1 ----


@needs_data
def test_a1_prepare_from_block_matches_villa(dev, block):
    out = from_block(dev, block, SPECS)
    bad = {}
    for key, got in zip(KEYS, out):
        want = F.fixture_labels(key)
        assert got.dtype == np.uint32 and got.flags.c_contiguous
        assert got.shape == want.shape, key
        n = int((got != want).sum())
        if n:
            bad[key] = n
    assert not bad, f"mismatched voxels per slab: {bad}"


@needs_data
def test_a1_prepare_labels_matches_villa(dev, block):
    bad = {}
    for key, spec in zip(KEYS, SPECS):
        got = prep(dev, F.unpack_slab(block, spec))
        want = F.fixture_labels(key)
        assert got.dtype == np.uint32 and got.shape == want.shape, key
        n = int((got != want).sum())
        if n:
            bad[key] = n
    assert not bad, f"mismatched voxels per slab: {bad}"


@needs_data
def test_a1_bool_input_matches_uint8(dev, block):
    dense = F.unpack_slab(block, F.slab_spec("h:8500"))
    np.testing.assert_array_equal(prep(dev, dense.astype(bool)), prep(dev, dense))


# ---- A2 (labels; the skeleton half needs part B) ----


@needs_data
def test_a2_shuffle_and_split_change_nothing(dev, block):
    base = from_block(dev, block, SPECS)
    for per_call in (1, 7):
        for i in range(0, 40, per_call):
            got = dev.prepare_from_block(block, F.YX, SPECS[i : min(i + per_call, 40)], **F.PUBLISHED)
            for a, b in zip(got, base[i:]):
                np.testing.assert_array_equal(a, b)
    # shuffle within the baseline partition's first call
    first = F.slab_calls(SPECS)[0]
    order = list(range(len(first)))
    random.Random(7).shuffle(order)
    got = dev.prepare_from_block(block, F.YX, [first[i] for i in order], **F.PUBLISHED)
    for g, i in zip(got, order):
        np.testing.assert_array_equal(g, base[i])


@needs_data
@needs_testing_build()
def test_a2_internal_batching_changes_nothing(fresh, block):
    specs = SPECS[:8] + SPECS[192:200] + SPECS[288:296]
    want = fresh.prepare_from_block(block, F.YX, specs, **F.PUBLISHED)
    for cells in (1, 300_000, 2_000_000):
        fresh._set_batch_cells(cells)
        got = fresh.prepare_from_block(block, F.YX, specs, **F.PUBLISHED)
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)


# ---- A3 ----


@pytest.mark.skipif(os.environ.get("TEASAR_GPU") != "1", reason="GPU only")
def test_check_available_names_the_gpu():
    ok, reason = tm.check_available()
    assert ok, reason
    assert reason.startswith("Apple")


def test_a3_all_zero_slab(dev):
    z = np.zeros((6, 9, 10), np.uint8)
    got = prep(dev, z)
    assert got.shape == (3, 4, 5) and got.dtype == np.uint32 and not got.any()
    blk = np.packbits(z, axis=-1, bitorder="big")
    out = dev.prepare_from_block(blk, 10, [("h", 0, 2), ("vy", 2, 6)], **F.PUBLISHED)
    assert out[0].shape == (1, 4, 5) and not out[0].any()
    assert out[1].shape == (3, 2, 5) and not out[1].any()


def test_a3_all_zero_slab_extract_is_empty(dev):
    blk = np.zeros((8, 8, 1), np.uint8)
    assert_empty_packed(dev.extract(blk, 8, [("h", 0, 4)], **F.EXTRACT_VILLA)[0])


def assert_empty_packed(p):
    assert p.ids.dtype == np.uint32 and p.ids.shape == (0,)
    assert p.v_off.tolist() == [0] and p.e_off.tolist() == [0]
    assert p.v_off.dtype == np.int64 and p.e_off.dtype == np.int64
    assert p.vertices.shape == (0, 3) and p.vertices.dtype == np.float32
    assert p.edges.shape == (0, 2) and p.edges.dtype == np.uint32
    assert p.radii.shape == (0,) and p.radii.dtype == np.float32


@needs_data
def test_a3_thin_slab(dev, block):
    # a slab may be thinner than the set's depth only where it ends at the block's edge
    thin = ("h", F.Z1 - F.Z0 - 1, F.Z1 - F.Z0)
    assert dev.prepare_from_block(block, F.YX, [thin, F.slab_spec("h:8500")], **F.PUBLISHED)[0] is None
    out = dev.extract(block, F.YX, [thin, ("vx", F.YX - 1, F.YX)], **F.EXTRACT_PUBLISHED)
    assert_empty_packed(out[0])
    assert_empty_packed(out[1])
    with pytest.raises(ValueError):
        prep(dev, F.unpack_slab(block, thin))


@needs_data
def test_a3_partial_last_byte(dev, block):
    dense = np.ascontiguousarray(np.unpackbits(block[432:436], axis=-1, bitorder="big")[:, :, :1533])
    for settings, depth in ((P, 2), (V, 4)):
        part = np.ascontiguousarray(dense[:depth])
        blk = np.packbits(part, axis=-1, bitorder="big")
        assert blk.shape[2] == 192
        kw = F.PUBLISHED if settings == P else F.VILLA
        want = prep(dev, part, settings)
        got = dev.prepare_from_block(blk, 1533, [("h", 0, depth)], **kw)[0]
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(want, F.reference_labels(part, *F.PREP_ARGS[settings]))


@needs_data
def test_a3_byte_crossing_vx_slab(dev, block):
    want_bits = np.unpackbits(block, axis=-1, bitorder="big")[:, :, 6:10]
    np.testing.assert_array_equal(F.unpack_slab(block, ("vx", 6, 10)), want_bits)
    got = dev.prepare_from_block(block, F.YX, [("vx", 6, 10)], **F.PUBLISHED)[0]
    np.testing.assert_array_equal(got, prep(dev, np.ascontiguousarray(want_bits)))
    np.testing.assert_array_equal(got, F.reference_labels(want_bits, 2, 50, 80000))


@needs_data
def test_a3_downsample_4_matches_villa(dev, block):
    keys = F.slab_keys(F.B1_DS4)
    specs = [F.slab_spec(k, V) for k in keys]
    from_blk = from_block(dev, block, specs, F.VILLA)
    bad = {}
    for key, spec, got_blk in zip(keys, specs, from_blk):
        want = F.fixture_labels(key, F.B1_DS4)
        got = prep(dev, F.unpack_slab(block, spec), V)
        assert got.dtype == np.uint32 and got.shape == want.shape == got_blk.shape, key
        n = int((got != want).sum()) + int((got_blk != want).sum())
        if n:
            bad[key] = n
    assert not bad, f"mismatched voxels per slab: {bad}"


def test_a3_two_components_in_one_cell(dev):
    # two 6-components that share every 2 x 2 x 2 cell: two x-lines differing in y and z
    v = np.zeros((2, 2, 120), np.uint8)
    v[0, 0] = 1
    v[1, 1] = 1
    got = prep(dev, v)
    assert got.tolist() == [[[2] * 60]]
    np.testing.assert_array_equal(got, F.reference_labels(v, 2, 50, 80000))


@needs_data
def test_a3_connection_only_through_the_crop(dev):
    # two strips joined only in row y = 2, which the crop to a multiple of 2 removes
    plane = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]], np.uint8)
    v = np.ascontiguousarray(np.broadcast_to(plane, (100, 3, 4)))
    assert prep(dev, v).tolist() == [[[1, 1]]] * 50
    cut = v.copy()
    cut[:, 2, 1] = 0
    assert prep(dev, cut).tolist() == [[[1, 2]]] * 50
    # a vy slab 3 thick passes because it ends at the block's edge
    blk = np.packbits(v, axis=-1, bitorder="big")
    assert dev.prepare_from_block(blk, 4, [("vy", 0, 3)], **F.PUBLISHED)[0].tolist() == [[[1, 1]]] * 50


@needs_data
@pytest.mark.parametrize("settings", [P, V])
def test_a3_size_filter_boundaries(dev, settings):
    f = F.size_filter_fixture(settings)
    binary = np.ascontiguousarray(f["binary"])
    got = prep(dev, binary, settings)
    np.testing.assert_array_equal(got, f["labels"])
    assert got.dtype == np.uint32
    # the components of min_count - 1 and max_count pooled voxels are dropped, the other two kept
    kept = [int((got == k).sum()) for k in np.unique(got[got > 0])]
    assert sorted(kept) == sorted(int(c) for c in f["counts"][1:3])


# ---- random volumes against the independent reference, including the node-capacity rerun ----


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("settings", [P, V])
def test_random_volumes_match_reference(dev, seed, settings):
    rng = np.random.default_rng(seed)
    args = F.PREP_ARGS[settings]
    kw = F.PUBLISHED if settings == P else F.VILLA
    ds, depth = args[0], F.H_DEPTH[settings]
    for _ in range(4):
        shape = tuple(int(s) for s in rng.integers(4, 40, size=3))
        v = (rng.random(shape) < rng.uniform(0.2, 0.7)).astype(np.uint8)
        np.testing.assert_array_equal(prep(dev, v, settings), F.reference_labels(v, *args))

        x0 = int(rng.integers(0, 9))
        wide = np.zeros(shape[:2] + (shape[2] + x0 + 5,), np.uint8)
        wide[:, :, x0 : x0 + shape[2]] = v
        blk = np.packbits(wide, axis=-1, bitorder="big")
        z0, y0, w = int(rng.integers(0, shape[0] - depth + 1)), int(rng.integers(0, shape[1] - 3)), wide.shape[2]
        specs = [("vx", x0, x0 + 4), ("h", z0, z0 + depth), ("vy", y0, y0 + 4), ("vx", w - 3, w)]
        got = dev.prepare_from_block(blk, w, specs, **kw)
        for spec, g in zip(specs, got):
            dense = F.unpack_slab(blk, spec, w)
            if min(dense.shape) < ds:
                assert g is None, spec
                continue
            np.testing.assert_array_equal(g, F.reference_labels(dense, *args), err_msg=str(spec))


@needs_data
@pytest.mark.parametrize("settings", [P, V])
def test_checkerboard_overflows_node_capacity(dev, settings):
    # every voxel of a 3D checkerboard is its own 6-component, ds^3 / 2 pieces per cell; the solid block after it
    # survives the size filter, so its label shows the numbering is right after the rerun
    z, y, x = np.indices((20, 12, 20))
    v = ((z + y + x) % 2 == 0).astype(np.uint8)
    v[8:] = 0
    v[10:18] = 1
    got = prep(dev, v, settings)
    want = F.reference_labels(v, *F.PREP_ARGS[settings])
    assert want.any()
    np.testing.assert_array_equal(got, want)


# ---- A4 ----


@needs_data
def test_a4_errors_are_exceptions(dev, block):
    v = np.zeros((4, 4, 4), np.uint8)
    with pytest.raises(ValueError):
        prep(dev, v.astype(np.float32))
    with pytest.raises(ValueError):
        prep(dev, np.zeros((4, 4, 8), np.uint8)[:, :, ::2])
    with pytest.raises(ValueError):
        prep(dev, np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError):
        dev.prepare_from_block(block.astype(np.int16), F.YX, [("h", 0, 2)], **F.PUBLISHED)
    with pytest.raises(ValueError):
        dev.prepare_from_block(block[:, :, ::2], F.YX, [("h", 0, 2)], **F.PUBLISHED)
    with pytest.raises(ValueError):
        dev.prepare_from_block(block, F.YX, [("h", 1531, 1533)], **F.PUBLISHED)
    with pytest.raises(ValueError):
        dev.prepare_from_block(block, F.YX, [("vx", 1534, 1538)], **F.PUBLISHED)
    with pytest.raises(ValueError):
        dev.prepare_from_block(block, F.YX, [("q", 0, 2)], **F.PUBLISHED)
    with pytest.raises(ValueError):
        dev.prepare_from_block(block, 1500, [("h", 0, 2)], **F.PUBLISHED)
    assert issubclass(tm.UnsupportedSettings, NotImplementedError)
    assert issubclass(tm.InputTooLarge, ValueError) and issubclass(tm.OutputTooLarge, ValueError)
    assert issubclass(tm.BackendUnavailable, RuntimeError) and issubclass(tm.GpuError, RuntimeError)
    assert issubclass(tm.SomaUnsupported, NotImplementedError)


@needs_data
def test_a4_settings_are_named(dev, block):
    v = np.ones((4, 4, 4), np.uint8)
    lab = [np.zeros((2, 2, 2), np.uint32)]
    slab = [F.slab_spec("h:8500")]
    unsupported = pytest.raises(tm.UnsupportedSettings)

    # the set is required and never inferred from the values
    with pytest.raises(TypeError):
        dev.prepare_labels(v, 2, 50, 80000)
    with pytest.raises(TypeError):
        dev.skeletonize(lab, anisotropy=(2, 2, 2), dust_threshold=49)
    for name in ("kimimaro", "published_0211 ", "PUBLISHED_0211"):
        with unsupported:
            dev.prepare_labels(v, 2, 50, 80000, settings=name)
        with unsupported:
            dev.prepare_from_block(block, F.YX, slab, **{**F.PUBLISHED, "settings": name})
        with unsupported:
            dev.skeletonize(lab, **{**F.SKEL_PUBLISHED, "settings": name})
        with unsupported:
            dev.extract(block, F.YX, slab, **{**F.EXTRACT_PUBLISHED, "settings": name})

    # a value that differs from the named set's names the field, the value and the set's value
    with pytest.raises(tm.UnsupportedSettings, match=r"downsample=4, got downsample=2"):
        dev.prepare_labels(v, 2, 6, 10000, settings=V)
    with pytest.raises(tm.UnsupportedSettings, match=r"max_count=80000, got max_count=80001"):
        dev.prepare_from_block(block, F.YX, slab, **{**F.PUBLISHED, "max_count": 80001})
    with pytest.raises(tm.UnsupportedSettings, match=r"dust_threshold=49, got dust_threshold=48"):
        dev.extract(block, F.YX, slab, **{**F.EXTRACT_PUBLISHED, "dust_threshold": 48})
    with pytest.raises(tm.UnsupportedSettings, match=r"const=2, got const=3"):
        dev.skeletonize(lab, **F.SKEL_PUBLISHED, const=3.0)
    with pytest.raises(tm.UnsupportedSettings, match="anisotropy"):
        dev.extract(block, F.YX, slab, **{**F.EXTRACT_PUBLISHED, "anisotropy": (4, 4, 4)})
    bad_values = [dict(anisotropy=(2, 2)), dict(anisotropy=2), dict(anisotropy="222"), dict(scale=None),
                  dict(fix_branching=False), dict(fix_branching=1), dict(fix_borders=1), dict(fix_borders=None),
                  dict(dust_threshold=True)]
    for bad in bad_values:
        with unsupported:
            dev.skeletonize(lab, **{**F.SKEL_PUBLISHED, **bad})

    # equality is numeric: ints, floats and numpy scalars all pass; fix_borders may be either bool
    dev.prepare_labels(v, 2.0, np.int64(50), 80000.0, settings=P)
    for aniso in ((2, 2, 2), (2.0, 2.0, 2.0), [2, 2.0, 2], np.array([2, 2, 2])):
        dev.skeletonize(lab, **{**F.SKEL_PUBLISHED, "anisotropy": aniso})
    dev.skeletonize(lab, **F.SKEL_PUBLISHED, scale=1, const=2, fix_borders=False, fix_branching=True)
    dev.skeletonize(lab, **{**F.SKEL_PUBLISHED, "dust_threshold": 49.0})

    # target_order: 'best' or 'kimimaro' with either set; nothing else passes
    for order in ("best", "kimimaro"):
        dev.skeletonize(lab, **F.SKEL_PUBLISHED, target_order=order)
        dev.skeletonize(lab, **F.SKEL_VILLA, target_order=order)
        dev.extract(block, F.YX, slab, **F.EXTRACT_PUBLISHED, target_order=order)
    for order in ("Kimimaro", "Best", "", None, 1):
        with unsupported:
            dev.skeletonize(lab, **F.SKEL_PUBLISHED, target_order=order)
        with unsupported:
            dev.extract(block, F.YX, slab, **F.EXTRACT_PUBLISHED, target_order=order)

    # slab thickness: villa's --downsample-factor 2 with its default geometry cuts 4-deep h slabs
    with unsupported:
        dev.extract(block, F.YX, [("h", 0, 4)], **F.EXTRACT_PUBLISHED)
    with unsupported:
        dev.prepare_from_block(block, F.YX, [("h", 0, 4)], **F.PUBLISHED)
    with unsupported:
        dev.prepare_from_block(block, F.YX, [("h", 0, 2)], **F.VILLA)
    with unsupported:
        dev.prepare_from_block(block, F.YX, [("vy", 0, 8)], **F.VILLA)
    with unsupported:
        dev.prepare_from_block(block, F.YX, [("vx", 0, 3)], **F.PUBLISHED)
    # thinner is allowed only where the slab ends at the block's edge, and never thicker there
    edge = F.Z1 - F.Z0
    got = dev.prepare_from_block(block, F.YX, [("h", edge - 3, edge), ("vx", F.YX - 2, F.YX)], **F.VILLA)
    assert got == [None, None]
    with unsupported:
        dev.prepare_from_block(block, F.YX, [("h", edge - 3, edge)], **F.PUBLISHED)


@needs_data
@needs_testing_build()
def test_a4_limits(fresh, block):
    big = 1 << 40
    spec = F.slab_spec("h:8500")
    vspec = F.slab_spec("vy:768")
    dense = F.unpack_slab(block, spec)
    h_pooled = F.pooled_voxels(spec)
    v_pooled = F.pooled_voxels(vspec)
    longest = max(s // 2 for s in dense.shape)

    def raises(limit, value, *calls):
        fresh._set_limits(**{limit: value})
        for call in calls:
            with pytest.raises(tm.InputTooLarge, match=limit):
                call()
        fresh._set_limits(**{limit: big})

    from_blk = lambda specs: lambda: fresh.prepare_from_block(block, F.YX, specs, **F.PUBLISHED)  # noqa: E731
    extract = lambda specs: lambda: fresh.extract(block, F.YX, specs, **F.EXTRACT_PUBLISHED)  # noqa: E731
    skel = lambda shape: lambda: fresh.skeletonize([np.zeros(shape, np.uint32)], **F.SKEL_PUBLISHED)  # noqa: E731
    labels = lambda: prep(fresh, dense)  # noqa: E731

    raises("block_bytes", block.nbytes - 1, from_blk([spec]), extract([spec]))
    raises("axis_pooled", longest - 1, from_blk([spec]), extract([spec]), labels, skel((1, 1, longest)))
    # spec A4's per-axis case, with the limit lowered: a 1 x 1 x 70,001 pooled slab
    raises("axis_pooled", 70_000, skel((1, 1, 70_001)))
    raises("slab_voxels", dense.size - 1, from_blk([spec]), labels)
    raises("slab_pooled", h_pooled - 1, from_blk([spec]), labels, skel((1, 768, 768)))
    raises("call_pooled", h_pooled + v_pooled - 1, from_blk([spec, vspec]), extract([spec, vspec]))

    # exactly at each limit passes
    fresh._set_limits(block_bytes=block.nbytes, axis_pooled=longest, slab_voxels=dense.size, slab_pooled=h_pooled,
                      call_pooled=h_pooled, output_bytes=4 * h_pooled)
    np.testing.assert_array_equal(from_blk([spec])()[0], F.fixture_labels("h:8500"))
    np.testing.assert_array_equal(labels(), F.fixture_labels("h:8500"))


@needs_data
def test_a4_whole_b1_in_one_call_is_too_large(dev, block):
    # the default limits admit B1 only over several calls (spec part A §2.5's baseline partition); the input checks
    # run before any work, so this fails at once
    assert len(F.slab_calls(SPECS)) > 1
    with pytest.raises(tm.InputTooLarge, match="call_pooled"):
        dev.extract(block, F.YX, SPECS, **F.EXTRACT_PUBLISHED)


@needs_data
@needs_testing_build()
def test_a4_output_too_large_names_the_slab(fresh, block):
    edge = F.Z1 - F.Z0
    thin = ("h", edge - 1, edge)
    specs = [thin, F.slab_spec("h:8500"), F.slab_spec("vy:768"), F.slab_spec("h:8508")]

    def slab_index(call, limit):
        fresh._set_limits(output_bytes=limit)
        with pytest.raises(tm.OutputTooLarge, match="output_bytes") as e:
            call()
        fresh._set_limits(output_bytes=1 << 40)
        return e.value.slab

    # prepare_from_block: a thin slab returns None and counts 0 bytes, but keeps its place in the list
    sizes = [0] + [4 * F.pooled_voxels(s) for s in specs[1:]]
    from_blk = lambda: fresh.prepare_from_block(block, F.YX, specs, **F.PUBLISHED)  # noqa: E731
    assert slab_index(from_blk, sum(sizes[:3])) == 3
    assert slab_index(from_blk, sizes[1] - 1) == 1

    # extract and skeletonize: sizes known only after path extraction
    packed = fresh.extract(block, F.YX, specs, **F.EXTRACT_PUBLISHED)
    sizes = [packed_nbytes(p) for p in packed]
    assert sizes[0] > 0 and min(sizes[1:]) > sizes[0]
    run = lambda: fresh.extract(block, F.YX, specs, **F.EXTRACT_PUBLISHED)  # noqa: E731
    assert slab_index(run, sum(sizes[:3])) == 3
    assert slab_index(run, sum(sizes[:2]) - 1) == 1
    assert slab_index(run, 0) == 0

    labels = [F.fixture_labels(k) for k in ("h:8500", "vy:768", "h:8508")]
    skel = lambda: fresh.skeletonize(labels, **F.SKEL_PUBLISHED)  # noqa: E731
    sizes = [packed_nbytes(p) for p in skel()]
    assert slab_index(skel, sum(sizes[:2])) == 2
    assert slab_index(skel, sizes[0] - 1) == 0


@needs_testing_build()
@pytest.mark.skipif(os.environ.get("TEASAR_GPU") != "1", reason="GPU only")
def test_a4_injected_gpu_failure():
    d = tm.Device()
    v = np.ones((4, 4, 4), np.uint8)
    d._inject_gpu_failure()
    with pytest.raises(tm.GpuError):
        prep(d, v, V)
    # the device keeps working after a failed call; an 8-cube pools to 8 cells of one label, over villa_default's
    # min_count of 6
    big = np.ones((8, 8, 8), np.uint8)
    assert prep(d, big, V).tolist() == [[[1, 1], [1, 1]], [[1, 1], [1, 1]]]
    d.close()


# ---- A6 ----


@needs_data
def test_a6_determinism_in_and_across_processes(dev, block):
    first = labels_digest(from_block(dev, block, SPECS))
    assert labels_digest(from_block(dev, block, SPECS)) == first
    kind = "gpu" if dev.backend == "metal" else "cpu"
    code = (
        "import sys; sys.path.insert(0, 'tests'); import fixtures as F, test_part_a as T, conftest as C;"
        f"print(T.labels_digest(T.from_block(C.open_device({kind!r}), F.block(), T.SPECS)))"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    other = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
    assert other.stdout.strip().splitlines()[-1] == first


@needs_data
def test_a6_outputs_are_owned(fresh, block):
    a = fresh.prepare_from_block(block, F.YX, SPECS[:3], **F.PUBLISHED)
    b = prep(fresh, F.unpack_slab(block, SPECS[200]))
    snap = [x.copy() for x in a] + [b.copy()]
    assert all(x.flags.writeable and x.flags.c_contiguous for x in a + [b])
    fresh.prepare_from_block(block, F.YX, SPECS[3:9], **F.PUBLISHED)
    prep(fresh, F.unpack_slab(block, SPECS[201]))
    fresh.close()
    for x, s in zip(a + [b], snap):
        np.testing.assert_array_equal(x, s)
    a[0][...] = 7
    assert (a[0] == 7).all()
    with pytest.raises(RuntimeError):
        prep(fresh, np.zeros((2, 2, 2), np.uint8))


def test_device_properties(dev):
    assert dev.backend in ("cpu", "metal")
    assert isinstance(dev.name, str) and dev.name
    used, working = dev.memory()
    assert used >= 0 and working >= 0
    assert tm.__version__ == "0.1.0"
    if os.environ.get("TEASAR_GPU") == "1":
        ok, reason = tm.check_available()
        assert ok and reason == dev.name or dev.backend == "cpu", reason
    with tm.CpuDevice(threads=2) as c:
        assert c.backend == "cpu"


# ---- §3.3: which labels get a skeleton, and their order ----


@needs_data
@needs_testing_build()
def test_skeleton_ids_and_order_match_kimimaro():
    bad = {}
    for key in KEYS:
        f = F.fixture(key)
        want = f["skel_ids"]
        assert want.dtype == np.uint64 and (want <= np.iinfo(np.uint32).max).all(), key
        got = np.asarray(tm._core._skeleton_ids(f["labels"], 49), np.uint32)
        if not np.array_equal(got, want.astype(np.uint32)):
            bad[key] = (len(got), len(want))
    assert not bad, bad


@needs_data
def test_call_statistics(dev, block):
    """`last_stats()` (§1.4): the most recent call's wall and GPU-busy time, host threads and the piece split."""
    keys = ("wall_s", "gpu_busy_s", "host_threads", "cpu_pieces", "cpu_voxels", "gpu_pieces", "gpu_voxels",
            "cpu_busy_s")
    slabs = [F.slab_spec(k) for k in F.slab_keys()[:3]]
    out = dev.extract(block, F.YX, slabs, **F.EXTRACT_PUBLISHED)
    s = dev.last_stats()
    assert sorted(s) == sorted(keys)
    assert s["wall_s"] > 0 and s["host_threads"] >= 1
    pieces = s["cpu_pieces"] + s["gpu_pieces"]
    assert pieces >= sum(len(p.ids) for p in out) > 0, "every skeleton comes from a counted piece"
    assert s["cpu_voxels"] + s["gpu_voxels"] >= pieces
    if dev.backend == "metal":
        assert 0 < s["gpu_busy_s"] <= s["wall_s"]
    else:
        assert s["gpu_busy_s"] == 0 and s["gpu_pieces"] == 0 and s["cpu_busy_s"] > 0

    # the next call replaces them
    dev.prepare_labels(np.ones((4, 4, 4), np.uint8), 2, 50, 80000, settings=P)
    s = dev.last_stats()
    assert s["cpu_pieces"] == s["gpu_pieces"] == 0 and s["wall_s"] > 0
