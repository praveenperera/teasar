"""Part B acceptance tests that run in-process (spec part B §4): B3 invariants and the DBF rule, B4 coverage, B5
synthetic shapes, B6 determinism, and B7 CPU and Metal identity of the skeletons and of each piece's fields.

B1, B2 and B9 run end to end through the harness (scripts/b1_selfcheck.sh, scripts/b1_counts.py). Fixture-wide B3
and B4 checks run on every 16th B1 slab by default and on all 384 with TEASAR_FULL=1; B8's run on all 144
`villa_default` fixture slabs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pytest
from scipy import ndimage

import b5
import fixtures
import teasar as tm
from conftest import needs_data

FULL = os.environ.get("TEASAR_FULL") == "1"
A = 2.0
SKEL = dict(settings="published_0211", anisotropy=(2, 2, 2), dust_threshold=49)
CUBE = np.ones((3, 3, 3), bool)


def fixture_slabs() -> list[str]:
    keys = fixtures.listed_slab_keys()
    return keys if FULL else keys[::16]


def skeletonize(dev, labels, **kw):
    return dev.skeletonize([np.ascontiguousarray(labels, np.uint32)], **{**SKEL, **kw})[0]


def pieces(labels: np.ndarray, dust: int):
    """(label, bbox slices, piece mask within the bbox) for every non-dust 26-connected piece."""
    out = []
    for label, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        lab, n = ndimage.label(labels[sl] == label, structure=CUBE)
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() > dust:
                out.append((label, sl, m))
    return out


def dbf_at(labels: np.ndarray, label: int, sl, mask: np.ndarray, at: np.ndarray, margin: int, a: float) -> np.ndarray:
    """Spec §2.2's DBF at pooled points `at` of one piece, from a float64 EDT rounded once to float32."""
    lo = [max(s.start - margin, 0) for s in sl]
    hi = [min(s.stop + margin, n) for s, n in zip(sl, labels.shape)]
    box = np.zeros([h - l for l, h in zip(lo, hi)], bool)
    inner = tuple(slice(s.start - l, s.stop - l) for s, l in zip(sl, lo))
    box[inner] = mask
    if box.all():
        # a piece that fills the array measures to the border (§2.2)
        box = np.pad(box, 1)
        lo = [l - 1 for l in lo]
    dist = ndimage.distance_transform_edt(box)
    idx = tuple((at - np.array(lo)).T)
    return (a * dist[idx]).astype(np.float32)


def ball_cover(covered: np.ndarray, q: np.ndarray, rad: float, a: float) -> None:
    """Marks in `covered` the voxels strictly inside the invalidation ball of a vertex at `q` with radius `rad`:
    r = fma(1, rad, 2), and offset d is inside when a² |d|² < r² in float32; fma(1, rad, 2) rounds rad + 2 once, as
    a float32 addition does. Only the ball's part inside `covered` is built, so a long thin piece stays cheap"""
    f32 = np.float32
    r = f32(f32(rad) + f32(2.0))
    h = int(np.floor(r / f32(a)))
    lo = np.maximum(q - h, 0)
    hi = np.minimum(q + h + 1, covered.shape)
    d = [np.arange(l, u) - c for l, u, c in zip(lo, hi, q)]
    d2 = d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2
    covered[tuple(slice(l, u) for l, u in zip(lo, hi))] |= f32(a) * f32(a) * d2.astype(f32) < r * r


def check_sorted(p):
    """The sorted form of §2.7: per skeleton, vertices in increasing C-order index, edges (lower, higher), rows sorted"""
    for k in range(len(p.ids)):
        v = p.vertices[p.v_off[k] : p.v_off[k + 1]]
        flat = [tuple(x) for x in v.tolist()]
        assert all(a < b for a, b in zip(flat, flat[1:])), f"skeleton {k}: vertices not in increasing C order"
        e = p.edges[p.e_off[k] : p.e_off[k + 1]].tolist()
        assert all(u < w for u, w in e), f"skeleton {k}: an edge not written (lower, higher)"
        assert e == sorted(e), f"skeleton {k}: edge rows not sorted"


def check_invariants(labels: np.ndarray, p, ref_ids=None, a: float = A, dust: int = SKEL["dust_threshold"]):
    """B3 and B4 for one slab's output, at pooled voxel size `a`."""
    shape = np.array(labels.shape)
    ps = pieces(labels, dust)
    by_label: dict[int, list] = {}
    for label, sl, m in ps:
        by_label.setdefault(label, []).append((sl, m))
    assert sorted(p.ids.tolist()) == sorted(by_label), "ids are the labels with a non-dust piece"
    if ref_ids is not None:
        assert p.ids.tolist() == list(ref_ids), "ids in Fortran first-occurrence order"
    assert len(p.v_off) == len(p.ids) + 1 and len(p.e_off) == len(p.ids) + 1
    assert b5.is_forest(p)
    check_sorted(p)

    idx = p.vertices / a
    assert np.array_equal(idx, np.round(idx)), "vertices are index × a exactly"
    idx = idx.astype(np.int64)
    assert (idx >= 0).all() and (idx < shape).all()
    for k, label in enumerate(p.ids):
        lo, hi = p.v_off[k], p.v_off[k + 1]
        v = idx[lo:hi]
        assert (labels[tuple(v.T)] == label).all(), f"label {label}: a vertex off its label"
        e = p.edges[p.e_off[k] : p.e_off[k + 1]].astype(np.int64)
        assert len(e) == 0 or e.max() < hi - lo
        step = np.abs(v[e[:, 0]] - v[e[:, 1]])
        assert (step.max(axis=1, initial=1) == 1).all(), f"label {label}: an edge that is not a 26-step"
        # one tree per non-dust piece
        assert (hi - lo) - len(e) == len(by_label[label]), f"label {label}: trees != pieces"

        r = p.radii[lo:hi]
        margin = int(np.ceil(r.max(initial=0) / a)) + 2
        cover = 0
        for sl, m in by_label[label]:
            start = np.array([s.start for s in sl])
            inside = m[tuple(np.clip(v - start, 0, np.array(m.shape) - 1).T)] & (v >= start).all(1) & (
                v < start + m.shape
            ).all(1)
            pv = v[inside]
            want = dbf_at(labels, label, sl, m, pv, margin, a)
            assert np.array_equal(r[inside], want), f"label {label}: radii differ from §2.2"
            # B4: every piece voxel strictly inside the ball of some vertex of its piece (§2.6 step 3)
            covered = np.zeros(m.shape, bool)
            for q, rad in zip(pv - start, r[inside]):
                ball_cover(covered, q, float(rad), a)
            assert covered[m].all(), f"label {label}: {(~covered[m]).sum()} voxels uncovered"
            cover += 1
        assert cover == len(by_label[label])


# ---- B3 synthetic radii ----


def test_b3_synthetic_radii(dev):
    two = np.zeros((1, 2, 60), np.uint32)
    two[0, 0], two[0, 1] = 1, 2
    p = skeletonize(dev, two)
    assert len(p.radii) and (p.radii == 2.0).all()

    p = skeletonize(dev, np.ones((1, 5, 200), np.uint32))
    assert (p.radii == 2.0).all()

    p = skeletonize(dev, np.ones((3, 9, 60), np.uint32))
    z, y, x = (p.vertices / A).astype(int).T
    edge = (z == 0) | (z == 2) | (y == 0) | (y == 8) | (x == 0) | (x == 59)
    assert np.array_equal(p.radii, np.where(edge, 2.0, 4.0).astype(np.float32))

    hole = np.ones((1, 9, 60), np.uint32)
    hole[0, 8, 59] = 0
    p = skeletonize(dev, hole)
    _, y, x = (p.vertices / A).astype(int).T
    want = (A * np.sqrt((y - 8.0) ** 2 + (x - 59.0) ** 2)).astype(np.float32)
    assert np.array_equal(p.radii, want)


# ---- B3, B4 on fixtures ----


ORDERS = ("best", "kimimaro")


@needs_data
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("key", fixture_slabs())
def test_b3_b4_fixture(dev, key, order):
    f = fixtures.fixture(key)
    labels = f["labels"]
    p = skeletonize(dev, labels, target_order=order)
    check_invariants(labels, p, ref_ids=f["skel_ids"].astype(np.uint32))


# ---- B8: downsample 4 ----

def ds4_slabs() -> list[str]:
    """The ds4 fixture slabs; empty without TEASAR_DATA, where the tests skip"""
    return sorted(p.stem for p in fixtures.data(fixtures.B1_DS4).glob("*.npz")) if fixtures.DATA else []


@needs_data
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("key", ds4_slabs())
def test_b8_b3_b4(dev, key, order):
    f = np.load(fixtures.data(fixtures.B1_DS4, f"{key}.npz"))
    labels = f["labels"]
    p = skeletonize(dev, labels, **fixtures.SKEL_VILLA, target_order=order)
    check_invariants(labels, p, ref_ids=f["skel_ids"].astype(np.uint32), a=4.0, dust=5)


@needs_data
@pytest.mark.skipif(os.environ.get("TEASAR_GPU") != "1", reason="needs the GPU")
@pytest.mark.parametrize("order", ORDERS)
def test_b8_cpu_equals_metal(order):
    labels = [np.load(fixtures.data(fixtures.B1_DS4, f"{k}.npz"))["labels"] for k in ds4_slabs()]
    cpu = tm.CpuDevice().skeletonize(labels, **fixtures.SKEL_VILLA, target_order=order)
    gpu = tm.Device().skeletonize(labels, **fixtures.SKEL_VILLA, target_order=order)
    assert len(cpu) == len(gpu) == 144
    bad = [k for k, c, g in zip(ds4_slabs(), cpu, gpu, strict=True) if not same(c, g)]
    assert not bad, f"CPU and Metal differ on {bad[:8]}"


# ---- B5 ----

# empty without TEASAR_DATA, where every test that uses them skips
MASKS = b5.masks() if fixtures.DATA else {}
REF = b5.reference() if fixtures.DATA else {}


@needs_data
@pytest.mark.parametrize("name", sorted(MASKS))
def test_b5_synthetic(dev, name):
    mask = MASKS[name]
    p = skeletonize(dev, mask)
    check_invariants(mask, p)
    fails = b5.check(name, mask, p, REF[name])
    if name == "diag_ribbon_w4":
        # structure not gated: reported beside the reference's 60 leaves and 58 branch points
        print(f"diag_ribbon_w4: {len(b5.leaf_points(p))} leaves, {len(b5.branch_points(p))} branch points")
    if fails and name in b5.KNOWN_GAP:
        pytest.xfail(f"{name}: known gap of the next-target rule before the 'kimimaro' addendum: {fails}")
    assert not fails, f"{name}: {fails}"


# ---- B6, B7 ----


def same(p, q) -> bool:
    return all(np.array_equal(getattr(p, k), getattr(q, k)) for k in ("ids", "v_off", "e_off", "vertices", "edges",
                                                                       "radii"))


@needs_data
def test_b6_repeatable(dev):
    labels = [fixtures.fixture_labels(k) for k in fixture_slabs()[:6]]
    a = dev.skeletonize(labels, **SKEL)
    b = dev.skeletonize(labels, **SKEL)
    assert all(same(p, q) for p, q in zip(a, b))
    for p, q in zip(a, dev.skeletonize(labels[::-1], **SKEL)[::-1]):
        assert same(p, q), "a slab's output does not depend on its batch neighbours"


B6_CHILD = """
import sys, numpy as np, teasar as tm
sys.path.insert(0, sys.argv[1])
import fixtures
dev = tm.Device() if sys.argv[2] == 'gpu' else tm.CpuDevice()
labels = [fixtures.fixture_labels(k) for k in sys.argv[3:]]
out = dev.skeletonize(labels, settings="published_0211", anisotropy=(2, 2, 2), dust_threshold=49)
import hashlib
h = hashlib.sha256()
for p in out:
    for k in ('ids', 'v_off', 'e_off', 'vertices', 'edges', 'radii'):
        h.update(np.ascontiguousarray(getattr(p, k)).tobytes())
print(h.hexdigest())
"""


def digest(out) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in out:
        for k in ("ids", "v_off", "e_off", "vertices", "edges", "radii"):
            h.update(np.ascontiguousarray(getattr(p, k)).tobytes())
    return h.hexdigest()


def child_digest(kind: str, keys: list[str], env: dict[str, str]) -> str:
    child = subprocess.run(
        [sys.executable, "-c", B6_CHILD, os.path.dirname(fixtures.__file__), kind, *keys],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **env},
    )
    return child.stdout.strip()


@needs_data
def test_b6_new_process(dev):
    keys = fixture_slabs()[:4]
    here = digest(dev.skeletonize([fixtures.fixture_labels(k) for k in keys], **SKEL))
    kind = "gpu" if dev.backend == "metal" else "cpu"
    assert child_digest(kind, keys, {}) == here


# B6 in hybrid mode (§3.1): the output does not depend on the work budget, on which side runs a piece (0 puts every
# piece on the host threads), or on the host thread count
B6_SPLITS = [
    {"TEASAR_GPU_BUDGET": "4096"},
    {"TEASAR_CPU_PIECE": "0"},
    {"TEASAR_CPU_THREADS": "1"},
]


@needs_data
@pytest.mark.skipif(os.environ.get("TEASAR_GPU") != "1", reason="needs the GPU")
@pytest.mark.parametrize("env", B6_SPLITS, ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items()))
def test_b6_hybrid_split(env):
    keys = fixture_slabs()[::48]
    base = child_digest("gpu", keys, {})
    assert child_digest("gpu", keys, env) == base


def b7_slabs() -> list[str]:
    n = os.environ.get("TEASAR_B7_SLABS")
    keys = fixture_slabs()
    return keys if n is None else keys[:: max(len(keys) // int(n), 1)][: int(n)]


@needs_data
@pytest.mark.skipif(os.environ.get("TEASAR_GPU") != "1", reason="needs the GPU")
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("key", b7_slabs())
def test_b7_cpu_equals_metal(key, order):
    labels = fixtures.fixture_labels(key)
    cpu, gpu = tm.CpuDevice(), tm.Device()
    assert same(skeletonize(cpu, labels, target_order=order), skeletonize(gpu, labels, target_order=order))


def b7_field_inputs() -> list[tuple[str, str]]:
    """B7's field-level inputs: the B1 slabs (every 16th; all 384 with TEASAR_FULL=1) and every B5 mask"""
    return [("b1", k) for k in fixture_slabs()] + [("b5", n) for n in sorted(MASKS)]


def first_field_mismatch(shape, cpu: dict, gpu: dict) -> str | None:
    """The first voxel at which the Metal dump differs from the CPU's, field by field, then the roots"""
    for k in ("dbf", "b", "d", "p"):
        diff = np.flatnonzero(cpu[k] != gpu[k])
        if diff.size:
            g = diff[0]
            return (f"{k} at voxel {np.unravel_index(g, shape)}: CPU {cpu[k][g]:#010x}, Metal {gpu[k][g]:#010x} "
                    f"({diff.size} voxels differ)")
    if not np.array_equal(cpu["roots"], gpu["roots"]):
        only = np.setxor1d(cpu["roots"], gpu["roots"])
        return f"roots differ at {len(only)} voxels, first {np.unravel_index(only[0], shape)}"
    return None


@needs_data
@pytest.mark.skipif(os.environ.get("TEASAR_GPU") != "1", reason="needs the GPU")
@pytest.mark.parametrize("kind,name", b7_field_inputs())
def test_b7_fields(kind, name):
    """Spec part B 4.7: per piece, DBF, B, D, P and the root from the Metal path loop equal the CPU's bit for bit"""
    labels = fixtures.fixture_labels(name) if kind == "b1" else np.ascontiguousarray(MASKS[name], np.uint32)
    cpu = tm.CpuDevice()._fields([labels], **SKEL)[0]
    gpu = tm.Device()._fields([labels], **SKEL)[0]
    # the dumps cover exactly the non-dust pieces, one root each, so an empty dump cannot pass
    kept = pieces(labels, SKEL["dust_threshold"])
    assert len(cpu["roots"]) == len(kept) > 0
    assert int((cpu["p"] != 0xFFFFFFFF).sum()) == sum(int(m.sum()) for _, _, m in kept)
    msg = first_field_mismatch(labels.shape, cpu, gpu)
    assert msg is None, f"{kind} {name}: {msg}"


# ---- A4: SomaUnsupported strips (spec part A 2.4, 2.7; addendum KM1) ----


@pytest.mark.parametrize("n", [2102, 552, 551])
def test_a4_soma_strips_best(dev, n):
    """1 x 1 x N strips with x = 0 background (largest DBF 2 (N - 1)): 'best' never raises SomaUnsupported, and the
    skeleton meets B3 and B4"""
    strip = np.ones((1, 1, n), np.uint32)
    strip[0, 0, 0] = 0
    p = dev.skeletonize([strip], **SKEL, target_order="best")[0]
    check_invariants(strip, p)
    assert p.ids.tolist() == [1]


@pytest.mark.parametrize("n", [2102, 552, 551])
def test_a4_soma_strips_kimimaro(dev, n):
    """The same strips with target_order='kimimaro': a largest DBF over 1,100 refuses the whole call with
    SomaUnsupported naming the first such slab; exactly 1,100 (N = 551) traces"""
    strip = np.ones((1, 1, n), np.uint32)
    strip[0, 0, 0] = 0
    small = np.zeros((1, 4, 60), np.uint32)
    small[0, :, 2:58] = 3
    if n == 551:
        p = dev.skeletonize([strip], **SKEL, target_order="kimimaro")[0]
        check_invariants(strip, p)
        return
    with pytest.raises(tm.SomaUnsupported) as e:
        dev.skeletonize([small, strip, strip], **SKEL, target_order="kimimaro")
    assert e.value.slab == 1


# ---- compatibility mode (addendum KM) ----


def ref_sets(sk) -> tuple[set, set, dict]:
    """A reference skeleton as its vertex set, edge set (pairs of pooled coordinates) and radius per vertex"""
    v = [tuple(x) for x in sk["vertices"]]
    edges = {frozenset((v[a], v[b])) for a, b in sk["edges"]}
    return set(v), edges, dict(zip(v, sk["radii"]))


def cand_sets(p, k: int, a: float = A) -> tuple[set, set, dict]:
    lo, hi = p.v_off[k], p.v_off[k + 1]
    v = [tuple(int(c) for c in x) for x in (p.vertices[lo:hi] / a).tolist()]
    e = p.edges[p.e_off[k] : p.e_off[k + 1]].tolist()
    return set(v), {frozenset((v[x], v[y])) for x, y in e}, dict(zip(v, p.radii[lo:hi].tolist()))


@needs_data
@pytest.mark.parametrize("name", sorted(MASKS))
def test_b5_kimimaro(dev, name):
    """B5 in compatibility mode: every condition passes, L included, and each skeleton equals the reference's
    vertex, edge and radius sets (addendum evidence §1: all 16 masks identical)"""
    mask = MASKS[name]
    p = skeletonize(dev, mask, target_order="kimimaro")
    check_invariants(mask, p)
    fails = b5.check(name, mask, p, REF[name])
    assert not fails, f"{name}: {fails}"
    ref = REF[name]["skeletons"]
    assert sorted(str(i) for i in p.ids.tolist()) == sorted(ref), "labels"
    for k, label in enumerate(p.ids.tolist()):
        rv, re_, rr = ref_sets(ref[str(label)])
        cv, ce, cr = cand_sets(p, k)
        assert cv == rv, f"label {label}: {len(cv ^ rv)} vertices differ"
        assert ce == re_, f"label {label}: {len(ce ^ re_)} edges differ"
        assert all(np.float32(cr[x]) == np.float32(rr[x]) for x in cv), f"label {label}: radii differ"


# ---- B12: exact masks, resume and fallback (spec part B 4.12) ----

B12_CHILD = """
import sys, hashlib, numpy as np, teasar as tm
m = np.load(sys.argv[1])[sys.argv[2]]
out = tm.Device().skeletonize([m], settings="published_0211", anisotropy=(2, 2, 2), dust_threshold=49)
h = hashlib.sha256()
for p in out:
    for k in ('ids', 'v_off', 'e_off', 'vertices', 'edges', 'radii'):
        h.update(np.ascontiguousarray(getattr(p, k)).tobytes())
print(h.hexdigest())
"""
# a budget small enough that each mask needs many dispatches, every piece on the GPU path, and no fallback
B12_ENV = {"TEASAR_GPU_BUDGET": "4096", "TEASAR_CPU_PIECE": "100000", "TEASAR_CPU_FALLBACK": "0", "TEASAR_PROFILE": "1"}


def b12_child(name: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", B12_CHILD, str(fixtures.data("fixtures/synthetic/b12_masks.npz")), name],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )


def b12_run(name: str, env: dict[str, str]) -> tuple[str, str]:
    """One mask's output digest and profile, in a fresh process"""
    child = b12_child(name, env)
    assert child.returncode == 0, child.stderr[-2000:]
    return child.stdout.strip(), child.stderr


@needs_data
@pytest.mark.skipif(os.environ.get("TEASAR_GPU") != "1", reason="needs the GPU")
@pytest.mark.parametrize("name", ["long_bar", "comb", "square_280"])
def test_b12_resume_on_gpu(name):
    """Each mask pauses and resumes on the GPU, in the invalidation phase too for square_280, and its output equals the
    run with an unlimited budget"""
    small, err = b12_run(name, B12_ENV)
    m = re.search(r"(\d+) resume dispatches, (\d+) pieces paused, (\d+) ordinary", err)
    assert m, err[-2000:]
    resumes, paused, ordinary = map(int, m.groups())
    pauses = re.search(r"pauses by phase: (.*)", err).group(1)
    host = int(re.search(r"host pieces: (\d+) pieces", err).group(1))
    print(f"{name}: {resumes} resume dispatches, {paused} paused, {ordinary} ordinary traces; pauses by phase: {pauses}")
    assert resumes >= 1 and paused >= 1
    assert host == 0, "every piece on the GPU path"
    if name == "comb":
        assert ordinary >= 2500
    if name == "square_280":
        assert re.search(r"invalidation \d+", pauses), pauses
    full, _ = b12_run(name, {**B12_ENV, "TEASAR_GPU_BUDGET": str(1 << 30)})
    assert small == full, "output depends on the budget"


@needs_data
@pytest.mark.skipif(os.environ.get("TEASAR_GPU") != "1", reason="needs the GPU")
def test_b12_fallback():
    """A piece over TEASAR_LARGE_PIECE goes to the host, counted as fallback, with the same output; with
    TEASAR_CPU_FALLBACK=0 that is an error"""
    base = {"TEASAR_CPU_PIECE": "100000", "TEASAR_PROFILE": "1"}
    on_gpu, _ = b12_run("comb", base)
    fell, err = b12_run("comb", {**base, "TEASAR_LARGE_PIECE": "50000"})
    assert "1 by the fallback" in err, err[-2000:]
    assert fell == on_gpu
    refused = b12_child("comb", {**base, "TEASAR_LARGE_PIECE": "50000", "TEASAR_CPU_FALLBACK": "0"})
    assert refused.returncode != 0 and "GpuError" in refused.stderr and "TEASAR_LARGE_PIECE" in refused.stderr
