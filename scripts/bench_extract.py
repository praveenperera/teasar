"""A10: the fused extract stage benchmark (spec part A §5.3), which part B's B9 projection uses.

extract on all 384 B1 slabs, over the baseline partition of §2.5, at published_0211; wall clock from the first call's
entry to the last call's return, including every part A and part B dispatch and copying the Packed arrays out. One
warm-up pass, then the median of 5 passes, in one process. Run unprofiled, at medium; TEASAR_PROFILE=1 runs one
profiled pass for the per-kernel profile instead.

usage: bench_extract.py [gpu|cpu] [best|kimimaro]
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import fixtures as F  # noqa: E402
import teasar as tm  # noqa: E402

# the B9 projection's host costs on the harness mock: read and bit-pack, track walk, DBM write (part B §4.9)
HOST_SECS = 3.7


def one_pass(dev, blk, plan, order):
    """The pass's seconds and vertex count; prints its summed call statistics"""
    stats = F.StatsSum()
    t = time.perf_counter()
    out = []
    for c in plan:
        out += dev.extract(blk, F.YX, c, **F.EXTRACT_PUBLISHED, target_order=order)
        stats.add(dev)
    secs = time.perf_counter() - t
    assert len(out) == 384, len(out)
    print(f"  pass {secs:.3f} s: {stats}", flush=True)
    return secs, sum(len(p.vertices) for p in out)


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "gpu"
    order = sys.argv[2] if len(sys.argv) > 2 else "best"
    blk = F.block()
    plan = F.slab_calls([F.slab_spec(k) for k in F.slab_keys()])
    dev = F.open_device(kind)
    print(f"device {dev.name}, target_order {order!r}; {len(plan)} calls of {[len(c) for c in plan]} slabs", flush=True)
    if os.environ.get("TEASAR_PROFILE") == "1":
        secs, _ = one_pass(dev, blk, plan, order)
        print(f"profiled pass {secs:.3f} s (not a headline number)")
        return
    print(f"load {os.getloadavg()}")
    warm, verts = one_pass(dev, blk, plan, order)
    runs = [one_pass(dev, blk, plan, order) for _ in range(5)]
    assert all(v == verts for _, v in runs), "the vertex count changed between passes"
    times = [s for s, _ in runs]
    med = statistics.median(times)
    print(f"load {os.getloadavg()}")
    print(f"warm-up {warm:.3f} s; passes {' '.join(f'{t:.3f}' for t in times)}; {verts} vertices")
    print(f"A10 median {med:.3f} s; B9 projection {med + HOST_SECS:.2f} s (gate 24 s, target 8 s)")
    print(f"device memory (allocated, working set) {dev.memory()}")


if __name__ == "__main__":
    main()
