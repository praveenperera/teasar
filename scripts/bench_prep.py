"""Benchmark of the labelling stage, plus the copy-kernel bandwidth and the device memory.

prepare_from_block on all 384 slabs of the fixture box, over the baseline partition; wall clock from the first call's
entry to the last call's return. One warm-up pass, then the median of 5 passes, in one process. Run unprofiled, at
medium.
"""

import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import fixtures as F  # noqa: E402
import teasar as tm  # noqa: E402


def gpu_util():
    r = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if '"PerformanceStatistics"' in line:
            i = line.find('"Device Utilization %"=')
            if i >= 0:
                return int(line[i + 23 :].split(",")[0].split("}")[0])
    return None


def one_pass(dev, blk, plan):
    t = time.perf_counter()
    n = 0
    for c in plan:
        n += len(dev.prepare_from_block(blk, F.YX, c, **F.PUBLISHED))
    assert n == 384
    return time.perf_counter() - t


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "gpu"
    blk = F.block()
    specs = [F.slab_spec(k) for k in F.slab_keys()]
    plan = F.slab_calls(specs)
    dev = F.open_device(kind)
    print(f"device {dev.name}; {len(plan)} calls of {[len(c) for c in plan]} slabs", flush=True)
    if os.environ.get("TEASAR_PROFILE") == "1":
        print(f"profiled pass {one_pass(dev, blk, plan):.4f} s (not a headline number)")
        return
    print(f"gpu util before {gpu_util()}%, load {os.getloadavg()}")
    warm = one_pass(dev, blk, plan)
    times = [one_pass(dev, blk, plan) for _ in range(5)]
    print(f"gpu util after {gpu_util()}%, load {os.getloadavg()}")
    print(f"warm-up {warm:.4f} s; passes {' '.join(f'{t:.4f}' for t in times)}")
    print(f"prepare_from_block median {statistics.median(times):.4f} s")
    print(f"device memory (allocated, working set) {dev.memory()}")
    if kind == "gpu" and getattr(tm._core, "TESTING", False):
        bw = [dev._bench_copy(1 << 30, 20) for _ in range(3)]
        print(f"copy-kernel bandwidth GB/s (read + write): {' '.join(f'{b:.1f}' for b in bw)}")


if __name__ == "__main__":
    main()
