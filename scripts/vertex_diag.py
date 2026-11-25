#!/usr/bin/env python3
"""Per-slab skeleton diagnostics against Kimimaro's fixture skeletons, for settling the open choices quickly.

For a sample of B1 slabs, prints for each TEASAR_CHOICES variant given: vertex count ratio (ours / reference),
the share of our vertices that are reference vertices and the reverse, and the leaf count ratio.

usage: vertex_diag.py [--every N] VARIANT...   (VARIANT is a TEASAR_CHOICES string; "" is the defaults)"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

CHILD = r"""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
import fixtures, teasar as tm
dev = tm.CpuDevice()
keys = sys.argv[2:]
labels = [fixtures.fixture_labels(k) for k in keys]
out = [p for call in fixtures.label_calls(labels) for p in dev.skeletonize(call, **fixtures.SKEL_PUBLISHED)]
tot = dict(v=0, rv=0, hit=0, rhit=0, leaves=0, rleaves=0)
def leaves(edges, n):
    return int((np.bincount(edges.ravel(), minlength=n) == 1).sum()) if n else 0
for k, p in zip(keys, out):
    f = fixtures.fixture(k)
    rv = f["vertices"].astype(np.float32)
    mine = {tuple(x) for x in p.vertices.tolist()}
    ref = {tuple(x) for x in rv.tolist()}
    tot["v"] += len(p.vertices); tot["rv"] += len(rv)
    tot["hit"] += sum(1 for x in mine if x in ref); tot["rhit"] += sum(1 for x in ref if x in mine)
    for j in range(len(p.ids)):
        tot["leaves"] += leaves(p.edges[p.e_off[j]:p.e_off[j+1]], int(p.v_off[j+1] - p.v_off[j]))
    so, eo = f["skel_v_off"], f["skel_e_off"]
    for j in range(len(f["skel_ids"])):
        tot["rleaves"] += leaves(f["edges"][eo[j]:eo[j+1]], int(so[j+1] - so[j]))
print(json.dumps(tot))
"""


def main():
    import json

    import fixtures

    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=24)
    ap.add_argument("variants", nargs="+")
    a = ap.parse_args()
    keys = fixtures.slab_keys()[:: a.every]
    groups = {p: [k for k in keys if k.split(":")[0] == p] for p in ("h", "vy", "vx")}
    for v in a.variants:
        env = dict(os.environ, TEASAR_CHOICES=v)
        line = []
        for p, ks in groups.items():
            r = subprocess.run([sys.executable, "-c", CHILD, os.path.dirname(fixtures.__file__), *ks], env=env,
                               capture_output=True, text=True, check=True)
            t = json.loads(r.stdout)
            line.append(f"{p}: v {t['v'] / t['rv']:.3f} ours-in-ref {t['hit'] / t['v']:.3f} "
                        f"ref-in-ours {t['rhit'] / t['rv']:.3f} leaves {t['leaves'] / t['rleaves']:.3f}")
        print(f"{v or 'default':40s} " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
