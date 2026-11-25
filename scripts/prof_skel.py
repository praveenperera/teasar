#!/usr/bin/env python3
"""Times `skeletonize` on all 384 B1 fixture label arrays, in calls over spec part A §2.5's baseline partition
(TEASAR_PROFILE=1 prints the phase split to stderr).

usage: prof_skel.py [cpu|gpu] [--every N]"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import fixtures  # noqa: E402
import teasar as tm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("device", nargs="?", default="cpu")
ap.add_argument("--every", type=int, default=1)
a = ap.parse_args()
keys = fixtures.slab_keys()[:: a.every]
labels = [fixtures.fixture_labels(k) for k in keys]
dev = fixtures.open_device(a.device)
stats = fixtures.StatsSum()
t = time.perf_counter()
for call in fixtures.label_calls(labels):
    dev.skeletonize(call, **fixtures.SKEL_PUBLISHED)
    stats.add(dev)
print(f"skeletonize {len(keys)} slabs on {dev.name}: {time.perf_counter() - t:.2f} s; {stats}")
