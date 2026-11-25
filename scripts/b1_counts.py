#!/usr/bin/env python3
"""B1's and B8's count checks (spec part B §4.1): leaves and branch points per pass against the reference counts and band.

Skeletonizes the fixture label arrays over the baseline partition, writes their counts in the harness's counts JSON
(via its skel_counts.counts) and runs `skel_counts.py compare` against the set's reference counts and band; exits as
that does. `b1` is the 384 published_0211 slabs against ref/counts_ref.json and ref/count_band.json; `b8` is the 144
villa_default ds4 slabs against ref/counts_ref_ds4.json and ref/count_band_ds4.json.

usage: b1_counts.py [cpu|gpu] OUT.json [b1|b8] [best|kimimaro]
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import fixtures as F  # noqa: E402
import teasar as tm  # noqa: E402

# the parity harness directory, with skel_counts.py; it is not part of this repo, so there is no default
if not os.environ.get("TEASAR_HARNESS"):
    sys.exit("b1_counts.py: set TEASAR_HARNESS to the parity harness directory (the one holding skel_counts.py)")
HARNESS = os.path.expanduser(os.environ["TEASAR_HARNESS"])
# the reference counts and bands sit beside data/ in the fixture set
REF = F.data().parent / "ref"
sys.path.insert(0, HARNESS)
import skel_counts  # noqa: E402


# per set: fixture root, skeletonize settings, and the suffix of its ref/ counts and band files
SETS = {
    "b1": (F.B1, F.SKEL_PUBLISHED, ""),
    "b8": (F.B1_DS4, F.SKEL_VILLA, "_ds4"),
}


def main():
    kind, out = sys.argv[1], sys.argv[2]
    root, skel, suffix = SETS[sys.argv[3] if len(sys.argv) > 3 else "b1"]
    order = sys.argv[4] if len(sys.argv) > 4 else "best"
    dev = F.open_device(kind)
    keys = F.slab_keys(root)
    res = {}
    # sizes first, then one call's labels at a time, so the whole set is never in memory
    sizes = {k: F.fixture_labels(k, root).size for k in keys}
    for call in F.baseline_partition(keys, sizes.__getitem__):
        packed = dev.skeletonize([F.fixture_labels(k, root) for k in call], **skel, target_order=order)
        for k, p in zip(call, packed, strict=True):
            res[k] = skel_counts.counts(p.v_off, p.e_off, p.edges)
    assert len(res) == len(keys)
    with open(out, "w") as f:
        json.dump({"keys": res}, f)
    cmd = [sys.executable, f"{HARNESS}/skel_counts.py", "compare", str(REF / f"counts_ref{suffix}.json"), out,
           "--band", str(REF / f"count_band{suffix}.json")]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
