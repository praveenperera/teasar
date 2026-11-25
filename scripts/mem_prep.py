"""Interim A8: a fresh process runs prepare_from_block on all 384 B1 slabs with the block in memory (extract needs
part B); reports the device allocation high-water mark. Run under /usr/bin/time -l for the peak footprint."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import fixtures as F  # noqa: E402
import teasar as tm  # noqa: E402

blk = F.block()
dev = tm.Device()
high = 0
kept = 0
for c in F.slab_calls([F.slab_spec(k) for k in F.slab_keys()]):
    out = dev.prepare_from_block(blk, F.YX, c, **F.PUBLISHED)
    high = max(high, dev.memory()[0])
    kept += sum(o.nbytes for o in out)
    del out
print(f"device allocation high water {high / 2**20:.1f} MiB; labels returned {kept / 2**20:.1f} MiB in total")
