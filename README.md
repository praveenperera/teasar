# teasar

TEASAR skeletonisation of [villa](https://github.com/ScrollPrize/villa)'s surface-track slabs, written in Rust with
PyO3 bindings. Its GPU backend is Metal, for Apple GPUs; a pure-Rust CPU reference sits behind the same API.
Each backend is a Rust module behind its own Cargo feature (`metal`), so other GPU backends can sit beside it.

teasar is an independent implementation. It contains no code from Kimimaro or Brook; it was written from its
specification, the TEASAR paper and published GPU algorithms.

## Credits

- Sato, Bitter, Bender, Kaufman and Nakajima, "TEASAR: tree-structure extraction algorithm for accurate and robust
  skeletons", Pacific Graphics 2000, doi:10.1109/PCCGA.2000.883951
- [Kimimaro](https://github.com/seung-lab/kimimaro) (seung-lab), the reference whose behaviour this package matches

## Install

```sh
pip install teasar          # or: uv pip install teasar
```

PyPI has prebuilt wheels, one abi3 wheel per platform for CPython 3.11 or newer, so installing needs no Rust toolchain:

| platform | wheel | backends |
|---|---|---|
| macOS on Apple silicon (arm64) | `macosx_11_0_arm64` | Metal and the CPU reference |
| Linux x86_64 and arm64 (glibc 2.28 or newer) | `manylinux_2_28` | the CPU reference |

The Metal backend needs macOS 26 or newer and an M1 or newer GPU; elsewhere `Device()` raises `BackendUnavailable`
and the CPU reference, `CpuDevice()`, still works. Check the install:

```sh
python -c "import teasar; print(teasar.check_available())"   # (True, 'Apple M4 Pro') on a supported Mac
```

### Building from source

On any other platform pip builds teasar from PyPI's source distribution, which needs a stable Rust toolchain
([rustup](https://rustup.rs); on macOS also the Xcode command line tools, `xcode-select --install`). To build it
yourself:

```sh
pip install --no-binary teasar teasar                             # PyPI's source distribution
pip install git+https://github.com/praveenperera/teasar          # or the repository
```

A source build enables the `python` and `metal` features from `pyproject.toml`. The `metal` feature is a no-op off
macOS, so a Linux build has the CPU reference only.

### From Rust

The same code is the `teasar` crate on crates.io:

```sh
cargo add teasar
```

Its `metal` feature is on by default and does nothing off macOS; `teasar::Device::gpu()` opens the Metal backend and
`teasar::Device::cpu(threads)` the CPU reference.

## Python API

```python
import teasar as tm

ok, reason = tm.check_available()          # (True, "Apple M4 Pro") or (False, why); never raises
dev = tm.Device()                          # Metal; raises tm.BackendUnavailable elsewhere
cpu = tm.CpuDevice(threads=0)              # the CPU reference, all cores

labels = dev.prepare_labels(binary, 2, 50, 80000, settings="published_0211")
labels = dev.prepare_from_block(block, x_size, slabs, settings="published_0211", downsample=2, min_count=50,
                                max_count=80000)
packed = dev.skeletonize(labels_list, settings="published_0211", anisotropy=(2, 2, 2), dust_threshold=49)
packed = dev.extract(block, x_size, slabs, settings="published_0211", downsample=2, min_count=50, max_count=80000,
                     anisotropy=(2, 2, 2), dust_threshold=49)
dev.last_stats()   # {"wall_s", "gpu_busy_s", "host_threads", "cpu_pieces", "cpu_voxels", "gpu_pieces", ...}
```

- `binary` is a C-contiguous (Z, Y, X) `uint8` or `bool` array; `block` is a (Z, Y, ceil(x_size / 8)) `uint8` array,
  bit-packed along x with the lowest x in the most significant bit (`np.packbits(..., axis=-1, bitorder="big")`).
- `slabs` is a list of `(kind, lo, hi)` with kind `'h'` (z range), `'vy'` (y range) or `'vx'` (x range), block-local.
- Labels equal villa's `prepare_cc_labels` bit for bit: 6-connected components at full resolution numbered in C-order
  scan order of each component's first voxel, cropped to a multiple of `downsample`, max-pooled, and filtered to
  `min_count <= count < max_count` pooled voxels.
- `skeletonize` and `extract` take `target_order='best'` (the default): border targets first, then ordinary targets
  farthest-first by geodesic distance from the root.
- `target_order='kimimaro'` is a compatibility mode that reproduces Kimimaro's skeletons as villa calls it: its
  face picks, root, target order, path and claim rules. It takes two orders from the running
  Python interpreter, a fresh built-in set's iteration order and numpy's default `argsort`, so its output matches only
  with CPython and numpy versions whose orders equal the tested ones (CPython 3.14.10, numpy 2.5.2). Both backends run
  its skeleton stage on host threads; the Metal backend runs the prepare stages on the GPU first. A call with a piece
  whose largest boundary distance exceeds 1,100 raises `tm.SomaUnsupported` before any path search, with `slab` set
  to the first such slab; villa then runs that slab on its CPU path. On the test fixtures its skeletons equal
  Kimimaro's on 16 synthetic masks, and its leaf and branch-point counts on a 384-slab test box are within 0.7% of
  Kimimaro's.
- Each skeleton is in sorted form: vertices in increasing C-order index of their voxel, edges written (lower,
  higher), edge rows sorted.
- In `target_order='best'` the root is the first border target, the penalty is the README's
  `(1 - B / max B)^16` plus `D / max D`, and invalidation removes the piece voxels strictly inside the ball of radius
  `scale * DBF + const` around each path vertex.

### Settings

Every method takes a required keyword `settings` naming one of the two sets villa runs, and every other settings
value it passes must equal that set's. The set is never inferred from the values. Anything else raises
`tm.UnsupportedSettings` (a `NotImplementedError`), naming the field, the value passed and the set's value.

| set | `downsample` | `min_count` | `max_count` | `anisotropy` | `dust_threshold` | `scale` | `const` | `fix_branching` | `h` depth | vertical thickness |
|---|---|---|---|---|---|---|---|---|---|---|
| `published_0211` | 2 | 50 | 80000 | (2, 2, 2) | 49 | 1.0 | 2.0 | True | 2 | 4 |
| `villa_default` | 4 | 6 | 10000 | (4, 4, 4) | 5 | 1.0 | 2.0 | True | 4 | 4 |

| method | checked against the set |
|---|---|
| `prepare_labels` | `downsample`, `min_count`, `max_count` |
| `prepare_from_block` | the same, and every slab's thickness |
| `skeletonize` | `anisotropy`, `dust_threshold`, `scale`, `const`, `fix_branching`; `fix_borders` is True or False |
| `extract` | all of the above, and every slab's thickness |

- Equality is numeric (`2` equals `2.0`; `anisotropy` is any sequence of three numbers), and the bool fields must be
  bools.
- A slab's thickness (`hi - lo` on its axis) must be the set's `h` depth or vertical thickness. A slab that ends at
  the block's edge on that axis may be thinner; one thinner than `downsample` returns an empty `Packed` from
  `extract` and `None` from `prepare_from_block`.
- `fix_borders=True` turns border targets off for a slab whose pooled shape has two or more axes of length 1, as
  villa does.

### Limits

Input checks run before any work and raise `tm.InputTooLarge`; the output check raises `tm.OutputTooLarge`. Both are
`ValueError`s whose message names the limit.

| limit | value |
|---|---|
| `block.nbytes` | 512 MiB |
| each pooled axis of a slab (or of a `skeletonize` label array) | 32,768 voxels |
| one slab, full resolution | 3 × 2^26 voxels |
| one slab, pooled | 2^24 voxels |
| one call, pooled voxels summed over its slabs | 2^26 |
| the arrays a call returns | 512 MiB |

- `OutputTooLarge.slab` is the index, in the call's slab list (thin slabs included), of the first slab whose result
  takes the running total over the limit. Rerun the slabs before it in one call and continue from it in a new one.
- Split a long slab list with the baseline partition: keep the slabs in order and start a new call whenever the next
  slab would take the call's pooled total over 2^26 (`tests/fixtures.py`'s `slab_calls`; 6 calls on the 384-slab test box).
- Calls release the GIL; one call at a time runs on a device. Returned arrays are new and owned by the caller.
- `TEASAR_PROFILE=1` prints per-kernel GPU time and the internal batching to stderr after each Metal call.

### Host pieces

The Metal skeletoniser runs the slab stages and every piece of at most 1,000 pooled voxels on the GPU, and traces larger
pieces on host threads beside the GPU's path loop, with the CPU reference's code. `TEASAR_CPU_THREADS` sets the threads,
read once per `Device` (default: the performance cores; the thread driving the GPU is extra). `dev.last_stats()` reports
the split for every call, and `TEASAR_PROFILE=1` prints it with the host pieces' time. The output does not depend on the
split: every piece's skeleton is the same, bit for bit, on either side. Tuning knobs, for experiments only:

| variable | default | what |
|---|---|---|
| `TEASAR_CPU_PIECE` | 1000 | pieces over this many pooled voxels run on host threads |
| `TEASAR_GPU_BUDGET` | 2^16 | path-loop work units per GPU lane before it pauses |
| `TEASAR_LARGE_PIECE` | the size filter's maximum | pieces over this many pooled voxels fall back to host threads; none do at the supported settings |
| `TEASAR_CPU_FALLBACK=0` | off | makes that fallback an error, for tests |
| `TEASAR_SKEL_BATCH_VOXELS` | 2^23 | pooled voxels per internal skeletonize batch |
| `TEASAR_PROFILE_BUCKETS=1` | off | runs and times the path loop by piece-size bucket (slower) |

With `TEASAR_PROFILE=1` the profile also counts resume dispatches, pieces paused and pauses by phase, and splits the
host pieces into hybrid-split and fallback pieces.

## CPU reference contract

The CPU reference returns exactly what the Metal backend returns. Its connected components use a different algorithm
(union-find over x-runs) from the GPU's (union-find over the 6-connected pieces of each pooled cell), so each checks
the other.

The CPU reference's skeletons equal the Metal backend's bit for bit on the fixture slabs in both target orders, and a
`testing` build's field-level dump (`_fields`) shows each piece's DBF, `B`, `D`, `P` and root from the GPU path loop
equal the CPU's bit for bit, on the test box's slabs and the synthetic masks. In `target_order='kimimaro'` both
backends run the same host code for the skeleton stage.

## Development

```sh
uv venv .venv && source .venv/bin/activate
uv pip install numpy scipy zarr pytest maturin
maturin develop --uv --release -F python,metal,testing   # testing adds the test hooks
```

`maturin build --release` makes the release wheel; like `pip install .`, it enables `python` and `metal` from
`pyproject.toml`, and `test_macos_build_has_metal` fails if a macOS build lacks the Metal backend.

## Tests

`pytest tests` runs the test suite on the CPU reference, on a `testing` build. GPU runs set `TEASAR_GPU=1`
(`scripts/gpu_tests.sh`, and `scripts/gpu_validate.sh` under Metal's API and shader validation). The fixture set,
villa and the parity harness are not part of this repo, so their paths have no defaults:

```sh
export TEASAR_FIXTURES=/path/to/fixtures        # the fixture root, holding data/, ref/ and villa/
export TEASAR_DATA=$TEASAR_FIXTURES/data        # the fixture set's inputs and per-slab fixtures
export TEASAR_VILLA=$TEASAR_FIXTURES/villa      # a villa checkout, for spiral-fitting/fast_tracks.py
export TEASAR_HARNESS=/path/to/parity/harness   # run_candidate.py, compare.py and skel_counts.py
```

Tests that read the fixture set skip with a message naming `TEASAR_DATA` when it is unset, and the track checks on
the synthetic masks name `TEASAR_VILLA`. `scripts/parity_check.sh`, `hash_lock.sh` and `count_check.py` run the
harness on the test box against the reference in the fixture root's `ref/`, on the repo's `.venv`, or with `$TEASAR_PYTHON` on another interpreter, such
as a clean venv with a release wheel installed. The timing scripts
(`scripts/bench_extract.py`, `bench_prep.py`, `prof_skel.py`) print `last_stats()`, summed over each pass's calls,
beside every timing; set `TEASAR_CPU_THREADS` for the host threads.

## Licence

MIT, copyright Praveen Perera.
