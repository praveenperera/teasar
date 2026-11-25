# Clean-room record

teasar is written from a specification and the sources listed below only. No source code of Kimimaro, Brook,
dijkstra3d, edt, cc3d, fill_voids, fastremap or osteoid was read, copied or paraphrased while writing it. For those
tools, only their public READMEs and package descriptions were consulted, as listed below; osteoid's were not opened
at all.

## Process

The work was split between two teams that shared no code.

- The specification team ran the reference tools (Kimimaro as villa calls it, and Brook), built the test fixtures and
  the parity harness, and wrote the specification: the package's API, formats and limits, the skeletoniser's
  behaviour, and the acceptance tests with their expected outputs.
- The implementation team wrote this repository from the specification alone. It never read the reference tools'
  source, never opened their installed packages or repositories, and asked the specification team whenever the
  specification left a behaviour undefined, rather than looking it up.

The first release of the specification covered the package and the skeletoniser in its own words. Matching the
reference's skeletons exactly (`target_order='kimimaro'`) needed more: the specification team then wrote an addendum
that describes the reference's observable behaviour (its face picks, root, target order, path and claim rules, and a
table of the platform's floating-point power results where they differ from correctly rounded ones). Before the
addendum reached the implementation team it was reviewed for leakage: it states behaviour, observed outputs and test
cases, not code, code structure or names taken from the reference's source.

The private development history marks this boundary with an annotated tag, `clean-room-phase1`, on the last commit
written before the addendum was released (Mon Nov 23, 2026, 2:52 PM CT; the addendum followed at about 2:55 PM CT).
Everything up to that tag was written without the addendum; later commits may use it.

## Sources

| source | what was used | licence |
|---|---|---|
| The specification, parts A and B and the compatibility addendum | the package, API, formats, limits, pre-skeleton stages and tests; the skeletoniser's behaviour, design and tests; the behaviour of `target_order='kimimaro'` and its power table (`data/km6_pow.txt`) | written for this package by the specification team; behaviour only, no code |
| villa `spiral-fitting/extract_surface_tracks.py` (ScrollPrize/villa) | `prepare_cc_labels`, `downsample_maxpool`, `skeletonize_kwargs`, the slab settings: the input and output contract only | MIT |
| villa `vc_reader/src/vc_reader/geometry.py` | the docstring and argument handling of `label_components`: the numbering and dtype contract | MIT |
| villa `spiral-fitting/fast_tracks.py` | imported by the tests to turn `Packed` into tracks; only its function list and module docstring were read | MIT |
| inkrs `src/metal/gpu.rs`, by the same author | the Metal host layer, adapted in `src/metal/gpu.rs` with attribution | MIT OR Apache-2.0 |
| objc2, objc2-foundation, objc2-metal, pyo3 and rust-numpy crate sources | API signatures | objc2 family: MIT; pyo3 and numpy: MIT OR Apache-2.0 |
| Kimimaro's public README (its PyPI project description) | the "TEASAR Derived Algorithm" section and the `teasar_params` descriptions: field definitions, the invalidation cube, parameter meanings | GPL-3.0-or-later project; README only, no code |
| dijkstra3d's public README (its PyPI project description) | that a step from A to B costs B's field value | GPL-3.0-or-later project; README only, no code |
| The specification team's parity harness | how the acceptance self-check is run and judged | written for this project |
| Sato, Bitter, Bender, Kaufman and Nakajima, "TEASAR", Pacific Graphics 2000 (doi:10.1109/PCCGA.2000.883951); Bitter, Kaufman and Sato 2001, penalized-distance skeletons | the algorithm family; the behaviour implemented is the specification's | published papers |
| Playne and Hawick, "A new algorithm for parallel connected-component labelling on GPUs", IEEE TPDS 2018 | the lock-free union-find merge with atomic min | published paper |
| Felzenszwalb and Huttenlocher, "Distance transforms of sampled functions", Theory of Computing 2012 | the lower-envelope 1D distance transform used per run in `src/cpu/edt.rs` | published paper |

## Black-box runs and one incident

Early on Mon Nov 23, 2026, before the rules were tightened, the implementation team ran Kimimaro as a black box on
a handful of small synthetic masks (bars, an L, combs and nubbed bars) and inspected only the outputs, to settle the
invalidation bound and the root rule. During one of those runs a NumPy warning raised inside Kimimaro printed one
source line and its file path. That line was not used for any design decision, and every later probe ran with
warnings suppressed. The same day the rule became stricter: the implementation team no longer ran any of the
reference tools, even as a black box, and no probe has run since. Apart from that one printed line, no reference
source text was seen.

The acceptance self-check runs the specification team's harness on the implementation's own Python environment,
which holds none of the reference packages, and passes the reference outputs to the harness's comparison scripts by
path; their contents were not read by hand.
