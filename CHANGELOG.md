# Changelog

All notable changes to teasar are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-24

The first release on PyPI; 0.1.0 was published to crates.io only. The library's behaviour and output are the same as
0.1.0's.

### Added

- Prebuilt wheels: macOS arm64 with the Metal backend, and Linux x86_64 and
  arm64 with the CPU reference, each one abi3 wheel for CPython 3.11 and newer.

### Changed

- The tests and helper scripts are named for what they check:
  `tests/test_prepare.py` and `tests/test_skeleton.py`, and the parity scripts
  `scripts/parity_check.sh`, `scripts/hash_lock.sh` and `scripts/count_check.py`.
- The Metal shader libraries that `metal::gpu::check_libraries` reports are
  named `prep` and `skel`.

### Removed

- Development notes: the clean-room record, the README's status table, and the
  scripts that only timed or tracked the package while it was being built.

## [0.1.0] - 2026-11-24

### Added

- TEASAR skeletonisation of villa's surface-track slabs, from Python
  (`import teasar`) and Rust: `prepare_labels`, `prepare_from_block`,
  `skeletonize` and `extract`, with villa's two settings sets
  (`published_0211` and `villa_default`) and checked per-call limits.
- A Metal backend for Apple GPUs (`teasar.Device()`, macOS 26 or newer on an M1
  or newer GPU) and a pure-Rust CPU reference (`teasar.CpuDevice()`) behind the
  same API, whose skeletons are identical bit for bit.
- `target_order='kimimaro'`, a compatibility mode that reproduces Kimimaro's
  skeletons as villa calls it, beside the default `target_order='best'`.

[Unreleased]: https://github.com/praveenperera/teasar/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/praveenperera/teasar/releases/tag/v0.1.1
[0.1.0]: https://crates.io/crates/teasar/0.1.0
