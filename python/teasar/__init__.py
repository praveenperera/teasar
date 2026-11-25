"""TEASAR skeletonisation of villa's surface-track slabs, with a Metal backend for Apple GPUs and a CPU reference.

``Device()`` opens the platform's GPU backend and ``CpuDevice()`` the pure-Rust CPU reference. Both expose
``prepare_labels``, ``prepare_from_block``, ``skeletonize`` and ``extract`` with the same results::

    labels = dev.prepare_labels(binary, 2, 50, 80000, settings="published_0211")
    packed = dev.extract(block, x_size, slabs, settings="published_0211", downsample=2, min_count=50,
                         max_count=80000, anisotropy=(2, 2, 2), dust_threshold=49)

Every method takes a required keyword ``settings``, ``"published_0211"`` or ``"villa_default"``, and every other
settings value must equal that set's; anything else raises ``UnsupportedSettings``. Inputs over a limit raise
``InputTooLarge`` before any work. A call whose results would exceed the output limit raises ``OutputTooLarge``,
whose ``slab`` attribute is the index of the first slab that takes the running total over it.

``skeletonize`` and ``extract`` take ``target_order="best"`` (the default) or ``"kimimaro"``, a compatibility mode
that reproduces the reference skeletons villa's CPU path makes. It takes two orders from the running interpreter
(a built-in set's iteration order and numpy's default argsort), and a call with a piece whose largest boundary
distance exceeds 1,100 raises ``SomaUnsupported``, whose ``slab`` attribute names the first such slab.
``dev.last_stats()`` describes the device's most recent call: wall and GPU-busy time, host threads, and where the
pieces ran.
"""

from ._core import (
    BackendUnavailable,
    CpuDevice,
    Device,
    GpuError,
    InputTooLarge,
    OutputTooLarge,
    Packed,
    SomaUnsupported,
    UnsupportedSettings,
    __version__,
    check_available,
)

__all__ = [
    "BackendUnavailable",
    "CpuDevice",
    "Device",
    "GpuError",
    "InputTooLarge",
    "OutputTooLarge",
    "Packed",
    "SomaUnsupported",
    "UnsupportedSettings",
    "__version__",
    "check_available",
]
