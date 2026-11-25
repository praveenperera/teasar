//! The error kinds the Python API maps to its exceptions

use std::fmt;

/// A failed call, by the exception it becomes at the Python boundary
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// No supported GPU, or the OS or GPU family is too old
    BackendUnavailable(String),
    /// Settings outside the supported sets
    UnsupportedSettings(String),
    /// An input over a limit; the message names the limit
    InputTooLarge(String),
    /// A call whose returned arrays would exceed the retained-output limit; `slab` indexes the call's slab list at
    /// the first slab whose result takes the running total over it
    OutputTooLarge { slab: usize, message: String },
    /// With `target_order='kimimaro'`, a piece whose largest DBF exceeds the compatibility limit; `slab` indexes the
    /// call's slab list at the first such slab
    SomaUnsupported { slab: usize, message: String },
    /// A failed GPU submission or validation error
    Gpu(String),
    /// Malformed arrays, thin slabs in `prepare_labels`, out-of-range slabs
    Value(String),
    /// Work this build does not do yet
    NotImplemented(String),
    /// The device was closed
    Closed,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BackendUnavailable(m)
            | Error::UnsupportedSettings(m)
            | Error::InputTooLarge(m)
            | Error::OutputTooLarge { message: m, .. }
            | Error::SomaUnsupported { message: m, .. }
            | Error::Gpu(m)
            | Error::Value(m)
            | Error::NotImplemented(m) => f.write_str(m),
            Error::Closed => f.write_str("the device is closed"),
        }
    }
}

impl std::error::Error for Error {}

/// Raises [`Error::Value`] with a formatted message unless the condition holds
#[macro_export]
macro_rules! ensure_value {
    ($cond:expr, $($arg:tt)+) => {
        if !$cond {
            return Err($crate::Error::Value(format!($($arg)+)));
        }
    };
}
