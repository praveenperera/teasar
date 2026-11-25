//! The Python module `teasar._core`
//!
//! Every call releases the GIL while the device works; a device's mutex lets one call at a time run on it

use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::{
    Block, Device, Error, Field, Labels, LabelsRef, Method, Packed, SettingsSet, SlabKind, SlabSpec, TargetOrder, Value,
};

create_exception!(_core, BackendUnavailable, PyRuntimeError, "No supported GPU, or the OS or GPU family is too old");
create_exception!(_core, UnsupportedSettings, PyNotImplementedError, "Settings outside the supported sets");
create_exception!(_core, InputTooLarge, PyValueError, "An input over one of the per-call limits");
create_exception!(
    _core,
    OutputTooLarge,
    PyValueError,
    "A call whose results would exceed the retained-output limit; `slab` is the index, in the call's slab list, of \
     the first slab whose result takes the running total over it"
);
create_exception!(
    _core,
    SomaUnsupported,
    PyNotImplementedError,
    "With target_order='kimimaro', a piece whose largest DBF exceeds 1,100 physical units; `slab` is the index of \
     the first such slab. Never raised with target_order='best'"
);
create_exception!(_core, GpuError, PyRuntimeError, "A failed GPU submission or validation error");

fn to_py(py: Python<'_>, e: Error) -> PyErr {
    let msg = e.to_string();
    match e {
        Error::BackendUnavailable(_) => BackendUnavailable::new_err(msg),
        Error::UnsupportedSettings(_) => UnsupportedSettings::new_err(msg),
        Error::InputTooLarge(_) => InputTooLarge::new_err(msg),
        Error::OutputTooLarge { slab, .. } => {
            let err = OutputTooLarge::new_err(msg);
            // villa reruns the slabs before `slab` and continues from it in a new call
            match err.value(py).setattr("slab", slab) {
                Ok(()) => err,
                Err(e) => e,
            }
        }
        Error::SomaUnsupported { slab, .. } => {
            let err = SomaUnsupported::new_err(msg);
            // villa runs that slab on its CPU path and the others here
            match err.value(py).setattr("slab", slab) {
                Ok(()) => err,
                Err(e) => e,
            }
        }
        Error::Gpu(_) => GpuError::new_err(msg),
        Error::Value(_) => PyValueError::new_err(msg),
        Error::NotImplemented(_) => PyNotImplementedError::new_err(msg),
        Error::Closed => PyRuntimeError::new_err(msg),
    }
}

/// The skeletons of one slab, as numpy arrays
#[pyclass(name = "Packed", module = "teasar", get_all, frozen)]
pub struct PyPacked {
    ids: Py<PyArray1<u32>>,
    v_off: Py<PyArray1<i64>>,
    e_off: Py<PyArray1<i64>>,
    vertices: Py<PyArray2<f32>>,
    edges: Py<PyArray2<u32>>,
    radii: Py<PyArray1<f32>>,
}

#[pymethods]
impl PyPacked {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Packed(n={}, vertices={}, edges={})",
            self.ids.bind(py).len(),
            self.vertices.bind(py).shape()[0],
            self.edges.bind(py).shape()[0]
        )
    }
}

fn packed_to_py(py: Python<'_>, p: Packed) -> PyResult<PyPacked> {
    let nv = p.vertices.len();
    let ne = p.edges.len();
    let vertices = Array2::from_shape_vec((nv, 3), p.vertices.into_flattened()).expect("V x 3 vertices");
    let edges = Array2::from_shape_vec((ne, 2), p.edges.into_flattened()).expect("E x 2 edges");
    Ok(PyPacked {
        ids: PyArray1::from_owned_array(py, Array1::from_vec(p.ids)).unbind(),
        v_off: PyArray1::from_owned_array(py, Array1::from_vec(p.v_off)).unbind(),
        e_off: PyArray1::from_owned_array(py, Array1::from_vec(p.e_off)).unbind(),
        vertices: PyArray2::from_owned_array(py, vertices).unbind(),
        edges: PyArray2::from_owned_array(py, edges).unbind(),
        radii: PyArray1::from_owned_array(py, Array1::from_vec(p.radii)).unbind(),
    })
}

fn labels_to_py(py: Python<'_>, l: Labels) -> Py<PyArray3<u32>> {
    let a = Array3::from_shape_vec(l.shape, l.data).expect("labels match their shape");
    PyArray3::from_owned_array(py, a).unbind()
}

/// A C-contiguous 3D uint8 or bool array, read in place
fn binary_arg<'py>(obj: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArray3<'py, u8>> {
    if let Ok(a) = obj.cast::<PyArray3<u8>>() {
        return contiguous(a.readonly(), "binary");
    }
    if let Ok(a) = obj.cast::<PyArray3<bool>>() {
        // numpy bools are one byte each, 0 or 1, so the same memory reads as uint8
        let view = a.call_method1("view", (numpy::dtype::<u8>(obj.py()),))?;
        return contiguous(view.cast_into::<PyArray3<u8>>()?.readonly(), "binary");
    }
    Err(PyValueError::new_err("binary must be a 3D numpy array of uint8 or bool"))
}

fn contiguous<'py, T: numpy::Element>(a: PyReadonlyArray3<'py, T>, what: &str) -> PyResult<PyReadonlyArray3<'py, T>> {
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err(format!("{what} must be C-contiguous")));
    }
    Ok(a)
}

fn block_arg<'py>(obj: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArray3<'py, u8>> {
    let a = obj.cast::<PyArray3<u8>>().map_err(|_| PyValueError::new_err("block must be a 3D numpy array of uint8"))?;
    contiguous(a.readonly(), "block")
}

fn slabs_arg(slabs: &[(String, i64, i64)]) -> crate::Result<Vec<SlabSpec>> {
    slabs.iter().map(|(kind, lo, hi)| Ok(SlabSpec { kind: SlabKind::parse(kind)?, lo: *lo, hi: *hi })).collect()
}

/// A settings argument as the caller passed it
///
/// Conversion never fails, so a value of the wrong type raises `UnsupportedSettings` from the settings check, like
/// any other value outside the named set, rather than a `TypeError`
struct Setting(Value);

impl FromPyObject<'_, '_> for Setting {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        // bool first: Python bools are ints, and would otherwise pass as the numbers 0 and 1
        if let Ok(b) = obj.extract::<bool>() {
            return Ok(Self(Value::Bool(b)));
        }
        if let Ok(x) = obj.extract::<f64>() {
            return Ok(Self(Value::Number(x)));
        }
        if let Ok(t) = obj.extract::<String>() {
            return Ok(Self(Value::Text(t)));
        }
        if let Ok(xs) = obj.extract::<Vec<f64>>() {
            return Ok(Self(Value::Numbers(xs)));
        }
        Ok(Self(Value::Other(obj.repr().map_or_else(|_| "<unprintable>".into(), |r| r.to_string()))))
    }
}

/// Checks a call's settings against its row of the accepted-settings table
fn accept(py: Python<'_>, method: Method, settings: &str, given: Vec<(Field, Setting)>) -> PyResult<SettingsSet> {
    let given: Vec<(Field, Value)> = given.into_iter().map(|(f, v)| (f, v.0)).collect();
    SettingsSet::accept(method, settings, &given).map_err(|e| to_py(py, e))
}

/// fix_borders after the settings check, which admits only a bool
fn fix_borders_value(v: &Setting) -> bool {
    matches!(v.0, Value::Bool(true))
}

/// Whether `target_order` asks for compatibility mode; the settings check rejects every value but the two names
fn kimimaro_value(v: &Setting) -> bool {
    matches!(&v.0, Value::Text(t) if t == "kimimaro")
}

/// The orders compatibility mode takes from the running interpreter, compiled once: a fresh
/// built-in set the picks are added to in order, and numpy's default argsort
const ORDERS_SOURCE: &std::ffi::CStr = c"
import numpy

def set_order(picks):
    s = set()
    for p in picks:
        s.add(p)
    first = {}
    for i, p in enumerate(picks):
        first.setdefault(p, i)
    return [first[p] for p in s]

def set_orders(lists):
    return [set_order(p) for p in lists]

def argsorts(ds):
    return [numpy.argsort(d) for d in ds]
";

/// [`crate::CompatOrders`] through the interpreter, taking the GIL for each call
struct PyOrders {
    set_orders: Py<PyAny>,
    argsorts: Py<PyAny>,
}

impl PyOrders {
    fn new(py: Python<'_>) -> PyResult<Self> {
        let m = PyModule::from_code(py, ORDERS_SOURCE, c"compat_orders.py", c"compat_orders")?;
        Ok(Self { set_orders: m.getattr("set_orders")?.unbind(), argsorts: m.getattr("argsorts")?.unbind() })
    }
}

fn py_err(what: &str, e: PyErr) -> Error {
    Error::Value(format!("{what} failed in Python: {e}"))
}

/// `f` under the GIL, adding the wait since `asked` and the time held to the compatibility-mode profile
fn gil_timed<T>(asked: std::time::Instant, f: impl FnOnce(Python<'_>) -> T) -> T {
    use crate::cpu::km::prof;
    Python::attach(|py| {
        let got = std::time::Instant::now();
        prof::add_wall(prof::GIL_WAIT, (got - asked).as_secs_f64());
        let out = f(py);
        prof::add_wall(prof::GIL_HELD, got.elapsed().as_secs_f64());
        out
    })
}

impl crate::CompatOrders for PyOrders {
    fn set_orders(&self, lists: &[&[[u32; 3]]]) -> crate::Result<Vec<Vec<u32>>> {
        let asked = std::time::Instant::now();
        gil_timed(asked, |py| {
            let lists = lists
                .iter()
                .map(|picks| PyList::new(py, picks.iter().map(|p| PyTuple::new(py, p)).collect::<PyResult<Vec<_>>>()?))
                .collect::<PyResult<Vec<_>>>()?;
            self.set_orders.call1(py, (PyList::new(py, lists)?,))?.extract::<Vec<Vec<u32>>>(py)
        })
        .map_err(|e| py_err("the set order", e))
    }

    fn argsorts(&self, lists: &[&[f32]]) -> crate::Result<Vec<Vec<u32>>> {
        let asked = std::time::Instant::now();
        gil_timed(asked, |py| {
            let arrays: Vec<_> = lists.iter().map(|d| PyArray1::from_slice(py, d)).collect();
            let out = self.argsorts.call1(py, (PyList::new(py, arrays)?,))?;
            let out = out.bind(py).cast::<PyList>().map_err(PyErr::from)?;
            out.iter()
                .map(|idx| {
                    let idx = idx.cast::<PyArray1<i64>>().map_err(PyErr::from)?.readonly();
                    Ok(idx.as_slice()?.iter().map(|&k| k as u32).collect())
                })
                .collect::<PyResult<Vec<Vec<u32>>>>()
        })
        .map_err(|e| py_err("numpy's argsort", e))
    }
}

/// A device behind its call mutex; `None` once closed
struct Inner(Mutex<Option<Device>>);

impl Inner {
    fn with<T>(&self, f: impl FnOnce(&mut Device) -> crate::Result<T>) -> crate::Result<T> {
        let mut guard = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let dev = guard.as_mut().ok_or(Error::Closed)?;
        f(dev)
    }

    /// Runs `f` on the device with the GIL released, so another Python thread can work meanwhile
    fn run<T: Send>(&self, py: Python<'_>, f: impl FnOnce(&mut Device) -> crate::Result<T> + Send) -> PyResult<T> {
        py.detach(|| self.with(f)).map_err(|e| to_py(py, e))
    }
}

/// The methods `Device` and `CpuDevice` share
macro_rules! device_methods {
    ($ty:ident, { $($ctor:tt)* }) => {
        #[pymethods]
        impl $ty {
            $($ctor)*

            /// "metal" or "cpu"
            #[getter]
            fn backend(&self, py: Python<'_>) -> PyResult<&'static str> {
                self.0.run(py, |d| Ok(d.backend()))
            }

            /// The device's name
            #[getter]
            fn name(&self, py: Python<'_>) -> PyResult<String> {
                self.0.run(py, |d| Ok(d.name()))
            }

            /// The statistics of the device's most recent call: `wall_s`, `gpu_busy_s`,
            /// `host_threads`, `cpu_pieces`, `cpu_voxels`, `gpu_pieces`, `gpu_voxels` and `cpu_busy_s`
            fn last_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                let s = self.0.run(py, |d| Ok(d.last_stats()))?;
                let out = PyDict::new(py);
                out.set_item("wall_s", s.wall_s)?;
                out.set_item("gpu_busy_s", s.gpu_busy_s)?;
                out.set_item("host_threads", s.host_threads)?;
                out.set_item("cpu_pieces", s.cpu_pieces)?;
                out.set_item("cpu_voxels", s.cpu_voxels)?;
                out.set_item("gpu_pieces", s.gpu_pieces)?;
                out.set_item("gpu_voxels", s.gpu_voxels)?;
                out.set_item("cpu_busy_s", s.cpu_busy_s)?;
                Ok(out)
            }

            /// (bytes this process has allocated on the device, recommended working set)
            fn memory(&self, py: Python<'_>) -> PyResult<(u64, u64)> {
                self.0.run(py, |d| Ok(d.memory()))
            }

            /// Releases the device; later calls raise
            fn close(&self, py: Python<'_>) {
                py.detach(|| {
                    let mut guard = self.0.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    guard.take();
                });
            }

            fn __enter__(slf: Py<Self>) -> Py<Self> {
                slf
            }

            fn __exit__(
                &self,
                py: Python<'_>,
                _ty: &Bound<'_, PyAny>,
                _value: &Bound<'_, PyAny>,
                _tb: &Bound<'_, PyAny>,
            ) -> bool {
                self.close(py);
                false
            }

            /// The pre-skeleton stages for one slab: pooled, filtered uint32 labels
            ///
            /// `settings` names the set ('published_0211' or 'villa_default'), and `downsample`, `min_count` and
            /// `max_count` must be that set's
            #[pyo3(signature = (binary, downsample, min_count, max_count, *, settings))]
            fn prepare_labels(
                &self,
                py: Python<'_>,
                binary: &Bound<'_, PyAny>,
                downsample: Setting,
                min_count: Setting,
                max_count: Setting,
                settings: &str,
            ) -> PyResult<Py<PyArray3<u32>>> {
                let given =
                    vec![(Field::Downsample, downsample), (Field::MinCount, min_count), (Field::MaxCount, max_count)];
                let set = accept(py, Method::PrepareLabels, settings, given)?;
                let arr = binary_arg(binary)?;
                let shape: [usize; 3] = arr.shape().try_into().expect("3D");
                let data = arr.as_slice().expect("checked C-contiguous");
                let labels = self.0.run(py, |d| d.prepare_labels(data, shape, set))?;
                Ok(labels_to_py(py, labels))
            }

            /// The pre-skeleton stages of `extract`: one label array per slab, None for a thin slab
            #[pyo3(signature = (block, x_size, slabs, *, settings, downsample, min_count, max_count))]
            // the keyword arguments are the Python API's signature
            #[allow(clippy::too_many_arguments)]
            fn prepare_from_block(
                &self,
                py: Python<'_>,
                block: &Bound<'_, PyAny>,
                x_size: usize,
                slabs: Vec<(String, i64, i64)>,
                settings: &str,
                downsample: Setting,
                min_count: Setting,
                max_count: Setting,
            ) -> PyResult<Py<PyList>> {
                let given =
                    vec![(Field::Downsample, downsample), (Field::MinCount, min_count), (Field::MaxCount, max_count)];
                let set = accept(py, Method::PrepareFromBlock, settings, given)?;
                let arr = block_arg(block)?;
                let shape: [usize; 3] = arr.shape().try_into().expect("3D");
                let blk = Block::new(arr.as_slice().expect("checked C-contiguous"), shape, x_size)
                    .map_err(|e| to_py(py, e))?;
                let specs = slabs_arg(&slabs).map_err(|e| to_py(py, e))?;
                let out = self.0.run(py, |d| d.prepare_from_block(&blk, &specs, set))?;
                let items: Vec<Option<Py<PyArray3<u32>>>> =
                    out.into_iter().map(|l| l.map(|l| labels_to_py(py, l))).collect();
                Ok(PyList::new(py, items)?.unbind())
            }

            /// The skeletons of each pooled label array
            #[pyo3(signature = (labels_list, *, settings, anisotropy, scale = Setting(Value::Number(1.0)),
                                r#const = Setting(Value::Number(2.0)), dust_threshold,
                                fix_borders = Setting(Value::Bool(true)),
                                fix_branching = Setting(Value::Bool(true)),
                                target_order = Setting(Value::Text("best".into()))))]
            #[allow(clippy::too_many_arguments)]
            fn skeletonize(
                &self,
                py: Python<'_>,
                labels_list: Vec<Bound<'_, PyAny>>,
                settings: &str,
                anisotropy: Setting,
                scale: Setting,
                r#const: Setting,
                dust_threshold: Setting,
                fix_borders: Setting,
                fix_branching: Setting,
                target_order: Setting,
            ) -> PyResult<Vec<PyPacked>> {
                let borders = fix_borders_value(&fix_borders);
                let kimimaro = kimimaro_value(&target_order);
                let given = vec![
                    (Field::Anisotropy, anisotropy),
                    (Field::DustThreshold, dust_threshold),
                    (Field::Scale, scale),
                    (Field::Const, r#const),
                    (Field::FixBranching, fix_branching),
                    (Field::FixBorders, fix_borders),
                    (Field::TargetOrder, target_order),
                ];
                let set = accept(py, Method::Skeletonize, settings, given)?;
                let arrays = labels_list
                    .iter()
                    .map(|a| {
                        let a = a
                            .cast::<PyArray3<u32>>()
                            .map_err(|_| PyValueError::new_err("each labels array must be 3D uint32"))?;
                        contiguous(a.readonly(), "labels")
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let refs: Vec<LabelsRef<'_>> = arrays
                    .iter()
                    .map(|a| LabelsRef {
                        shape: a.shape().try_into().expect("3D"),
                        data: a.as_slice().expect("checked C-contiguous"),
                    })
                    .collect();
                let orders = if kimimaro { Some(PyOrders::new(py)?) } else { None };
                let order = orders.as_ref().map_or(TargetOrder::Best, |o| TargetOrder::Kimimaro(o));
                let out = self.0.run(py, |d| d.skeletonize(&refs, set, borders, order))?;
                out.into_iter().map(|p| packed_to_py(py, p)).collect()
            }

            /// The fused path: pre-skeleton stages and skeletons for each slab of a bit-packed block
            #[pyo3(signature = (block, x_size, slabs, *, settings, downsample, min_count, max_count, anisotropy,
                                scale = Setting(Value::Number(1.0)), r#const = Setting(Value::Number(2.0)),
                                dust_threshold, fix_borders = Setting(Value::Bool(true)),
                                fix_branching = Setting(Value::Bool(true)),
                                target_order = Setting(Value::Text("best".into()))))]
            #[allow(clippy::too_many_arguments)]
            fn extract(
                &self,
                py: Python<'_>,
                block: &Bound<'_, PyAny>,
                x_size: usize,
                slabs: Vec<(String, i64, i64)>,
                settings: &str,
                downsample: Setting,
                min_count: Setting,
                max_count: Setting,
                anisotropy: Setting,
                scale: Setting,
                r#const: Setting,
                dust_threshold: Setting,
                fix_borders: Setting,
                fix_branching: Setting,
                target_order: Setting,
            ) -> PyResult<Vec<PyPacked>> {
                let borders = fix_borders_value(&fix_borders);
                let kimimaro = kimimaro_value(&target_order);
                let given = vec![
                    (Field::Downsample, downsample),
                    (Field::MinCount, min_count),
                    (Field::MaxCount, max_count),
                    (Field::Anisotropy, anisotropy),
                    (Field::DustThreshold, dust_threshold),
                    (Field::Scale, scale),
                    (Field::Const, r#const),
                    (Field::FixBranching, fix_branching),
                    (Field::FixBorders, fix_borders),
                    (Field::TargetOrder, target_order),
                ];
                let set = accept(py, Method::Extract, settings, given)?;
                let arr = block_arg(block)?;
                let shape: [usize; 3] = arr.shape().try_into().expect("3D");
                let blk = Block::new(arr.as_slice().expect("checked C-contiguous"), shape, x_size)
                    .map_err(|e| to_py(py, e))?;
                let specs = slabs_arg(&slabs).map_err(|e| to_py(py, e))?;
                let orders = if kimimaro { Some(PyOrders::new(py)?) } else { None };
                let order = orders.as_ref().map_or(TargetOrder::Best, |o| TargetOrder::Kimimaro(o));
                let out = self.0.run(py, |d| d.extract(&blk, &specs, set, borders, order))?;
                out.into_iter().map(|p| packed_to_py(py, p)).collect()
            }

            /// Test hook: replaces any of the per-call limits
            #[cfg(feature = "testing")]
            #[pyo3(signature = (*, block_bytes = None, axis_pooled = None, slab_voxels = None, slab_pooled = None,
                                call_pooled = None, output_bytes = None))]
            #[allow(clippy::too_many_arguments)]
            fn _set_limits(
                &self,
                py: Python<'_>,
                block_bytes: Option<u64>,
                axis_pooled: Option<u64>,
                slab_voxels: Option<u64>,
                slab_pooled: Option<u64>,
                call_pooled: Option<u64>,
                output_bytes: Option<u64>,
            ) -> PyResult<()> {
                self.0.run(py, |d| {
                    let mut l = d.limits();
                    let given = [
                        (crate::Limit::BlockBytes, block_bytes),
                        (crate::Limit::AxisPooled, axis_pooled),
                        (crate::Limit::SlabVoxels, slab_voxels),
                        (crate::Limit::SlabPooled, slab_pooled),
                        (crate::Limit::CallPooled, call_pooled),
                        (crate::Limit::OutputBytes, output_bytes),
                    ];
                    for (limit, v) in given {
                        if let Some(v) = v {
                            l.set(limit, v);
                        }
                    }
                    d.set_limits(l);
                    Ok(())
                })
            }

            /// Test hook: caps the pooled cells of one internal batch
            #[cfg(feature = "testing")]
            fn _set_batch_cells(&self, py: Python<'_>, cells: usize) -> PyResult<()> {
                self.0.run(py, |d| {
                    d.backend_mut().set_batch_cells(cells);
                    Ok(())
                })
            }

            /// Test hook: copy-kernel bandwidth in GB/s
            #[cfg(feature = "testing")]
            fn _bench_copy(&self, py: Python<'_>, bytes: usize, iters: usize) -> PyResult<f64> {
                self.0.run(py, |d| d.backend_mut().bench_copy(bytes, iters))
            }

            /// Test hook: a kernel that writes out of bounds, for the validation negative control
            #[cfg(feature = "testing")]
            fn _oob_probe(&self, py: Python<'_>) -> PyResult<()> {
                self.0.run(py, |d| d.backend_mut().oob_probe())
            }

            /// Test hook: a GPU self-test (`atomic64`, `float_rules`), returning its report
            #[cfg(feature = "testing")]
            fn _selftest(&self, py: Python<'_>, name: &str) -> PyResult<String> {
                self.0.run(py, |d| d.backend_mut().selftest(name))
            }

            /// Test hook: the field-level dump of each labels array under target_order='best',
            /// as dicts of flat uint32 arrays `dbf`, `b`, `d`, `p` (float32 bits by slab voxel, 0xFFFFFFFF outside
            /// traced pieces) and `roots` (slab voxel indices, ascending)
            #[cfg(feature = "testing")]
            #[pyo3(signature = (labels_list, *, settings, anisotropy, dust_threshold,
                                fix_borders = Setting(Value::Bool(true))))]
            fn _fields<'py>(
                &self,
                py: Python<'py>,
                labels_list: Vec<Bound<'py, PyAny>>,
                settings: &str,
                anisotropy: Setting,
                dust_threshold: Setting,
                fix_borders: Setting,
            ) -> PyResult<Vec<Bound<'py, PyDict>>> {
                let borders = fix_borders_value(&fix_borders);
                let given = vec![
                    (Field::Anisotropy, anisotropy),
                    (Field::DustThreshold, dust_threshold),
                    (Field::Scale, Setting(Value::Number(1.0))),
                    (Field::Const, Setting(Value::Number(2.0))),
                    (Field::FixBranching, Setting(Value::Bool(true))),
                    (Field::FixBorders, fix_borders),
                    (Field::TargetOrder, Setting(Value::Text("best".into()))),
                ];
                let set = accept(py, Method::Skeletonize, settings, given)?;
                let arrays = labels_list
                    .iter()
                    .map(|a| {
                        let a = a
                            .cast::<PyArray3<u32>>()
                            .map_err(|_| PyValueError::new_err("each labels array must be 3D uint32"))?;
                        contiguous(a.readonly(), "labels")
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let refs: Vec<LabelsRef<'_>> = arrays
                    .iter()
                    .map(|a| LabelsRef {
                        shape: a.shape().try_into().expect("3D"),
                        data: a.as_slice().expect("checked C-contiguous"),
                    })
                    .collect();
                let skel = set.skel_settings(borders);
                let dumps = self.0.run(py, |d| d.backend_mut().fields(&refs, &skel))?;
                dumps
                    .into_iter()
                    .map(|f| {
                        let out = PyDict::new(py);
                        for (k, v) in [("dbf", f.dbf), ("b", f.b), ("d", f.d), ("p", f.p), ("roots", f.roots)] {
                            out.set_item(k, PyArray1::from_owned_array(py, Array1::from_vec(v)))?;
                        }
                        Ok(out)
                    })
                    .collect()
            }

            /// Test hook: makes the next GPU submission fail
            #[cfg(feature = "testing")]
            fn _inject_gpu_failure(&self, py: Python<'_>) -> PyResult<()> {
                self.0.run(py, |d| {
                    d.backend_mut().inject_gpu_failure();
                    Ok(())
                })
            }
        }
    };
}

/// The platform's GPU backend (Metal on macOS)
#[pyclass(name = "Device", module = "teasar", frozen)]
struct GpuDevice(Inner);

/// The CPU reference
#[pyclass(name = "CpuDevice", module = "teasar", frozen)]
struct CpuDevice(Inner);

device_methods!(GpuDevice, {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        let dev = py.detach(Device::gpu).map_err(|e| to_py(py, e))?;
        Ok(Self(Inner(Mutex::new(Some(dev)))))
    }
});

device_methods!(CpuDevice, {
    #[new]
    #[pyo3(signature = (threads = 0))]
    fn new(py: Python<'_>, threads: usize) -> PyResult<Self> {
        let dev = Device::cpu(threads).map_err(|e| to_py(py, e))?;
        Ok(Self(Inner(Mutex::new(Some(dev)))))
    }
});

/// Test hook: the labels of a pooled array that get a skeleton, in skeleton order
#[cfg(feature = "testing")]
#[pyfunction]
fn _skeleton_ids(labels: PyReadonlyArray3<'_, u32>, dust_threshold: u32) -> PyResult<Vec<u32>> {
    let shape: [usize; 3] = labels.shape().try_into().expect("3D");
    let data = labels.as_slice().map_err(|_| PyValueError::new_err("labels must be C-contiguous"))?;
    let l = LabelsRef { shape, data };
    let pieces = crate::cpu::pieces::pieces26(&l);
    Ok(crate::cpu::pieces::skeleton_ids(&l, &pieces, dust_threshold))
}

/// (True, device name) or (False, the reason the GPU backend can't open); never raises
#[pyfunction]
fn check_available(py: Python<'_>) -> (bool, String) {
    py.detach(Device::check_available)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TESTING", cfg!(feature = "testing"))?;
    m.add_class::<GpuDevice>()?;
    m.add_class::<CpuDevice>()?;
    m.add_class::<PyPacked>()?;
    m.add_function(wrap_pyfunction!(check_available, m)?)?;
    #[cfg(feature = "testing")]
    m.add_function(wrap_pyfunction!(_skeleton_ids, m)?)?;
    m.add("BackendUnavailable", py.get_type::<BackendUnavailable>())?;
    m.add("UnsupportedSettings", py.get_type::<UnsupportedSettings>())?;
    m.add("InputTooLarge", py.get_type::<InputTooLarge>())?;
    m.add("OutputTooLarge", py.get_type::<OutputTooLarge>())?;
    m.add("SomaUnsupported", py.get_type::<SomaUnsupported>())?;
    m.add("GpuError", py.get_type::<GpuError>())?;
    Ok(())
}
