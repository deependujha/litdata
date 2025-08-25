use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBool, PyFloat, PyLong};
use std::fs;

pub trait Serializer {
    /// Serialize data into a byte vector and optional metadata string
    fn serialize<'py>(&self, data: Bound<'py, PyAny>) -> PyResult<(Vec<u8>, Option<String>)>;

    /// Deserialize from bytes back to Python object
    fn deserialize(&self, data: &[u8], py: Python) -> PyResult<PyObject>;

    /// Check if the data can be serialized by this serializer
    fn can_serialize<'py>(&self, data: Bound<'py, PyAny>) -> bool;

    /// Optional setup hook (e.g., for metadata)
    fn setup(&mut self, _metadata: Option<&PyAny>) -> PyResult<()> {
        Ok(())
    }
}

pub struct IntegerSerializer;

impl IntegerSerializer {
    pub fn new() -> Self {
        Self
    }
}

impl Serializer for IntegerSerializer {
    fn serialize<'py>(&self, data: Bound<'py, PyAny>) -> PyResult<(Vec<u8>, Option<String>)> {
        let val: i64 = data.extract()?; // now uses extract_bound internally
        Ok((val.to_le_bytes().to_vec(), None))
    }

    fn deserialize(&self, data: &[u8], py: Python) -> PyResult<PyObject> {
        if data.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid byte length for i64",
            ));
        }
        let val = i64::from_le_bytes(data.try_into().unwrap());
        Ok(val.into_py(py))
    }

    fn can_serialize<'py>(&self, data: Bound<'py, PyAny>) -> bool {
        data.is_instance_of::<PyLong>()
    }
}

pub struct FloatSerializer;

impl FloatSerializer {
    pub fn new() -> Self {
        Self
    }
}

impl Serializer for FloatSerializer {
    fn serialize<'py>(&self, data: Bound<'py, PyAny>) -> PyResult<(Vec<u8>, Option<String>)> {
        let val: f64 = data.extract()?; // Bound supports extract directly
        Ok((val.to_le_bytes().to_vec(), None))
    }

    fn deserialize(&self, data: &[u8], py: Python) -> PyResult<PyObject> {
        if data.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid byte length for f64",
            ));
        }
        let val = f64::from_le_bytes(data.try_into().unwrap());
        Ok(val.into_py(py))
    }

    fn can_serialize<'py>(&self, data: Bound<'py, PyAny>) -> bool {
        data.is_instance_of::<PyFloat>()
    }
}

/// String Serializer

pub struct StringSerializer;

impl StringSerializer {
    pub fn new() -> Self {
        Self
    }
}

impl Serializer for StringSerializer {
    fn serialize<'py>(&self, data: Bound<'py, PyAny>) -> PyResult<(Vec<u8>, Option<String>)> {
        // Own the string to avoid any lifetime ties to 'py
        let s: String = data.extract()?;
        Ok((s.into_bytes(), None))
    }

    fn deserialize(&self, data: &[u8], py: Python) -> PyResult<PyObject> {
        let s = std::str::from_utf8(data)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid UTF-8 data"))?;
        Ok(s.into_py(py))
    }

    fn can_serialize<'py>(&self, data: Bound<'py, PyAny>) -> bool {
        // Extract owned String; if it's a path to a file, reject
        if let Ok(s) = data.extract::<String>() {
            fs::metadata(&s).is_err()
        } else {
            false
        }
    }
}

/// Boolean Serializer
pub struct BooleanSerializer;

impl BooleanSerializer {
    pub fn new() -> Self {
        Self
    }
}

impl Serializer for BooleanSerializer {
    fn serialize<'py>(&self, data: Bound<'py, PyAny>) -> PyResult<(Vec<u8>, Option<String>)> {
        let val: bool = data.extract()?;
        // Single-byte representation, 1 for true, 0 for false
        Ok((vec![val as u8], None))
    }

    fn deserialize(&self, data: &[u8], py: Python) -> PyResult<PyObject> {
        if data.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid byte length for bool",
            ));
        }
        let val = data[0] != 0;
        Ok(val.into_py(py))
    }

    fn can_serialize<'py>(&self, data: Bound<'py, PyAny>) -> bool {
        data.is_instance_of::<PyBool>()
    }
}
