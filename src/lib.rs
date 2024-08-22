mod bridge;
mod grammar;
mod parser;

use crate::bridge::bnf::extension_bnf;
use crate::bridge::charts::extension_charts;
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_native")]
fn incremental_parsing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    extension_bnf(m)?;
    extension_charts(m)?;
    Ok(())
}
