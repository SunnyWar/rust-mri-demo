use flate2::read::GzDecoder;
use ndarray::ArrayD;
use nifti::{InMemNiftiObject, IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::ProcessError;

pub struct Volume {
    pub data: ArrayD<f32>,
    pub header: NiftiHeader,
}

pub fn load_nifti(path: &str) -> Result<Volume, ProcessError> {
    let file_path = Path::new(path);

    let file = File::open(file_path).map_err(ProcessError::Io)?;

    let obj = if path.ends_with(".gz") {
        let mut gz = GzDecoder::new(BufReader::new(file));
        let mut buffer = Vec::new();
        gz.read_to_end(&mut buffer).map_err(ProcessError::Io)?;

        InMemNiftiObject::from_reader(&buffer[..])
            .map_err(|e| ProcessError::Io(std::io::Error::other(e)))?
    } else {
        ReaderOptions::new()
            .read_file(file_path)
            .map_err(|e| ProcessError::Io(std::io::Error::other(e)))?
    };

    let header = obj.header().clone();

    let data = obj
        .into_volume()
        .into_ndarray::<f32>()
        .map_err(|e| ProcessError::Io(std::io::Error::other(e)))?;

    Ok(Volume { data, header })
}
