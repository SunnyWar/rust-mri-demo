use flate2::read::GzDecoder;
use ndarray::ArrayD;
use nifti::{
    InMemNiftiObject, IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions, writer::WriterOptions,
};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub struct Volume {
    pub data: ArrayD<f32>,
    pub header: NiftiHeader,
}

pub fn load_nifti(path: &str) -> Volume {
    let file_path = Path::new(path);
    let file = File::open(file_path).unwrap_or_else(|e| panic!("Failed to open {}: {}", path, e));

    let obj = if path.ends_with(".gz") {
        let mut gz_decoder = GzDecoder::new(BufReader::new(file));
        let mut buffer = Vec::new();
        gz_decoder
            .read_to_end(&mut buffer)
            .unwrap_or_else(|e| panic!("Failed to decompress {}: {}", path, e));

        let file =
            File::open(file_path).unwrap_or_else(|e| panic!("Failed to open {}: {}", path, e));
        let gz_decoder = GzDecoder::new(BufReader::new(file));

        InMemNiftiObject::from_reader(gz_decoder)
            .unwrap_or_else(|e| panic!("Failed to parse NIfTI stream for {}: {}", path, e))
    } else {
        ReaderOptions::new()
            .read_file(file_path)
            .unwrap_or_else(|e| panic!("Failed to read NIfTI file {}: {}", path, e))
    };

    let header = obj.header().clone();
    let data = obj
        .into_volume()
        .into_ndarray::<f32>()
        .unwrap_or_else(|e| panic!("Failed to extract volume data for {}: {}", path, e));

    Volume { data, header }
}

pub fn save_nifti(vol: &Volume, path: &str) {
    let file_path = Path::new(path);

    WriterOptions::new(file_path)
        .reference_header(&vol.header)
        .write_nifti(&vol.data)
        .unwrap_or_else(|e| panic!("Failed to write NIfTI file {}: {}", path, e));
}
