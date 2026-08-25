use ndarray::ArrayD;
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions, writer::WriterOptions};

pub struct Volume {
    pub data: ArrayD<f32>,
    pub header: NiftiHeader,
}

pub fn load_nifti(path: &str) -> Volume {
    let obj = ReaderOptions::new()
        .read_file(path)
        .expect("failed to load nifti");
    let header = obj.header().clone();

    // Keeps the natural dimensionality (3D, 4D, etc.) as ArrayD
    let data = obj
        .into_volume()
        .into_ndarray::<f32>()
        .expect("failed to extract volume data");

    Volume { data, header }
}

pub fn save_nifti(vol: &Volume, path: &str) {
    WriterOptions::new(path)
        .reference_header(&vol.header)
        .write_nifti(&vol.data)
        .expect("failed to write nifti");
}
