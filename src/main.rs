mod dti;
mod imaging;

use clap::Parser;
use rayon::prelude::*;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
mod pipeline;

use crate::{dti::types::DtiError, imaging::io::find_nii_gz_files, pipeline::process_file};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    root: String,

    #[arg(short = 's', long = "sigma-3d", default_value_t = 1.5)]
    sigma_3d: f32,

    #[arg(long = "sigma-fa", default_value_t = 1.0)]
    sigma_fa: f32,

    #[arg(short, long, default_value_t = 1000.0)]
    bvalue: f32,

    /// Also output a mean diffusivity map
    #[arg(long)]
    emit_md: bool,

    /// Also output an axial diffusivity map
    #[arg(long)]
    emit_ad: bool,

    /// Also output a radial diffusivity map
    #[arg(long)]
    emit_rd: bool,
}

#[derive(Debug)]
enum ProcessError {
    UnexpectedShape { path: PathBuf, ndim: usize },
    TensorFitFailed { path: PathBuf, reason: DtiError },
    Io(std::io::Error),
}

fn main() {
    let args = Cli::parse();
    let root = Path::new(&args.root);
    let all_start = Instant::now();
    let files = find_nii_gz_files(root);

    println!("Found {} NIfTI files", files.len());
    println!("Processing dataset in parallel with Rayon...");

    // Parallelize pipeline across files concurrently
    let results: Vec<_> = files
        .par_iter()
        .map(|file| (file, process_file(file, &args)))
        .collect();

    for (file, result) in &results {
        if let Err(e) = result {
            eprintln!("FAILED: {} — {:?}", file.display(), e);
        }
    }
    let failed = results.iter().filter(|(_, r)| r.is_err()).count();
    println!(
        "Files processed: {} ({} failed)",
        results.len() - failed,
        failed
    );

    let all_elapsed = all_start.elapsed();
    println!("----------------------------------------");
    println!(
        "All processing completed in {:.3} seconds",
        all_elapsed.as_secs_f32()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CLI argument parsing tests ---

    #[test]
    fn test_cli_defaults() {
        let args = Cli::try_parse_from(["dti_pipeline", "/path/to/data"]).unwrap();
        assert_eq!(args.root, "/path/to/data");
        assert_eq!(args.sigma_3d, 1.5);
        assert_eq!(args.sigma_fa, 1.0);
        assert_eq!(args.bvalue, 1000.0);
        assert!(!args.emit_md);
        assert!(!args.emit_ad);
        assert!(!args.emit_rd);
    }

    #[test]
    fn test_cli_custom_flags() {
        let args = Cli::try_parse_from([
            "dti_pipeline",
            "/path/to/data",
            "-s",
            "2.5",
            "--sigma-fa",
            "0.8",
            "-b",
            "1500.0",
            "--emit-md",
            "--emit-rd",
        ])
        .unwrap();

        assert_eq!(args.sigma_3d, 2.5);
        assert_eq!(args.sigma_fa, 0.8);
        assert_eq!(args.bvalue, 1500.0);
        assert!(args.emit_md);
        assert!(!args.emit_ad);
        assert!(args.emit_rd);
    }
}
