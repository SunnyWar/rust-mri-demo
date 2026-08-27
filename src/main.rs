mod dti;
mod imaging;

use clap::Parser;
use rayon::prelude::*;
use std::{path::Path, time::Instant};
mod cli;
mod pipeline;

use crate::{cli::Cli, imaging::io::find_nii_gz_files, pipeline::process_file};

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
