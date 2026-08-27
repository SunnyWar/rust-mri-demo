use std::fs;
use std::path::{Path, PathBuf};

use crate::pipeline::ProcessError;
use ndarray::Array3;
use nifti::NiftiHeader;
use nifti::writer::WriterOptions;

pub fn find_nii_gz_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Warning: Skipping directory {}: {}", root.display(), err);
            return files;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            files.extend(find_nii_gz_files(&path));
        } else {
            let filename = path.file_name().unwrap_or_default().to_string_lossy();

            // Match .nii.gz files, but ignore previously generated outputs
            if filename.ends_with(".nii.gz") && !filename.contains("_processed") {
                files.push(path);
            }
        }
    }

    files
}

// Helper to attempt reading associated .bvec gradient files for 4D DWI scans
pub fn load_gradients(nifti_path: &Path, n_dirs: usize) -> Result<Vec<[f32; 3]>, ProcessError> {
    let bvec_path = nifti_path.with_extension("").with_extension("bvec");

    // Try reading .bvec file
    if bvec_path.exists() {
        let content = fs::read_to_string(&bvec_path).map_err(ProcessError::Io)?;
        let lines: Vec<&str> = content.lines().collect();

        if lines.len() >= 3 {
            let parse_line = |line: &str| -> Result<Vec<f32>, ProcessError> {
                line.split_whitespace()
                    .map(|s| {
                        s.parse::<f32>().map_err(|e| {
                            ProcessError::Io(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Invalid float in {}: {}", bvec_path.display(), e),
                            ))
                        })
                    })
                    .collect()
            };

            let xs = parse_line(lines[0])?;
            let ys = parse_line(lines[1])?;
            let zs = parse_line(lines[2])?;

            if xs.len() == n_dirs && ys.len() == n_dirs && zs.len() == n_dirs {
                let grads = (0..n_dirs)
                    .map(|i| [xs[i], ys[i], zs[i]])
                    .collect::<Vec<_>>();
                return Ok(grads);
            }
        }
    }

    // Fallback: synthetic, well-distributed gradient directions
    Ok(electrostatic_gradients(n_dirs))
}

pub fn electrostatic_gradients(n_dirs: usize) -> Vec<[f32; 3]> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    if n_dirs == 0 {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

    let mut points = Vec::with_capacity(n_dirs);
    points.push([0.0, 0.0, 0.0]); // S0 baseline

    // rand 0.10: use random::<f32>() instead of gen_range
    for _ in 1..n_dirs {
        let z: f32 = rng.random::<f32>() * 2.0 - 1.0; // [-1, 1]
        let t: f32 = rng.random::<f32>() * std::f32::consts::TAU; // [0, 2π]
        let r = (1.0 - z * z).sqrt();
        points.push([r * t.cos(), r * t.sin(), z]);
    }

    let iterations = 200;
    let step0 = 0.05;

    for iter in 0..iterations {
        let step = step0 * (1.0 - (iter as f32 / iterations as f32));
        let mut forces = vec![[0.0f32; 3]; n_dirs];

        for i in 1..n_dirs {
            for j in 1..n_dirs {
                if i == j {
                    continue;
                }

                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                let dz = points[i][2] - points[j][2];
                let dist2 = dx * dx + dy * dy + dz * dz + 1e-6;
                // Coulomb force ~ 1/r^2 along the unit vector => dx/r^3
                let inv3 = 1.0 / (dist2 * dist2.sqrt());

                forces[i][0] += dx * inv3;
                forces[i][1] += dy * inv3;
                forces[i][2] += dz * inv3;
            }
        }

        for i in 1..n_dirs {
            points[i][0] += step * forces[i][0];
            points[i][1] += step * forces[i][1];
            points[i][2] += step * forces[i][2];

            let norm = (points[i][0] * points[i][0]
                + points[i][1] * points[i][1]
                + points[i][2] * points[i][2])
                .sqrt();

            if norm > 1e-12 {
                points[i][0] /= norm;
                points[i][1] /= norm;
                points[i][2] /= norm;
            }
        }
    }

    points
}

pub fn get_processed_output_path(file: &Path, suffix: &str) -> PathBuf {
    let filename = file.file_name().unwrap().to_string_lossy();

    // Strip trailing .nii.gz or .nii
    let stem = if let Some(stripped) = filename.strip_suffix(".nii.gz") {
        stripped
    } else if let Some(stripped) = filename.strip_suffix(".nii") {
        stripped
    } else {
        &filename
    };

    file.with_file_name(format!("{}_{}.nii.gz", stem, suffix))
}

pub fn save_scalar_map(
    vol: &Array3<f32>,
    file: &Path,
    suffix: &str,
    header: &NiftiHeader,
) -> Result<(), ProcessError> {
    let out = get_processed_output_path(file, suffix);

    WriterOptions::new(&out)
        .reference_header(header)
        .write_nifti(vol)
        .map_err(|e| {
            ProcessError::Io(std::io::Error::other(format!(
                "Failed to write NIfTI file {}: {}",
                out.display(),
                e
            )))
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::{f32, fs};
    use tempfile::tempdir;

    // --- get_processed_output_path tests ---

    #[test]
    fn test_get_processed_output_path_strips_extensions() {
        let p_gz = Path::new("/data/subject01_dwi.nii.gz");
        let out_fa = get_processed_output_path(p_gz, "fa_processed");
        assert_eq!(
            out_fa,
            PathBuf::from("/data/subject01_dwi_fa_processed.nii.gz")
        );

        let p_nii = Path::new("/data/subject01_t1.nii");
        let out_smooth = get_processed_output_path(p_nii, "smoothed_processed");
        assert_eq!(
            out_smooth,
            PathBuf::from("/data/subject01_t1_smoothed_processed.nii.gz")
        );
    }

    #[test]
    fn test_find_nii_gz_files_recurses_and_filters_outputs() -> std::io::Result<()> {
        let dir = tempdir()?;
        let sub_dir = dir.path().join("sub-01");
        fs::create_dir(&sub_dir)?;

        // Valid raw scans
        let scan1 = dir.path().join("raw1.nii.gz");
        let scan2 = sub_dir.join("dwi.nii.gz");
        // Output file to skip
        let processed = sub_dir.join("dwi_fa_processed.nii.gz");
        // Non-NIfTI file
        let txt_file = dir.path().join("notes.txt");

        File::create(&scan1)?;
        File::create(&scan2)?;
        File::create(&processed)?;
        File::create(&txt_file)?;

        let mut found = find_nii_gz_files(dir.path());
        found.sort();

        let mut expected = vec![scan1, scan2];
        expected.sort();

        assert_eq!(found, expected);
        Ok(())
    }

    // --- load_gradients tests ---

    #[test]
    fn test_load_gradients_from_valid_bvec_file() -> std::io::Result<()> {
        let dir = tempdir()?;
        let nifti_path = dir.path().join("dwi.nii.gz");
        let bvec_path = dir.path().join("dwi.bvec");

        File::create(&nifti_path)?;
        let mut f = File::create(&bvec_path)?;
        // Write 3x3 matrix (3 diffusion directions)
        writeln!(f, "1.0  0.0  0.70710677")?;
        writeln!(f, "0.0  1.0  0.0")?;
        writeln!(f, "0.0  0.0  0.70710677")?;

        let grads = load_gradients(&nifti_path, 3).unwrap();
        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0], [1.0, 0.0, 0.0]);
        assert_eq!(grads[1], [0.0, 1.0, 0.0]);
        assert_eq!(
            grads[2],
            [f32::consts::FRAC_1_SQRT_2, 0.0, f32::consts::FRAC_1_SQRT_2]
        );

        Ok(())
    }

    #[test]
    fn test_load_gradients_fallback_when_bvec_missing() {
        let dir = tempdir().unwrap();
        let nifti_path = dir.path().join("dwi_nobvec.nii.gz");

        let grads = load_gradients(&nifti_path, 4).unwrap();
        assert_eq!(grads.len(), 4);

        // First gradient direction must be baseline b=0 [0, 0, 0]
        assert_eq!(grads[0], [0.0, 0.0, 0.0]);

        // Synthetic directions must be normalized / distinct non-zero vectors
        for g in &grads[1..] {
            let norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            assert!(norm > 0.0);
        }
    }

    #[test]
    fn test_load_gradients_fallback_on_dimension_mismatch() -> std::io::Result<()> {
        let dir = tempdir()?;
        let nifti_path = dir.path().join("dwi.nii.gz");
        let bvec_path = dir.path().join("dwi.bvec");

        File::create(&nifti_path)?;
        let mut f = File::create(&bvec_path)?;
        // 2 directions in bvec file, but 3 expected
        writeln!(f, "1.0 0.0")?;
        writeln!(f, "0.0 1.0")?;
        writeln!(f, "0.0 0.0")?;

        let grads = load_gradients(&nifti_path, 3).unwrap();
        assert_eq!(grads.len(), 3);
        // Fallback synthetic baseline triggers due to count mismatch
        assert_eq!(grads[0], [0.0, 0.0, 0.0]);

        Ok(())
    }
}
