use std::fs;
use std::path::{Path, PathBuf};

use crate::ProcessError;

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

    // Fallback synthetic gradient directions
    let mut gradients = Vec::with_capacity(n_dirs);
    gradients.push([0.0, 0.0, 0.0]); // baseline S0

    for i in 1..n_dirs {
        let theta = i as f32 * 0.5;
        let phi = i as f32 * 0.3;
        gradients.push([theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos()]);
    }

    Ok(gradients)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::fs::File;
    use tempfile::tempdir;

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
}
