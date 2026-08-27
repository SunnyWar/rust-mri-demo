use std::fs;
use std::path::{Path, PathBuf};

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
