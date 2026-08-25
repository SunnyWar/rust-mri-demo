# Rust MRI Processing Demo (`rust-mri-demo`)

A high-performance, multi-threaded command-line tool written in Rust for batch processing 3D and 4D NIfTI neuroimaging datasets (`.nii.gz`).

Designed as a modern alternative to traditional MATLAB/SPM processing scripts, this tool leverages shared-memory thread parallelism (`rayon`) and explicit memory alignment (`ndarray`, `nalgebra`) to achieve significant speedups across large BIDS datasets.

---

## Features

- **Automated Dataset Traversal:** Recursively discovers NIfTI files (`.nii.gz`) across nested BIDS structures without hardcoded directory assumptions.
- **Dynamic Dimensionality Support:** Automatically detects 3D structural volumes (T1w) and 4D time-series/diffusion datasets (DWI).
- **Stage 1: Diffusion Tensor Imaging (DTI):** 
  - Fits a 3x3 symmetric Diffusion Tensor per voxel via least-squares pseudoinverse solver.
  - Computes Fractional Anisotropy (FA) scalar maps from tensor eigenvalues ($\lambda_1, \lambda_2, \lambda_3$).
  - Parses adjacent `.bvec` gradient files when available (falls back to synthetic gradient orientations if unmapped).
- **Stage 2: 3D Parallel Spatial Smoothing:** 
  - Applies a 3D separable Gaussian spatial filter ($3\times1\text{D}$ convolutions across spatial axes) parallelized via Rayon thread pooling.
- **Stage 3: Non-Linear Perfusion Scaling:** 
  - Performs high-throughput element-wise signal transformations ($v \cdot e^{-\alpha v}$) in-place.

---

## Prerequisites

- [Rust Toolchain](https://www.rust-lang.org/tools/install) (1.75+ recommended)
- Crate dependencies (`Cargo.toml`):
  - `nifti` (0.17)
  - `ndarray` (0.16 with `rayon` feature)
  - `rayon`
  - `nalgebra`
  - `clap`

---

## Build

Compile an optimized release binary:

```cmd
cargo build --release
```

---

## Usage

Usage: rust-mri-demo [OPTIONS] <ROOT>

Arguments:
  <ROOT>  Root folder containing NIfTI dataset to process recursively

Options:
  -a, --alpha <ALPHA>  Perfusion alpha scaling parameter [default: 0.01]
  -h, --help           Print help information
  -V, --version        Print version information

  ---