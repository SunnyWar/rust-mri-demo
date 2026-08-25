# Rust MRI Processing Demo (`rust-mri-demo`)

A high-performance, multi-threaded command-line tool written in Rust for batch processing 3D and 4D NIfTI neuroimaging datasets (`.nii.gz`).

Designed as a modern alternative to traditional MATLAB/SPM processing scripts, this tool leverages shared-memory thread parallelism (`rayon`) and explicit memory alignment (`ndarray`, `nalgebra`) to achieve significant speedups across large BIDS datasets.

---

## Scientific Assumptions & Pipeline Positioning

> **Note on Pipeline Context:** `rust-mri-demo` is designed as a ultra-fast compute engine for **BIDS derivatives**. It assumes input NIfTI volumes have undergone baseline motion correction, eddy current correction, and skull stripping via standard front-end tooling (e.g., fMRIPrep, QSIPrep).

### Microstructural & Signal Models
1. **Diffusion Tensor Imaging (DTI):** Monocompartment tensor fit using weighted linear least-squares (WLLS) on log-attenuated signals:
   $$\mathbf{S}(b) = S_0 \cdot e^{-b \cdot \mathbf{g}^T \mathbf{D} \mathbf{g}}$$
   Fractional Anisotropy (FA) is derived directly from the primary eigenvalues ($\lambda_1, \lambda_2, \lambda_3$) of $\mathbf{D}$.
2. **Spatial Anisotropy Handling:** Gaussian smoothing scales spatial variance $\sigma$ against the voxel dimensions ($dx, dy, dz$) extracted directly from the NIfTI header affine matrix.

---

## Performance & Benchmarks

`rust-mri-demo` achieves real-time cohort throughput by leveraging Rayon shared-memory thread parallelism, zero-cost abstractions, and SIMD-accelerated array transformations.

### Cohort Processing Benchmark

- **Dataset Size:** 140 NIfTI volumes (3D structural & 4D DWI series)
- **Operations:** Dataset traversal, DTI tensor fitting & FA map extraction, 3D separable Gaussian smoothing, signal scaling, and compressed `.nii.gz` I/O.

| Metric | Measured Value |
| :--- | :--- |
| **Total Files Processed** | **140 files** |
| **Total Pipeline Wall Time** | **23.870 seconds** |
| **Average Latency / Volume** | **~170 ms** |
| **Throughput** | **~5.86 volumes / sec** |

*Benchmarked on local workstation hardware.*

## Features

- **Automated Dataset Traversal:** Recursively discovers NIfTI files (`.nii.gz`) across nested BIDS structures without hardcoded directory assumptions.
- **Dynamic Dimensionality Support:** Automatically detects 3D structural volumes (T1w) and 4D time-series/diffusion datasets (DWI).
- **Stage 1: Diffusion Tensor Imaging (DTI):** 
  - Fits a $3 \times 3$ symmetric Diffusion Tensor per voxel via least-squares pseudoinverse solver using a configurable $b$-value (`--bvalue`).
  - Computes Fractional Anisotropy (FA) scalar maps from tensor eigenvalues ($\lambda_1, \lambda_2, \lambda_3$).
  - Parses adjacent `.bvec` gradient files when available (falls back to synthetic gradient orientations if unmapped).
- **Stage 2: 3D Parallel Spatial Smoothing:** 
  - Applies a 3D separable Gaussian spatial filter ($3 \times 1\text{D}$ convolutions across spatial axes) parallelized via Rayon thread pooling.
  - Accepts independent smoothing kernels for 3D volumes (`--sigma-3d`) and 4D FA maps (`--sigma-fa`).
- **Stage 3: Non-Linear Perfusion Scaling:** 
  - Performs high-throughput element-wise signal transformations ($v \cdot e^{-\alpha v}$) in-place using a customizable scaling factor (`--alpha`).

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

## Usage

Run the pipeline using Cargo or the compiled binary.  
All processing parameters have defaults and can be optionally configured.

### Example (Windows)

```cmd
cargo run --release -- "D:\ds004114" --bvalue 1000 --sigma-3d 1.5 --sigma-fa 1.0 -a 0.01
```

---

## Command-Line Reference

### Plaintext Usage

```
rust-mri-demo [OPTIONS] <ROOT>
```

### Arguments

- **<ROOT>**  
  Root folder containing NIfTI dataset to process recursively

### Options

- **-a, --alpha <ALPHA>**  
  Perfusion alpha scaling parameter  
  _default: 0.01_

- **-s, --sigma-3d <SIGMA_3D>**  
  Gaussian blur sigma for 3D structural volumes  
  _default: 1.5_

- **--sigma-fa <SIGMA_FA>**  
  Gaussian blur sigma for 4D FA maps  
  _default: 1.0_

- **-b, --bvalue <BVALUE>**  
  Diffusion b-value (s/mm²) for DTI tensor fitting  
  _default: 1000_



---
