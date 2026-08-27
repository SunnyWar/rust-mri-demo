# Rust Neuroimaging Pipeline (`rust-mri-demo`)

A high-throughput, multi-threaded command-line engine written in Rust for processing 3D structural and 4D Diffusion Tensor Imaging (DTI) datasets in BIDS format (`.nii.gz`).

Built as a high-performance compute backend for neuroimaging research, this pipeline leverages shared-memory thread parallelism (`rayon`), SIMD-accelerated array transformations (`ndarray`), and rigid linear algebra solvers (`nalgebra`).

---

## Key Algorithmic Stages

1. **BIDS Dataset Traversal:** Recursively resolves NIfTI structures across arbitrary subject/session directories without hardcoded paths.
2. **Otsu Brain Masking:** Performs global 256-bin Otsu thresholding to segment intracranial volume from background noise and skull space.
3. **WLLS DTI Tensor Fitting:** 
   - Fits a $3 \times 3$ symmetric diffusion tensor per voxel using Weighted Linear Least Squares (WLLS) with $W = \text{diag}(S^2)$ variance weighting.
   - Computes Fractional Anisotropy (FA) scalar maps derived from tensor eigenvalues ($\lambda_1, \lambda_2, \lambda_3$).
   - Automatically parses associated `.bvec` direction matrices.
4. **Physical Anisotropic Gaussian Smoothing:**
   - Reads spatial voxel dimensions ($dx, dy, dz$ in mm) from NIfTI header `pixdim` affine metadata.
   - Computes separable 1D Gaussian kernel sigmas in physical space ($\sigma_{\text{vox}} = \sigma_{\text{mm}} / \text{zoom}$) to prevent spatial blur distortions across anisotropic slice geometries.

---

## Performance & Cohort Benchmarks

`rust-mri-demo` achieves real-time cohort throughput by avoiding redundant heap allocations, utilizing memory-mapped stream handles, and leveraging parallel Rayon thread pooling across volume batches.

### Cohort Benchmark Results

| Metric | Measurement |
| :--- | :--- |
| **Dataset Size** | 140 NIfTI volumes (3D T1w / 4D DWI) |
| **Total Wall Clock Time** | **11.333 seconds** |
| **Average Per-Volume Latency** | **~81 ms / volume** |
| **Throughput** | **~12.35 volumes / sec** |

*Benchmarked across 140 subject files including I/O, Otsu masking, WLLS tensor fitting, anisotropic smoothing, and Gzip compression.*

---

## Scientific Assumptions & Pipeline Positioning

> **Note on Pipeline Context:** `rust-mri-demo` is designed as a high-throughput compute engine for **BIDS derivatives**. It assumes input NIfTI volumes have undergone baseline motion and eddy current correction via upstream pipelines (e.g., QSIPrep, fMRIPrep).

### Microstructural Model
Monocompartment tensor fit using weighted linear least-squares (WLLS) on log-attenuated signals:

$$\mathbf{S}(b) = S_0 \cdot e^{-b \cdot \mathbf{g}^T \mathbf{D} \mathbf{g}}$$

Fractional Anisotropy (FA) is derived directly from the tensor eigenvalues:

$$\text{FA} = \sqrt{\frac{3}{2}} \frac{\sqrt{(\lambda_1 - \bar{\lambda})^2 + (\lambda_2 - \bar{\lambda})^2 + (\lambda_3 - \bar{\lambda})^2}}{\sqrt{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}}$$

---

## Build & Usage

### Prerequisites
- [Rust Toolchain](https://www.rust-lang.org/tools/install) (1.75+)

## Compile

```cmd
cargo build --release
```

## Execution Example

```cmd
cargo run --release -- "D:\ds004114" --bvalue 1000 --sigma-3d 1.5 --sigma-fa 1.0
```

## Command-Line Reference

rust-mri-demo [OPTIONS] <ROOT>

### Arguments

- <ROOT> — Root folder containing BIDS NIfTI dataset to process recursively.

### Options

- -s, --sigma-3d <SIGMA_3D>
  Smoothing kernel σ (mm) for 3D structural volumes
  default: 1.5

- --sigma-fa <SIGMA_FA>
  Smoothing kernel σ (mm) for 4D FA maps
  default: 1.0

- -b, --bvalue <BVALUE>
  Diffusion b-value (s/mm²) for DTI tensor fitting
  default: 1000
