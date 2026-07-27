# Project Context

## 2026-07-27 Completed Tasks

- Updated `SIC_Capstone.ipynb` with a complete early-stage data workflow before model training.
- Added **Data Understanding** notebook cells:
  - dataset path discovery using `BRATS_DATA_ROOT`, `DATA_ROOT`, or the configured local `Datasets/...` path;
  - case manifest generation;
  - missing-file checks for `t1n`, `t1c`, `t2w`, `t2f`, and `seg`;
  - NIfTI header inspection including shape, spacing, dtype, and affine diagonal;
  - sampled descriptive statistics for MRI intensities and segmentation labels.
- Added **EDA** notebook cells:
  - dataset completeness plots;
  - segmentation availability and missing-file plots;
  - modality intensity distribution plots;
  - segmentation label voxel-count plot;
  - visual slice inspection for four MRI modalities and FLAIR + mask overlay.
- Added **Data Preprocessing** notebook cells:
  - non-zero voxel z-score normalization per MRI modality;
  - center crop/pad utilities for 3D volumes and 4-channel MRI tensors;
  - BraTS label remapping from older label `4` to class `3` when needed;
  - reusable `preprocess_case()` function returning processed image, mask, spacing, affine, original shape, and target shape;
  - before/after preprocessing visualization.
- Renumbered later notebook sections so the flow is now:
  1. install dependencies;
  2. imports/environment;
  3. dataset path config;
  4. data understanding;
  5. EDA;
  6. preprocessing;
  7. dataset loader;
  8. MONAI transforms/DataLoader;
  9-12. model training, evaluation, and visualization.

## Current Notes

- No `Datasets/` or `data/` folder was found in the local workspace during this update. The notebook now handles that case gracefully and prints instructions to set the dataset path.
- The local `.venv` Python executable is currently broken because it points to a missing base Python installation. Notebook JSON validation/editing was done with Node.js instead.
