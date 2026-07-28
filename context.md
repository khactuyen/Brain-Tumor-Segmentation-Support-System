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

## 2026-07-28 Các Nhiệm vụ Đã Hoàn thành

- Tiếp tục nhiệm vụ notebook còn dang dở (phiên trước bị dừng giữa lúc chèn phần EDA).
- **Phần EDA** (`SIC_Capstone.ipynb`):
  - Thêm tiêu đề markdown `## 3.6 EDA`.
  - Thêm ô **EDA 1**: trực quan hóa 4 modalities trên cùng một lát cắt axial + overlay mask phân đoạn (tự động chọn lát cắt có nhiều u nhất). Ô EDA 2 (histogram cường độ + thống kê lớp u + box plot thể tích) đã có sẵn từ trước.
- **Kiểm tra DataLoader (Sanity-Check)** (`SIC_Capstone.ipynb`):
  - Thêm tiêu đề `### 5.1 Sanity Check` + ô code ngay sau ô tạo DataLoader: kiểm tra (assert) shape của batch/patch, dải giá trị nhãn `{0,1,2,3}`, in ra dải cường độ, và trực quan hóa một augmented patch + mask.
  - Sửa lỗi vị trí: `cell id` của NotebookEdit là **chỉ số vị trí (positional index)** chứ không phải id cố định, nên phải chèn ô sanity-check SAU ô tạo `train_loader`, không phải sau ô định nghĩa class.
- Luồng notebook cuối cùng: Data Understanding (3.5) → EDA (3.6: EDA1 + EDA2) → Dataset Loader (4) → Augmentation/DataLoader (5) → Sanity Check (5.1) → Huấn luyện/Đánh giá/Trực quan hóa (6-9). JSON đã kiểm tra hợp lệ (25 ô).

## Current Notes

- No `Datasets/` or `data/` folder was found in the local workspace during this update. The notebook now handles that case gracefully and prints instructions to set the dataset path.
- The local `.venv` Python executable is currently broken because it points to a missing base Python installation. Notebook JSON validation/editing was done with Node.js instead.
