# Brain Tumor Segmentation Support System (BraTS 2023)

Hệ thống hỗ trợ bác sĩ và nhà nghiên cứu tự động phân đoạn khối u não từ ảnh chụp MRI 3D đa chuỗi (Multi-modal MRI) định dạng NIfTI. Dự án triển khai và so sánh hai kiến trúc học sâu tiên tiến **3D U-Net** và **Swin UNETR** bằng thư viện **MONAI** và **PyTorch**, tích hợp công cụ trực quan hóa 3 chiều, tính toán thể tích khối u, chỉ số đánh giá y khoa chi tiết và xuất báo cáo tự động.

---

## 🌟 Tính Năng Nổi Bật

1. **Pipeline Dữ Liệu Chuyên Nghiệp:** Đọc và ghi file NIfTI (.nii/.nii.gz), tiền xử lý đồng nhất (Z-score normalization, Center Crop/Pad về shape mục tiêu).
2. **Mô Hình Phân Đoạn Tiên Tiến:** 
   - **3D U-Net:** Phân đoạn hiệu quả dựa trên cấu trúc không gian cục bộ.
   - **Swin UNETR:** Ứng dụng Transformer nắm bắt ngữ cảnh toàn cục của các khối u phức tạp.
3. **Inference Tránh Tràn Bộ Nhớ:** Sử dụng cơ chế **Sliding Window Inference** để xử lý các khối ảnh MRI 3D kích thước lớn mà không gây quá tải VRAM.
4. **Hậu Xử Lý Thông Minh:** Khử nhiễu mask phân đoạn bằng các thuật toán hình thái học (Morphological Operations) và lọc thành phần liên thông lớn nhất (Connected Components).
5. **Chỉ Số Đánh Giá Y Khoa Chuyên Sâu:** Tính toán các chỉ số Dice, IoU, Precision, Recall, và khoảng cách Hausdorff 95 (HD95) cho từng phân vùng u cụ thể theo tiêu chuẩn BraTS:
   - **Whole Tumor (WT):** Toàn bộ vùng u (nhãn 1 + 2 + 3)
   - **Tumor Core (TC):** Lõi u (nhãn 1 + 3)
   - **Enhancing Tumor (ET):** Vùng u bắt thuốc (nhãn 3)
6. **Ước Tính Thể Tích Khối U:** Tự động tính toán thể tích u não ra đơn vị xăng-ti-mét khối (cm³) dựa trên kích thước voxel của ảnh gốc.
7. **Trực Quan Hóa Tương Tác:** Hiển thị lát cắt trên cả 3 trục (Axial, Coronal, Sagittal) chồng lớp mặt nạ phân đoạn (Overlay) nhiều màu sắc.
8. **Xuất Báo Cáo Tự Động:** Hỗ trợ tạo và tải báo cáo kết quả định dạng PDF/HTML.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
SIC_Capstone 2026/
├── configs/                                  # Cấu hình hệ thống
│   └── default.yaml                          # File yaml chứa thông số mô hình, đường dẫn, hyperparams
├── Datasets/                                 # Bộ dữ liệu BraTS 2023 (được ignore trên git)
│   └── brats2023-gli-dataset/
│       └── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
├── checkpoints/                              # Lưu weights mô hình huấn luyện (.pth)
│   ├── unet_brats_best.pth
│   └── swin_unetr_brats_best.pth
├── results/                                  # Đầu ra của quá trình suy luận
│   ├── masks/                                # Mask dự đoán định dạng .nii.gz
│   ├── overlays/                             # Trực quan slice PNG
│   └── reports/                              # Báo cáo PDF/HTML
├── scripts/                                  # Các python scripts độc lập
│   ├── train_unet.py                         # Huấn luyện 3D U-Net
│   └── train_swin_unetr.py                   # Huấn luyện Swin UNETR
├── app/                                      # Ứng dụng giao diện người dùng (Streamlit)
│   ├── streamlit_app.py                      # File chạy Streamlit main app
│   └── components/                           # Các component UI tương tác
├── src/                                      # Mã nguồn cốt lõi (Core Engine)
│   ├── config.py                             # Quản lý cấu hình dự án
│   ├── data/                                 # Xử lý dữ liệu (DataLoader, Preprocessor, Dataset)
│   ├── models/                               # Kiến trúc mô hình AI (Base, U-Net, Swin UNETR)
│   ├── training/                             # Bộ huấn luyện (Trainer, Loss, Augmentation)
│   ├── inference/                            # Bộ suy luận (Inference Engine, Postprocessor)
│   ├── visualization/                        # Trực quan hóa (Viewer, Comparison charts)
│   └── utils/                                # Tiện ích (Metrics, Logger, I/O)
├── SIC_Capstone.ipynb                        # Jupyter Notebook phục vụ R&D, EDA và Demo nhanh
├── requirements.txt                          # Khai báo các thư viện phụ thuộc
├── PRD.md                                    # Tài liệu Yêu cầu Sản phẩm
└── doc.md                                    # Tài liệu Kiến trúc Hệ thống
```

---

## ⚙️ Hướng Dẫn Cài Đặt

### 1. Yêu Cầu Hệ Thống
- Hệ điều hành: Windows / Linux / macOS
- Python từ phiên bản **3.8** trở lên
- Card đồ họa hỗ trợ CUDA (khuyên dùng để huấn luyện và suy luận nhanh)

### 2. Các Bước Cài Đặt

1. **Clone repository:**
   ```bash
   git clone <repository_url>
   cd "SIC_Capstone 2026"
   ```

2. **Khởi tạo môi trường ảo Python (Virtual Environment):**
   ```bash
   python -m venv .venv
   ```

3. **Kích hoạt môi trường ảo:**
   * **Trên Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **Trên Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```

4. **Cài đặt các thư viện phụ thuộc:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📊 Chuẩn Bị Dữ Liệu

Đặt bộ dữ liệu BraTS 2023 GLI theo cấu trúc như sau:
```text
Datasets/
└── brats2023-gli-dataset/
    └── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
        ├── BraTS-GLI-00000-000/
        │   ├── BraTS-GLI-00000-000-t1n.nii.gz    # Ảnh MRI T1-weighted dọc
        │   ├── BraTS-GLI-00000-000-t1c.nii.gz    # Ảnh MRI T1-weighted cản quang
        │   ├── BraTS-GLI-00000-000-t2w.nii.gz    # Ảnh MRI T2-weighted
        │   ├── BraTS-GLI-00000-000-t2f.nii.gz    # Ảnh MRI FLAIR
        │   └── BraTS-GLI-00000-000-seg.nii.gz    # Mask phân đoạn chuẩn (Ground Truth)
        └── ...
```
*Lưu ý: Bạn có thể cập nhật đường dẫn dữ liệu trong file cấu hình `configs/default.yaml`.*

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Phân Tích Dữ Liệu (EDA) & Tiền Xử Lý trên Jupyter Notebook
Khởi chạy Jupyter Notebook hoặc VS Code để mở file `SIC_Capstone.ipynb`. Notebook này chứa luồng nghiên cứu hoàn chỉnh từ:
- Cấu hình & Đọc dữ liệu NIfTI
- Thống kê & Trực quan hóa EDA (Modalities, Mask Overlay, Intensity distribution, Volume)
- Tiền xử lý dữ liệu và tạo PyTorch DataLoader
- Chạy thử nghiệm huấn luyện/đánh giá mô hình.

### 2. Huấn Luyện Mô Hình
Để chạy huấn luyện các mô hình độc lập qua dòng lệnh (CLI):
* **Huấn luyện mô hình 3D U-Net:**
  ```bash
  python scripts/train_unet.py
  ```
* **Huấn luyện mô hình Swin UNETR:**
  ```bash
  python scripts/train_swin_unetr.py
  ```
Các tham số huấn luyện (Learning rate, Batch size, Epochs,...) đều được định nghĩa tập trung trong `configs/default.yaml`. Checkpoint tốt nhất sẽ tự động lưu vào thư mục `checkpoints/`.

### 3. Khởi Chạy Giao Diện Web (Streamlit UI)
Sau khi đã có checkpoint mô hình trong thư mục `checkpoints/`, bạn có thể khởi chạy ứng dụng web Streamlit bằng câu lệnh:
```bash
streamlit run app/streamlit_app.py
```
Giao diện Streamlit cho phép bạn:
1. Upload các file ảnh MRI của một ca bệnh.
2. Chọn mô hình suy luận (U-Net, Swin UNETR hoặc cả hai để so sánh).
3. Xem lát cắt MRI kèm theo mặt nạ khối u bằng thanh kéo trực quan.
4. Đánh giá độ chính xác (so với Ground Truth nếu có) và ước lượng thể tích u.
5. Xuất báo cáo PDF/HTML của ca bệnh.

---

## 🩺 Quy Ước Nhãn & Chỉ Số Đánh Giá

Bộ dữ liệu BraTS quy định các nhãn trong file phân đoạn gốc (`-seg.nii.gz`) như sau:
* **Nhãn 0:** Vùng mô khỏe hoặc nền (Background)
* **Nhãn 1:** Vùng hoại tử và phần lõi u không bắt thuốc (Necrotic Tumor Core - NCR)
* **Nhãn 2:** Vùng phù nề quanh u (Peritumoral Edematous Stroma - ED)
* **Nhãn 3:** Vùng u bắt thuốc hoạt động (Enhancing Tumor - ET)

Để phục vụ đánh giá lâm sàng, các nhãn được gom nhóm thành 3 phân vùng chính:
* **Whole Tumor (WT - Toàn bộ u):** Phối hợp Nhãn 1, 2 và 3.
* **Tumor Core (TC - Lõi u):** Phối hợp Nhãn 1 và 3.
* **Enhancing Tumor (ET - U hoạt động):** Nhãn 3.

Hệ thống tự động chuyển đổi định dạng và đánh giá mô hình dựa trên các chỉ số Dice, IoU, Precision, Recall, HD95 trên 3 vùng này.

---

## 👥 Thành Viên Thực Hiện (SIC Capstone 2026)
Dự án được phân công triển khai phối hợp giữa các nhóm thành viên:
* **Nhóm 1 (Data Pipeline):** Quản lý cấu hình, tiền xử lý, DataLoader và tính toàn vẹn dữ liệu.
* **Nhóm 2 (Model Engineering):** Thiết kế kiến trúc và huấn luyện 3D U-Net & Swin UNETR.
* **Nhóm 3 (Inference Engine):** Xây dựng Sliding Window Inference, hậu xử lý khử nhiễu và lập công thức tính y khoa.
* **Nhóm 4 (UI & Visualization):** Phát triển ứng dụng Streamlit, vẽ đồ thị so sánh và sinh báo cáo tự động.

---
*Phiên bản dự án: 1.0.0. Tài liệu hướng dẫn phát triển và tài liệu yêu cầu chi tiết có thể tham khảo thêm tại `doc.md` và `PRD.md`.*
