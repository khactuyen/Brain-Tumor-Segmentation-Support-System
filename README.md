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

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
SIC_Capstone 2026/
├── Datasets/                                 # Bộ dữ liệu BraTS 2023 (được ignore trên git)
│   └── brats2023-gli-dataset/
│       └── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
├── checkpoints/                              # Lưu weights mô hình huấn luyện (.pth)
│   ├── unet_best.pth
│   └── swin_unetr_best.pth
├── models/                                   # Lưu model đã export dạng production (.onnx, .pt)
│   ├── unet.onnx
│   └── unet.pt
├── web/                                      # Ứng dụng Web App Inference (FastAPI) [MỚI]
│   ├── app.py                                # API server chính (FastAPI + Swagger UI)
│   ├── inference_utils.py                    # Tiện ích suy diễn (ONNX Runtime, Preprocessing)
│   ├── requirements_web.txt                  # Thư viện riêng cho Web Server
│   └── README.md                             # Hướng dẫn chạy Web App
├── SIC_Capstone.ipynb                        # Notebook chính (EDA, Training, Evaluation, Export ONNX)
├── requirements.txt                          # Thư viện phục vụ huấn luyện (Colab / Local GPU)
├── PRD.md                                    # Tài liệu Yêu cầu Sản phẩm
└── doc.md                                    # Tài liệu Kiến trúc Hệ thống
```

---

## ⚙️ Hướng Dẫn Cài Đặt

### 1. Yêu Cầu Hệ Thống
- Hệ điều hành: Windows / Linux / macOS
- Python từ phiên bản **3.8** trở lên
- Card đồ họa hỗ trợ CUDA (khuyên dùng khi chạy notebook train)

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
   - Dành cho huấn luyện (Notebook):
     ```bash
     pip install -r requirements.txt
     ```
   - Dành cho Web Inference (FastAPI):
     ```bash
     pip install -r web/requirements_web.txt
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

---

## 🚀 Quy Trình Huấn Luyện & Triển Khai

### Bước 1: Huấn luyện & Xuất Model (Notebook)
Mở file `SIC_Capstone.ipynb` trên Google Colab hoặc môi trường Jupyter Local để thực hiện huấn luyện:
1. **EDA & Data Loading** (Mục 1-3.14): Khảo sát dữ liệu, phân tích class imbalance.
2. **Dataset & Transform** (Mục 4-5): Khởi tạo tiền xử lý ảnh và DataLoader.
3. **Build Model** (Mục 6): Lựa chọn kiến trúc `unet` hoặc `swin_unetr` từ thư viện MONAI.
4. **Huấn luyện** (Mục 7): Chạy vòng lặp training, tự động validate và lưu file checkpoint tốt nhất `.pth` vào thư mục `checkpoints/`.
5. **Đồ thị huấn luyện** (Mục 8): Trực quan hóa train loss và val dice.
6. **Export Model** (Mục 9): Tự động trace và xuất model sang 2 định dạng:
   - **TorchScript** (`models/unet.pt`)
   - **ONNX** (`models/unet.onnx`)

### Bước 2: Chạy Web Inference API (FastAPI)
Sau khi có file model trong thư mục `models/`, bạn có thể khởi chạy server dự đoán nhanh chóng trên máy local mà không cần đến PyTorch (bằng cách dùng ONNX Runtime):

1. **Cài đặt thư viện cho web:**
   ```bash
   pip install -r web/requirements_web.txt
   ```

2. **Khởi chạy Server:**
   ```bash
   # Chạy với model ONNX mặc định
   MODEL_PATH=models/unet.onnx uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Sử dụng API:**
   - Truy cập giao diện API tương tác và thử nghiệm (Swagger UI): http://localhost:8000/docs
   - Gửi yêu cầu dự đoán với 4 file ảnh MRI:
     ```bash
     curl -X POST http://localhost:8000/predict \
       -F "t1n=@path/to/case-t1n.nii.gz" \
       -F "t1c=@path/to/case-t1c.nii.gz" \
       -F "t2w=@path/to/case-t2w.nii.gz" \
       -F "t2f=@path/to/case-t2f.nii.gz"
     ```

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

---

## 👥 Thành Viên Thực Hiện (SIC Capstone 2026)
Dự án được phân công triển khai phối hợp giữa các nhóm thành viên:
* **Nhóm 1 (Data & Preprocessing):** Quản lý tiền xử lý, DataLoader, xử lý mất cân bằng và độ sạch của dữ liệu.
* **Nhóm 2 (Model Engineering):** Thiết kế kiến trúc và huấn luyện 3D U-Net & Swin UNETR trên Notebook, xuất TorchScript và ONNX.
* **Nhóm 3 (Web Inference & Deployment):** Xây dựng Sliding Window Inference bằng ONNX Runtime, phát triển web API FastAPI phục vụ tích hợp.

---
*Phiên bản dự án: 2.0.0. Tài liệu hướng dẫn phát triển và tài liệu yêu cầu chi tiết có thể tham khảo thêm tại `doc.md` và `PRD.md`.*
