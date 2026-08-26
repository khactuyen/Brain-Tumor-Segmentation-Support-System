# Product Requirements Document (PRD)

## 1. Tên sản phẩm
**Brain Tumor Segmentation Support System (BraTS 2023)**

---

## 2. Mục tiêu sản phẩm & Phương pháp Tư duy Thiết kế (Design Thinking)

### 2.1 Mục tiêu sản phẩm
- **Phân đoạn tự động khối u não**: Hỗ trợ bác sĩ và nhà nghiên cứu phân đoạn tự động các vùng khối u (WT - Whole Tumor, TC - Tumor Core, ET - Enhancing Tumor) từ 4 chuỗi ảnh MRI (T1, T1ce, T2, FLAIR).
- **Trực quan hóa & Thống kê Y khoa**: Hiển thị lát cắt 3 trục (Axial, Coronal, Sagittal) với mask màu đa lớp (color overlay), tính toán thể tích khối u (cm³) và các chỉ số chẩn đoán (Dice, IoU, Precision, Recall, HD95).
- **Báo cáo Chẩn đoán Tự động**: Xuất báo cáo y khoa chuyên nghiệp định dạng **PDF** và **HTML** cho từng trường hợp chẩn đoán.

---

### 2.2 Năm bước tư duy thiết kế — Bước 1 & 2: Xác định Vấn đề với Minh chứng Dữ liệu Thực tế (Problem Definition & Real-world Evidence)

#### 1. Vấn đề cụ thể cần giải quyết là gì? (Dẫn chứng Thực tế & Trích nguồn)
- **Tốn thời gian & Biến thiên giữa các bác sĩ**: Theo nghiên cứu chuẩn benchmark BraTS của *Menze et al. (IEEE TMI, 2015)* và *Bakas et al. (Nature Scientific Data, 2017)*, việc khoanh vùng thủ công 3D trên 4 chuỗi xung MRI (T1, T1ce, T2, FLAIR) tốn từ **30 – 60 phút mỗi ca bệnh**, với độ biến thiên kết quả giữa các bác sĩ chẩn đoán hình ảnh (inter-observer variability) dao động tới **15% – 28%** tùy thuộc vào kinh nghiệm và sự mệt mỏi.
- **Mức độ nguy hiểm cao của Glioma**: Theo *Báo cáo Thống kê CBTRUS (Neuro-Oncology, 2022)*, U nguyên bào đệm (Glioblastoma - GBM, Grade IV Glioma) chiếm **49.1% các ca u ác tính nguyên phát**, với tỷ lệ sống sót sau 5 năm cực kỳ thấp (**dưới 6.8%**). Việc chẩn đoán và xác định thể tích u chính xác có ý nghĩa sinh tử đối với bệnh nhân.

#### 2. Tại sao bạn gặp phải vấn đề này? Và tại sao bạn muốn giải quyết vấn đề đó?
- **Quá tải và thiếu hụt nhân lực y tế**: Theo *Báo cáo Khảo sát Nhân lực Chẩn đoán Hình ảnh Lâm sàng RCR (2022)*, khối lượng ảnh chụp MRI tăng trung bình 8% – 12%/năm, trong khi số lượng bác sĩ chuyên khoa chẩn đoán hình ảnh thần kinh chỉ tăng dưới 2%/năm. Tỷ lệ thiếu hụt nhân lực chuyên khoa đạt mức **29% - 35%**, dẫn đến tình trạng tồn đọng kết quả chẩn đoán.
- **Lý do giải quyết**: Ứng dụng mô hình **3D U-Net** tự động hóa quy trình phân đoạn giúp rút ngắn thời gian chẩn đoán từ hàng chục phút xuống **dưới 2 phút**, hỗ trợ bác sĩ đưa ra phác đồ phẫu thuật/xạ trị chính xác trong "thời gian vàng".

#### 3. Bạn nghĩ đâu là nguyên nhân cốt lõi của vấn đề này?
- **Ranh giới vi thể Glioma thâm nhiễm phức tạp**: Ranh giới giữa mô hoại tử (NCR), mô phù nề (ED) và mô u tăng cường (ET) liên kết thâm nhiễm phức tạp vào mô não lành.
- **Hạn chế của mô hình 2D & Phương pháp thủ công**: Các công cụ 2D truyền thống không học được tính liên tục 3D theo trục Z-axis ($240 \times 240 \times 155$ voxels), gây sai sót khi khoanh vùng.
- **Nghẽn đĩa dữ liệu 3D**: Việc đọc và giải nén các file NIfTI thô (`.nii.gz`) tốn quá nhiều tài nguyên CPU/RAM, chưa có pipeline nạp siêu tốc.

#### 4. Giải quyết vấn đề này sẽ giúp ích thế nào cho thế giới? Đánh giá mức độ tác động

##### 📈 Đánh giá mức độ tác động tổng thể:
Mức độ tác động tổng thể được xác định dựa trên:
- **Thời gian tiết kiệm được:** Rút ngắn từ 45 phút xuống dưới 2 phút mỗi ca bệnh (tiết kiệm 43 phút/ca).
- **Quy mô số lượng bệnh nhân được hỗ trợ:** Phục vụ hàng ngàn ca chẩn đoán u não mỗi năm tại các cơ sở y tế.
- **Mức độ nâng cao độ chính xác:** Đạt chỉ số Dice trên 0.80, loại bỏ sự không đồng nhất giữa các lần chẩn đoán.

##### 🌍 Tác động đến Xã hội (Social Impact):
- **Giảm tải áp lực hệ thống y tế**: Tiết kiệm hàng ngàn giờ làm việc mỗi năm cho đội ngũ bác sĩ chẩn đoán hình ảnh tại các bệnh viện tuyến đầu.
- **Bình đẳng hóa dịch vụ y tế kỹ thuật cao**: Cung cấp công cụ AI hỗ trợ phân đoạn u não chính xác cho các bệnh viện tuyến dưới/vùng sâu vùng xa nơi thiếu vắng chuyên gia thần kinh hàng đầu.

##### 👤 Tác động đến Cá nhân (Individual Impact):
- **Đối với Bác sĩ**: Giảm căng thẳng công việc (burnout), nâng cao hiệu suất làm việc gấp 15 – 20 lần, cung cấp thêm chỉ báo thể tích khối u chính xác (cm³) để lập kế hoạch phẫu thuật.
- **Đối với Bệnh nhân**: Rút ngắn thời gian chờ nhận kết quả chẩn đoán, tăng cơ hội điều trị trong giai đoạn sớm, nâng cao tỷ lệ sống sót và chất lượng sống sau điều trị.

---

## 3. Trích nguồn Tài liệu Tham khảo Y khoa (Academic Citations)

1. **Bakas et al. (2017)**: *Advancing The Cancer Genome Atlas glioma MRI collections with consensus segmentations and white matter tractography*. **Nature Scientific Data**, 4:170117.
2. **Menze et al. (2015)**: *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)*. **IEEE Transactions on Medical Imaging (TMI)**, 34(10):1993-2024.
3. **Ostrom et al. (2022)**: *CBTRUS Statistical Report: Primary Brain and Other Central Nervous System Tumors Diagnosed in the United States in 2015–2019*. **Neuro-Oncology**, 24(Suppl 5):v1-v95.
4. **Royal College of Radiologists (RCR, 2022)**: *Clinical Radiology Workforce Census 2022 Report*.

---

## 4. Người dùng mục tiêu
- **Bác sĩ chuyên khoa chẩn đoán hình ảnh**: Cần công cụ hỗ trợ khoanh vùng u dựa trên mô hình 3D U-Net và tính thể tích tự động.
- **Nghiên cứu sinh & Bác sĩ nghiên cứu y khoa**: Phân tích hiệu năng phân đoạn khối u phát triển từ ảnh MRI.
- **Sinh viên & Đội ngũ phát triển capstone**: Sử dụng làm hệ thống thử nghiệm và demo sản phẩm thực tế.

---

## 5. Đặc tả kĩ thuật & Kiến trúc Pipeline Dữ liệu Hiện tại

### 5.1 Bộ dữ liệu BraTS 2023 GLI (1,251 Cases)
- **Tổng số lượng**: 1,251 cases ảnh MRI 3D.
- **Phân chia Dataset**: 
  - Tập Huấn luyện (**Train**): 1,063 cases (~85%).
  - Tập Kiểm thử nhanh (**Val**): 188 cases (~15%).
- **Cấu trúc Modalities**: T1n, T1c, T2w, T2f (Shape: 240 x 240 x 155).

### 5.2 Tối ưu hóa Nạp dữ liệu (Ultra-fast NPZ Cache Strategy)
- **Vấn đề đã khắc phục**: Giải nén Gzip trên `.nii.gz` đơn luồng mặc định khiến nạp dữ liệu bị nghẽn đĩa và tốn ~5.0 giây/case.
- **Giải pháp triển khai**:
  - Chuyển đổi toàn bộ 1,251 cases sang mảng Numpy nén `.npz` (`float32` cho ảnh, `uint8` cho mask).
  - Lưu trữ vĩnh viễn tại đường dẫn Google Drive: `/content/drive/MyDrive/BraTS2023/processed_npz`.
  - Tích hợp cơ chế kiểm tra tự động (`preprocess_all_cases`): Bỏ qua bước giải nén nếu dữ liệu `.npz` đã sẵn sàng.
  - Bộ nạp `BraTSDataLoader` hỗ trợ nạp `.npz` tức thì (~0.003s/case) hoặc giải nén song song đa luồng CPU (ThreadPoolExecutor) cho file NIfTI thô.

---

## 6. Tính năng chính (Features & MVP)

1. **Post-processing Filter**:
   - Khử nhiễu hình thái học (Morphological Operations) và lọc thành phần liên thông nhỏ (Connected Component Analysis).
2. **3D Multi-planar Visualizer**:
   - Cho phép kéo thanh trượt Slider xem lát cắt theo 3 mặt phẳng (Axial, Coronal, Sagittal) với mask color overlay rõ nét.
3. **Medical Metrics & Volume Calculation**:
   - Tính chỉ số Dice, IoU, Precision, Recall, HD95 theo từng phân vùng u (WT, TC, ET).
   - Quy đổi số lượng voxel sang thể tích thực tế (cm³).
4. **Automatic Medical Report Generator**:
   - Xuất file báo cáo y khoa dạng **PDF** (thông qua `fpdf2`) và **HTML** responsive.
5. **Interactive Web App (Streamlit)**:
   - Giao diện người dùng thân thiện: Upload file -> Chạy suy luận 3D U-Net -> Xem 3D Slices -> Đánh giá chỉ số -> Xuất báo cáo.

---

## 7. Yêu cầu phi chức năng (Non-Functional Requirements)

- **Tốc độ nạp dữ liệu**: < 0.01 giây/case khi đọc từ `.npz` trên Google Drive / Local SSD.
- **Thời gian suy luận**: < 2 phút / case trên GPU (Colab T4 / RTX GPU).
- **Bộ nhớ RAM/VRAM**: Đảm bảo không quá 8GB VRAM nhờ kỹ thuật Sliding Window Inference.
- **Bảo mật & Luồng làm việc**: Dữ liệu nén `.npz` lưu vĩnh viễn trên Drive cá nhân của người dùng, đảm bảo tính riêng tư.

---

## 8. Cấu trúc Thư mục Dự án Triển khai (Exact Directory Tree)

```text
SIC_Capstone 2026/
├── .gitignore                                # Git ignore configuration
├── PRD.md                                    # Product Requirements Document (v2.4 - No Formulas)
├── doc.md                                    # Architecture & Flow Documentation
├── PROJECT_PLAN_AND_TASKS.md                 # Project Plan & Tasks Checklist (v2.1)
├── PHASE_EXECUTION_LOOP_PROTOCOL.md          # Execution Loop Protocol
├── README.md                                 # System Operations Guide (Pending update)
├── requirements.txt                          # Requirements & Dependencies
├── convert_dataset.ipynb                     # Colab Notebook for Fast NPZ dataset conversion
├── SIC_Capstone.ipynb                        # Legacy Capstone Notebook
├── SIC_Capstone_v2.ipynb                     # Main 3D U-Net Capstone Pipeline Notebook
├── presentation_slides.html                  # Interactive Presentation Slide Deck
├── SLIDE_FRAMEWORK_GUIDE.md                  # Master Framework Guide for SIC Slides
│
├── configs/                                  # System Configurations
│   └── default.yaml                          # Hyperparameters & Paths config
│
├── scripts/                                  # Executable Scripts
│   ├── convert_dataset_to_npz.py             # Multi-processing NPZ Converter script
│   ├── download_drive_data.py                # Google Drive Sync / Downloader script
│   ├── train_unet.py                         # Independent 3D U-Net Training script
│   ├── fix_label_dim.py                      # Utility script for label dimensions
│   ├── fix_logger.py                         # Utility script for loggers
│   ├── replace_logger.py                     # Utility script for logger replacement
│   ├── translate_comments_v2.py              # Comment translation utility
│   └── update_notebook_pipeline.py           # Notebook pipeline updater script
│
├── src/                                      # Core Engine Source Code
│   ├── __init__.py                           # Package initialization
│   ├── config.py                             # Configuration Manager & Drive Scanner
│   │
│   ├── data/                                 # Data Engine Module
│   │   ├── __init__.py                       #
│   │   ├── data_loader.py                    # Fast NPZ Reader & Multi-threaded Fallback
│   │   ├── preprocessor.py                   # MRI Intensity Normalizer & Spatial Crop/Pad
│   │   └── dataset.py                        # PyTorch BraTSDataset & DataSplitter
│   │
│   ├── models/                               # AI Model Architectures
│   │   ├── __init__.py                       #
│   │   ├── base_model.py                     # BaseSegmentationModel Interface
│   │   └── unet3d.py                         # MONAI 3D U-Net Core Wrapper
│   │
│   ├── training/                             # Training Module
│   │   ├── __init__.py                       #
│   │   ├── trainer.py                        # SegmentationTrainer (AMP, Loss, Checkpoint)
│   │   ├── losses.py                         # DiceCELoss, FocalLoss
│   │   └── augmentation.py                   # MONAI Data Augmentation Pipeline
│   │
│   ├── inference/                            # Inference Engine
│   │   ├── __init__.py                       #
│   │   ├── engine.py                         # 3D Sliding Window Inference Engine
│   │   ├── postprocessor.py                  # Morphological Ops & CCA PostProcessor
│   │   └── prediction_result.py              # PredictionResult Dataclass
│   │
│   ├── visualization/                        # Visualization Engine
│   │   ├── __init__.py                       #
│   │   ├── viewer.py                         # 3-Plane Slice Visualizer & Color Overlay
│   │   └── comparison.py                     # Model Metrics Visualizer
│   │
│   └── utils/                                # Utilities
│       ├── __init__.py                       #
│       ├── io.py                             # NIfTI & YAML File IO
│       ├── logger.py                         # Console & File Logger
│       └── metrics.py                        # Dice, IoU, Precision, Recall, HD95, Volume
│
├── web/                                      # Web API Engine (FastAPI Backend Draft)
│   ├── README.md                             # Web API documentation
│   ├── app.py                                # FastAPI web server application
│   ├── inference_utils.py                    # Web inference utility functions
│   └── requirements_web.txt                  # Web dependencies
│
└── app/                                      # Streamlit Web UI (Phase 5 - To be created)
    ├── __init__.py                           #
    └── streamlit_app.py                      # Main Streamlit UI App
```

---

## 9. Tiêu chuẩn Hoàn thành (Definition of Done - DoD)

1. **Data Pipeline**: Pre-convert 100% dữ liệu sang `.npz` trên Drive thành công, thời gian nạp < 0.01s.
2. **Model Accuracy**: Validation Dice Score 3D U-Net $\ge 0.80$ trên các vùng khối u.
3. **Web UI & Reporting**: Ứng dụng Streamlit chạy trôi chảy từ Upload -> 3D U-Net Inference -> Visualizer -> PDF Report Export.
