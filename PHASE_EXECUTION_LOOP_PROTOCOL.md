# HƯỚNG DẪN LOOP ENGINEERING EXECUTION PROTOCOL
**Dành cho Dự án:** Brain Tumor Segmentation Support System  
**Mục đích:** Quy trình prompt tái sử dụng (Reusable Execution Loop Prompt) giúp AI tự động nhận diện, thực hiện, kiểm thử và cập nhật tiến độ theo từng Phase một cách bài bản, chặt chẽ.

---

## 1. NGUYÊN TẮC CỐT LÕI CỦA VÒNG LẶP (CORE LOOP PRINCIPLES)

Khi thực hiện bất kỳ Phase nào, AI Assistant phải tuân thủ nghiêm ngặt **Vòng lặp 5 Bước (5-Step Execution Loop)**:

```text
  ┌────────────────────────────────────────────────────────┐
  │ 1. INSPECT & PLAN                                      │
  │    - Đọc checklist Phase tương ứng từ PROJECT_PLAN...md │
  │    - Kiểm tra dependencies & files hiện có            │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. BUILD & IMPLEMENT                                   │
  │    - Viết mã nguồn hoàn chỉnh (không bỏ dở/TODO giả)    │
  │    - Đảm bảo đúng chuẩn OOP, PEP8 và modular architecture│
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. VERIFY & TEST                                       │
  │    - Kiểm tra cú pháp, import và logic                   │
  │    - Chạy thử nghiệm smoke test (nếu có thể)           │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4. UPDATE PROGRESS                                     │
  │    - Cập nhật dấu [x] vào file PROJECT_PLAN_AND_TASKS.md│
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 5. SUMMARY & HANDOVER                                  │
  │    - Tóm tắt sản phẩm đã làm                             │
  │    - Gợi ý câu lệnh / prompt cho Phase tiếp theo       │
  └────────────────────────────────────────────────────────┘
```

---

## 2. MẪU PROMPT CHUẨN (MASTER PROMPT TEMPLATES)

Bạn chỉ cần **copy một trong các mẫu prompt bên dưới** và gửi cho AI để yêu cầu thực hiện từng Phase tương ứng.

---

### 🔹 MẪU 1: Prompt thực hiện PHASE 2 (Core Models & Training)

```text
Hãy thực hiện PHASE 2: Core Models & Training theo tài liệu PROJECT_PLAN_AND_TASKS.md.

Yêu cầu chi tiết:
1. Đọc danh sách checklist Phase 2 trong file PROJECT_PLAN_AND_TASKS.md.
2. Triển khai các file mã nguồn:
   - src/models/base_model.py: Interface chuẩn BaseSegmentationModel.
   - src/models/unet3d.py: Mô hình 3D U-Net dựa trên MONAI.
   - src/models/swin_unetr.py: Mô hình Swin UNETR dựa trên MONAI (hỗ trợ pretrained weights).
   - src/models/__init__.py: Export các mô hình.
   - src/training/augmentation.py: Pipeline MONAI data transforms.
   - src/training/losses.py: DiceCELoss, FocalLoss.
   - src/training/trainer.py: Class SegmentationTrainer (AMP, gradient accumulation, model checkpointing).
   - scripts/train_unet.py & scripts/train_swin_unetr.py: Script huấn luyện độc lập.
3. Kiểm tra tính hợp lệ của mã nguồn (imports, logic, shape tensor đầu ra).
4. Cập nhật các ô [x] tương ứng trong PROJECT_PLAN_AND_TASKS.md.
5. Tóm tắt kết quả và hướng dẫn bước tiếp theo.
```

---

### 🔹 MẪU 2: Prompt thực hiện PHASE 3 (Inference Engine, Post-processing & Metrics)

```text
Hãy thực hiện PHASE 3: Inference Engine, Post-processing & Metrics theo tài liệu PROJECT_PLAN_AND_TASKS.md.

Yêu cầu chi tiết:
1. Đọc danh sách checklist Phase 3 trong file PROJECT_PLAN_AND_TASKS.md.
2. Triển khai các file mã nguồn:
   - src/inference/prediction_result.py: Dataclass PredictionResult lưu kết quả, metadata, metrics.
   - src/inference/postprocessor.py: Hậu xử lý mask (remove small components, morphological ops).
   - src/inference/engine.py: Class InferenceEngine triển khai Sliding Window Inference 3D của MONAI.
   - src/inference/__init__.py: Export module.
   - Nâng cấp src/utils/metrics.py: Bổ sung HD95 chuẩn xác và tính metrics theo từng vùng u (WT, TC, ET).
   - src/visualization/viewer.py: Vẽ lát cắt MRI theo 3 mặt phẳng (Axial, Coronal, Sagittal) với mask color overlay.
   - src/visualization/comparison.py: Biểu đồ so sánh chỉ số giữa 2 mô hình.
   - src/visualization/__init__.py: Export module.
3. Kiểm tra tính hợp lệ của mã nguồn và khớp nối với Phase 1 & 2.
4. Cập nhật các ô [x] tương ứng trong PROJECT_PLAN_AND_TASKS.md.
5. Tóm tắt kết quả triển khai.
```

---

### 🔹 MẪU 3: Prompt thực hiện PHASE 4 (Reporting, Streamlit UI & Integration)

```text
Hãy thực hiện PHASE 4: Reporting, Web UI & Integration theo tài liệu PROJECT_PLAN_AND_TASKS.md.

Yêu cầu chi tiết:
1. Đọc danh sách checklist Phase 4 trong file PROJECT_PLAN_AND_TASKS.md.
2. Triển khai các file mã nguồn:
   - src/reports/generator.py: Bộ tạo báo cáo PDF chuyên nghiệp (bằng fpdf2).
   - src/reports/html_report.py: Bộ tạo báo cáo HTML responsive.
   - src/reports/__init__.py: Export module.
   - app/streamlit_app.py: Giao diện Web UI tương tác hoàn chỉnh (Sidebar upload, chọn model, slider xem 3D slice, hiển thị metrics, thể tích khối u và nút xuất PDF).
   - README.md: Hướng dẫn cài đặt và vận hành hệ thống toàn diện.
3. Kiểm tra End-to-End luồng ứng dụng từ Upload -> Inference -> Visualizer -> Report.
4. Đánh dấu [x] tất cả các task đã hoàn thành trong PROJECT_PLAN_AND_TASKS.md.
5. Tổng kết dự án và cung cấp lệnh khởi chạy Streamlit app.
```

---

### 🔹 MẪU TỔNG QUÁT (GENERIC PROMPT TEMPLATE CHO PHASE BẤT KỲ)

```text
[PHASE EXECUTION REQUEST]
Hãy triển khai [TÊN PHASE / PHASE X] dựa theo tài liệu PROJECT_PLAN_AND_TASKS.md.

Quy trình thực hiện:
1. REVIEW: Kiểm tra các task trong checklist của Phase này.
2. BUILD: Tạo/Cập nhật đầy đủ tất cả các file code liên quan (không dùng placeholder hay mã giả).
3. VERIFY: Kiểm tra cú pháp, import tương đối và tính liên kết với các Phase trước.
4. UPDATE: Đánh dấu [x] cho các task đã hoàn thành trong PROJECT_PLAN_AND_TASKS.md.
5. REPORT: Tóm tắt danh sách file đã hoàn thành và hướng dẫn bước kế tiếp.
```

---

## 3. CHECKLIST KIỂM TRA CHẤT LƯỢNG CHO MỖI PHASE (QUALITY GATEWAYS)

Mỗi khi AI báo cáo hoàn thành 1 Phase, bạn có thể nhanh chóng kiểm tra theo 4 tiêu chí:

- [ ] **Tính đầy đủ (Completeness):** Tất cả các file trong thiết kế của Phase đó đã được tạo ra chưa? Có file nào còn chứa comment `TODO` chưa làm không?
- [ ] **Tính liên kết (Integration):** Các file mới tạo có import đúng các module từ Phase trước không? (vd: `from ..config import get_config`).
- [ ] **Tính nhất quãn (Consistency):** Tham số cấu hình có đọc từ `configs/default.yaml` hoặc `src/config.py` không?
- [ ] **Cập nhật tiến độ (Progress Tracking):** File `PROJECT_PLAN_AND_TASKS.md` đã chuyển các ô `[ ]` tương ứng thành `[x]` chưa?
