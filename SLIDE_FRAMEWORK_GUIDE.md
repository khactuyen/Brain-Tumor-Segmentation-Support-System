# 🎓 BỘ KHUNG CHUẨN THIẾT KẾ DỰ ÁN & SLIDE CAPSTONE SAMSUNG INNOVATION CAMPUS (SIC)
**Đề tài:** Brain Tumor Segmentation Support System (BraTS 2023)  
**Chương trình:** Samsung Innovation Campus (SIC) — Trí tuệ Nhân tạo (AI)  
**Tài liệu tham chiếu chuẩn:** *Chương 10: Bắt đầu một Dự án AI (`Quy trình.pdf`)*  

---

## 🎯 TỔNG QUAN PHƯƠNG PHÁP LUẬN TỪ GIÁO TRÌNH SIC

Dự án AI Capstone chuẩn Samsung Innovation Campus được triển khai dựa trên sự kết hợp giữa **Quy trình chuẩn Khai thác Dữ liệu (CRISP-DM)** và **5 Bước Tư duy Thiết kế (Design Thinking)** tiếp cận Từ trên xuống (Top-down approach):

```text
========================================================================================
                      QUY TRÌNH KẾT HỢP SIC (DESIGN THINKING + CRISP-DM)
========================================================================================
[BÀI 1: BẮT ĐẦU VÀ LẬP KẾ HOẠCH DỰ ÁN AI]
 ├── Lựa chọn kiểu dự án: "Giải pháp" (Chưa biết cách làm / Đã biết vấn đề u não)
 ├── Đánh giá Trade-off: Giảm giá trị theo Thời gian trì hoãn chẩn đoán (Trang 6-7)
 └── Quy trình khung: Lặp lại CRISP-DM (Business -> Data -> Prep -> Model -> Eval -> Deploy)

[BÀI 2: 5 BƯỚC TƯ DUY THIẾT KẾ (DESIGN THINKING)]
 ├── Bước 1: XÁC ĐỊNH (Define)      -> Trả lời 4 câu hỏi định hướng SIC & Đánh giá Tác động
 ├── Bước 2: KHÁM PHÁ (Discover)    -> Khảo sát dữ liệu thô (Research/Observation/Interview)
 ├── Bước 3: PHÂN TÍCH (Analyze)     -> Xây dựng Đối tượng đại diện (Persona: Bác sĩ)
 ├── Bước 4: XÂY DỰNG Ý TƯỞNG & MẪU -> Ma trận 2x2, Pipeline .NPZ, 3D U-Net, Streamlit UI
 └── Bước 5: THỬ NGHIỆM (Test)      -> Đánh giá kép: Tiêu chí HÀM (Chức năng) & CẢM XÚC
========================================================================================
```

---

## 📑 CHI TIẾT NỘI DUNG 10 SLIDE THEO QUY CHUẨN SIC

### 🔹 BƯỚC 1: XÁC ĐỊNH (DEFINE)

#### **Slide 1: Trang Bìa Đề Tài (Samsung Innovation Campus)**
- **Tiêu đề**: HỆ THỐNG HỖ TRỢ PHÂN ĐOẠN KHỐI U NÃO 3D MRI (BraTS 2023)
- **Đơn vị**: Dự án AI Capstone — Samsung Innovation Campus (SIC)
- **Tóm tắt**: 3D U-Net + Fast NPZ Drive Cache + Auto PDF Medical Report Generator.

#### **Slide 2: Xác Định Vấn Đề Lâm Sàng (Vấn Đề Là Gì? Ở Đâu? Ai?)**
- **Mẫu câu hỏi định hướng SIC**:  
  `Cách để [phân đoạn khối u brains 3D MRI] [của các ca bệnh u não ác tính Glioma] [ở các bệnh viện] [để đưa ra phác đồ điều trị kịp thời nhằm ngăn chặn tử vong và nguy hiểm tính mạng] là gì?`
- **4 Thách thức y khoa có dẫn chứng**:
  1. *Tốn 30-60 phút/ca & Biến thiên 15-28% giữa các bác sĩ* (📌 Bakas et al., Nature Sci Data 2017).
  2. *Glioblastoma (GBM) tỷ lệ sống 5 năm dưới 6.8%* (📌 CBTRUS 2022).
  3. *Thiếu hụt 35% nhân lực bác sĩ thần kinh* (📌 RCR Census 2022).
  4. *Ranh giới u ác tính thâm nhiễm phức tạp* (📌 Menze et al., IEEE TMI 2015).
- ⚠️ **Quy tắc SIC**: **Tẩy chay việc đưa giải pháp AI/Model vào Slide này!** (Chỉ tập trung vào Nỗi đau lâm sàng).

#### **Slide 3: Nguyên Nhân Cốt Lõi & Đánh Giá Mức Độ Tác Động**
- **Mục tiêu sản phẩm**: Tự động phân đoạn dưới 2 phút, tính thể tích u cm³, trực quan 3D slices, xuất báo cáo PDF.
- **Đánh giá mức độ tác động tổng thể**:
  - *Mức độ tác động tổng thể* = (Thời gian tiết kiệm 43 phút/ca) x (Số lượng bệnh nhân hỗ trợ) x (Gia tăng độ chính xác phân đoạn).
  - *Tác động Xã hội*: Giảm tải áp lực y tế tuyến đầu, bình đẳng hóa y tế kỹ thuật cao.
  - *Tác động Cá nhân*: Giảm áp lực burnout cho bác sĩ (tăng hiệu suất 15-20 lần), giúp bệnh nhân nhận phác đồ sớm trong "thời gian vàng".

---

### 🔹 BƯỚC 2: KHÁM PHÁ (DISCOVER)

#### **Slide 4: Khảo Sát Bộ Dữ Liệu Ban Đầu (BraTS 2023 GLI)**
- **Nghiên cứu dữ liệu thô**: **1,251 ca bệnh 3D MRI** (Train 1,063 | Val 188).
- **Nguồn gốc Bộ Dữ Liệu (Dataset Origin & Citation)**:
  - 📌 *Nguồn gốc*: Bộ dữ liệu thuộc Cuộc thi Phân đoạn Khối u Brain Quốc tế **MICCAI BraTS 2023 Challenge**.
  - 📌 *Tổ chức & Tài trợ*: **ASNR** (Hiệp hội Chẩn đoán Hình ảnh Thần kinh Hoa Kỳ) & **NCI/NIH** (Viện Ung thư Quốc gia / Viện Y tế Quốc gia Mỹ).
  - 📌 *Dán nhãn*: Được khoanh vùng và kiểm định bởi hàng chục bác sĩ chuyên khoa thần kinh giàu kinh nghiệm từ hơn 40 trung tâm y tế lớn trên toàn cầu.
- **4 Chuỗi xung MRI**: T1n (Cấu trúc), T1c (Tiêm thuốc - ET), T2w (Dịch/tổn thương), T2f (FLAIR - Phù nề ED).

---

### 🔹 BƯỚC 3: PHÂN TÍCH (ANALYZE)

#### **Slide 5: Chân Dung Bác Sĩ (Persona) & Phân Vùng Khối U Y Khoa**
- **Đối tượng đại diện (Persona)**: Bác sĩ Chẩn đoán Hình ảnh Thần kinh (Cần công cụ chính xác, tốc độ cao, hiển thị trực quan).
- **4 Nhãn gốc Ground Truth**: Label 0 (Background), Label 1 (NCR/NET), Label 2 (ED), Label 3 (ET).
- **3 Vùng khối u phẫu thuật**: **WT** (Label 1+2+3), **TC** (Label 1+3), **ET** (Label 3).
- **Khung chốt hạ (Why)**: *Tại sao phải gộp thành 3 vùng WT, TC, ET?*

---

### 🔹 BƯỚC 4: XÂY DỰNG Ý TƯỞNG & THIẾT KẾ MẪU (DEVELOP & PROTOTYPE)

#### **Slide 6: Chuẩn Bị Dữ Liệu & Giải Pháp .NPZ Drive Cache (CRISP-DM: Data Prep)**
- **Điểm nghẽn cũ**: File `.nii.gz` thô giải nén CPU Gzip mất ~5.0s/case, tốn RAM `float64`.
- **Giải pháp tối ưu**: Pre-convert 1,251 cases sang mảng `.npz` nén `float32`/`uint8` lưu vĩnh viễn trên Google Drive (`15.63 GB`).
- **Kết quả**: Tốc độ nạp giảm xuống **~0.003s/case (Tăng tốc ~1000 lần)**.

#### **Slide 7: Mô Hình Hóa AI 3D U-Net (CRISP-DM: Modeling - Tổng Quan)**
- **Mô hình Học sâu 3D U-Net (MONAI)**: Mạng nơ-ron dạng U-Shape chuyên dụng cho ảnh 3D y tế, tự động học đặc trưng hình thái u và khôi phục độ phân giải giải phẫu gốc.
- **Suy Luận 3D Sliding Window**: Cơ chế trượt cửa sổ không gian 3D xử lý từng vùng nhỏ rồi kết hợp mượt mà, giúp dự đoán ảnh 3D kích thước lớn mà không tràn bộ nhớ GPU.
- **Hậu xử lý Khử Nhiễu**: Lọc nhiễu tín hiệu giả và làm mịn ranh giới phân đoạn bằng thuật toán hình thái học và phân tích vùng liên thông.
- **Khung chốt hạ (Why)**: *Tại sao chọn 3D U-Net & Sliding Window?* (Học không gian 3D mượt mà và duy trì độ phân giải cao mà vẫn tối ưu bộ nhớ phần cứng).

#### **Slide 8: Thiết Kế Mẫu Prototype (Web Streamlit & PDF Medical Report)**
- **Prototype Giao diện Người dùng (UI)**: Ứng dụng Web App Streamlit (`app/streamlit_app.py`) hỗ trợ kéo Slider xem 3D Slices theo 3 trục Axial/Coronal/Sagittal và hiển thị bảng chỉ số.
- **Prototype Dịch vụ Báo cáo (Service)**: Module xuất Báo cáo Chẩn đoán y khoa PDF (`src/reports/generator.py` bằng `fpdf2`) phục vụ lưu trữ hồ sơ bệnh án.

---

### 🔹 BƯỚC 5: THỬ NGHIỆM & TRIỂN KHAI (TEST & DEPLOYMENT)

#### **Slide 9: Thử Nghiệm & Đánh Giá Tiêu Chí Kép (Hàm & Cảm Xúc)**
- **Khung Đánh Giá Tiêu Chí Kép Theo Giáo Trình SIC**:
  - ⚙️ **Tiêu chí HÀM (Chức năng / Function)**:
    - Tốc độ nạp dữ liệu dưới 0.01s/case.
    - Thời gian suy luận mô hình 3D U-Net dưới 2 phút/case.
    - Validation Dice Score trên 0.80 trên các vùng u (WT, TC, ET).
    - Tính thể tích khối u thực tế (cm³) chính xác.
  - ❤️ **Tiêu chí CẢM XÚC & TRẢI NGHIỆM (Emotions)**:
    - Mức độ hài lòng của bác sĩ khi thao tác đơn giản, loại bỏ thao tác thủ công phức tạp.
    - Giảm hẳn tâm lý căng thẳng, lo lắng sai sót trong chẩn đoán.

#### **Slide 10: Triển Khai & Tổng Kết Nghiệm Thu (CRISP-DM: Deployment)**
- **Triển khai (Deployment)**: Tóm tắt kết quả nghiệm thu toàn bộ hệ thống từ Nạp dữ liệu `.npz` trên Drive -> 3D U-Net Inference -> Streamlit Visualizer -> PDF Report Generator.
- **Lời cảm ơn**: Cảm ơn Ban tổ chức **Samsung Innovation Campus (SIC)** và Bác sĩ cố vấn chuyên môn.

---

## 🛠️ QUY CHUẨN VISUAL SLIDE THEO PHƯƠNG PHÁP SIC

1. **Thẻ phân loại SIC Badge**: Mọi Slide đều gắn tên bước tương ứng (`Bước 1: DEFINE`, `Bước 2: DISCOVER`, `Bước 3: ANALYZE`, `Bước 4: DEVELOP/MODELING`, `Bước 5: TEST`).
2. **Khung Chốt Hạ Lý Do (Why Callouts)**: Mọi slide từ Slide 2-7 đều giữ khung chốt hạ màu Cyan/Green trả lời câu hỏi cốt lõi "Tại sao chọn giải pháp này?".
3. **Màu sắc & Typography**: Dark Mode Y Khoa Công Nghệ Cao (Deep Slate `#0b0f19`, Cyan `#06b6d4`, Green `#10b981`, Google Fonts `Plus Jakarta Sans`).
