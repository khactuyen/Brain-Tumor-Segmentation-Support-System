# Brain Tumor Segmentation Support System

## 1. Tổng quan hệ thống

Hệ thống này được phát triển nhằm hỗ trợ bác sĩ và nhà nghiên cứu trong việc phân đoạn khối u não từ ảnh chụp MRI đa chuỗi bằng các mô hình học sâu. Mục tiêu chính là giảm thời gian kiểm tra hình ảnh, tăng độ chính xác trong nhận diện vùng khối u và cung cấp giao diện trực quan để xem kết quả, so sánh mô hình và xuất báo cáo.

Tên hệ thống: Brain Tumor Segmentation Support System

## 2. Mục tiêu của đề tài

Đề tài hướng tới việc xây dựng một hệ thống AI có thể:

- Nhận dữ liệu MRI đầu vào dưới dạng file ảnh y khoa (thường là NIfTI).
- Tiền xử lý dữ liệu trước khi đưa vào mô hình.
- Thực hiện phân đoạn khối u não bằng hai mô hình: U-Net và Swin UNETR.
- So sánh hiệu quả giữa các mô hình dựa trên các chỉ số đánh giá như Dice, IoU, Precision và Recall.
- Hiển thị kết quả phân đoạn dưới dạng mask, overlay và hình ảnh trực quan.
- Tính toán thể tích khối u và hỗ trợ tạo báo cáo kết quả.

## 3. Bối cảnh và ý nghĩa

Phân đoạn khối u não là một nhiệm vụ quan trọng trong chẩn đoán hình ảnh y khoa. Tuy nhiên, việc thực hiện thủ công bằng con người tốn nhiều thời gian và có thể bị ảnh hưởng bởi sự khác nhau giữa các bác sĩ cũng như chất lượng hình ảnh. Việc ứng dụng Deep Learning giúp tự động hóa quy trình, hỗ trợ chẩn đoán nhanh hơn và có thể làm nền tảng cho các hệ thống hỗ trợ lâm sàng trong tương lai.

## 4. Kiến trúc tổng thể hệ thống

Hệ thống được thiết kế theo mô hình 4 tầng chính như sau:

```text
Người dùng
   │
   ▼
Giao diện web (Streamlit / Gradio)
   │
   ▼
Tầng xử lý dữ liệu
   │
   ▼
Tầng suy luận AI
   │
   ▼
Tầng hiển thị và báo cáo
```

### 4.1 Luồng hoạt động tổng quát

1. Người dùng tải lên dữ liệu MRI.
2. Hệ thống đọc và tiền xử lý ảnh.
3. Dữ liệu được đưa vào mô hình phân đoạn.
4. Kết quả mask được tạo ra và hậu xử lý.
5. Hệ thống tính toán các chỉ số đánh giá và thể tích khối u.
6. Kết quả được hiển thị trực quan và có thể xuất báo cáo.

## 5. Kiến trúc phần mềm

### 5.1 Presentation Layer

Lớp này chịu trách nhiệm tương tác trực tiếp với người dùng, bao gồm:

- Upload dữ liệu MRI
- Chọn mô hình phân đoạn
- Hiển thị ảnh gốc, mask và overlay
- Trình bày kết quả số liệu và đồ thị
- Xuất báo cáo dưới dạng file

### 5.2 Application Layer

Lớp này chứa logic nghiệp vụ chính của hệ thống, bao gồm:

- Tải và kiểm tra dữ liệu đầu vào
- Gọi mô hình suy luận
- Tính Dice, IoU, Precision, Recall
- Tính thể tích khối u
- Quản lý luồng xử lý giữa các module

### 5.3 AI Layer

Lớp này thực hiện các tác vụ liên quan đến trí tuệ nhân tạo, gồm:

- Tiền xử lý ảnh MRI
- Suy luận bằng mô hình 3D U-Net
- Suy luận bằng mô hình Swin UNETR
- Hậu xử lý mask
- Đánh giá kết quả

### 5.4 Data Layer

Lớp này quản lý dữ liệu hệ thống, bao gồm:

- Bộ dữ liệu BraTS 2023
- File checkpoint của mô hình
- Kết quả suy luận đã lưu
- Log hệ thống và báo cáo

## 6. Các module chính trong hệ thống

### 6.1 Data Loader

Module này có nhiệm vụ đọc dữ liệu MRI từ các file NIfTI hoặc thư mục dữ liệu. Nó hỗ trợ:

- Đọc ảnh đầu vào
- Trích xuất các chuỗi MRI như T1, T1ce, T2, FLAIR
- Chuẩn hóa định dạng dữ liệu để đưa vào mô hình

### 6.2 Preprocessor

Module tiền xử lý thực hiện các bước như:

- Chuẩn hóa giá trị voxel
- Cắt hoặc resize ảnh cho phù hợp với đầu vào mô hình
- Tạo patch nếu cần thiết
- Chuyển đổi sang dạng tensor phù hợp với PyTorch

### 6.3 Model Manager

Module này quản lý các mô hình phân đoạn:

- U-Net
- Swin UNETR

Mỗi mô hình có thể được chọn tùy theo nhu cầu đánh giá hoặc so sánh.

### 6.4 Inference Engine

Module này thực hiện suy luận trên dữ liệu đã tiền xử lý và sinh ra output mask.

### 6.5 Evaluation

Module đánh giá so sánh kết quả mô hình với ground truth, bao gồm:

- Dice Score
- IoU
- Precision
- Recall
- Hausdorff Distance 95 (HD95)

### 6.6 Visualization

Module này chịu trách nhiệm hiển thị:

- Ảnh MRI gốc
- Mask dự đoán
- Overlay giữa ảnh và mask
- Biểu đồ so sánh giữa các mô hình

### 6.7 Report

Module xuất báo cáo chứa:

- Thông tin bệnh nhân hoặc case đầu vào
- Kết quả phân đoạn
- Chỉ số đánh giá
- Thể tích khối u
- Hình ảnh minh họa

## 7. Biểu đồ use case

### 7.1 Actor

- Người dùng: tải ảnh, chạy suy luận, xem kết quả, so sánh mô hình, xuất báo cáo.
- Quản trị viên: quản lý mô hình, kiểm tra hệ thống, theo dõi log.

### 7.2 Các use case chính

1. Upload MRI
2. Chọn mô hình phân đoạn
3. Thực hiện suy luận
4. Xem kết quả phân đoạn
5. So sánh kết quả giữa U-Net và Swin UNETR
6. Tính và xem thể tích khối u
7. Xuất báo cáo

## 8. Luồng hoạt động của hệ thống

### 8.1 Activity Diagram

```text
Start
  ↓
Upload MRI
  ↓
Preprocessing
  ↓
Select Model
  ↓
Inference
  ↓
Postprocessing
  ↓
Display Result
  ↓
Evaluate Metrics
  ↓
Export Report
  ↓
End
```

### 8.2 Sequence Diagram

```text
User → Web UI → Inference Controller → Preprocessor → Model → Postprocess → UI → User
```

## 9. Mô hình dữ liệu

### 9.1 MRIImage

Đại diện cho dữ liệu đầu vào, bao gồm:

- ID bệnh nhân
- File ảnh MRI
- Các chuỗi ảnh
- Thông tin về kích thước và định dạng

### 9.2 PredictionResult

Đại diện cho kết quả phân đoạn, bao gồm:

- Mask dự đoán
- Mô hình sử dụng
- Thời gian suy luận
- Các chỉ số đánh giá
- Thể tích khối u

### 9.3 Report

Đại diện cho báo cáo kết quả, gồm:

- Tóm tắt kết quả
- Hình ảnh minh họa
- Chỉ số đánh giá
- Kết luận ngắn

## 10. Cấu trúc thư mục dự kiến

```text
BrainTumorSegmentation/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── unet.py
│   └── swin_unetr.py
├── preprocessing/
├── evaluation/
├── visualization/
├── inference/
├── app/
│   └── streamlit.py
├── checkpoints/
├── reports/
└── configs/
```

## 11. Pipeline xử lý dữ liệu và AI

### 11.1 Pipeline đầu vào

```text
MRI Input
  ↓
Load NIfTI
  ↓
Normalize
  ↓
Crop / Resize
  ↓
Patch Extraction
  ↓
Inference
  ↓
Postprocess
  ↓
Segmentation Mask
  ↓
Metrics Calculation
  ↓
Volume Estimation
```

### 11.2 Các bước xử lý chính

- Đọc ảnh MRI từ file NIfTI.
- Chuẩn hóa độ sáng và giá trị voxel.
- Cắt hoặc resize ảnh theo kích thước phù hợp.
- Chia ảnh thành patch nếu cần thiết.
- Dự đoán mask bằng mô hình.
- Loại bỏ nhiễu và làm mịn mask.
- Tính toán các chỉ số hiệu năng.

## 12. Mô hình AI sử dụng

### 12.1 U-Net

U-Net là mô hình kiến trúc encoder-decoder phổ biến trong phân đoạn ảnh y khoa. Mô hình này hiệu quả với dữ liệu ảnh có cấu trúc không gian rõ ràng và thường được dùng làm baseline cho các bài toán segmentation.

### 12.2 Swin UNETR

Swin UNETR là kiến trúc hiện đại dựa trên Transformer, phù hợp với dữ liệu 3D. Mô hình có khả năng nắm bắt mối quan hệ toàn cục giữa các vùng ảnh, đặc biệt hữu ích trong việc phân đoạn các cấu trúc phức tạp như khối u não.

### 12.3 So sánh giữa hai mô hình

- U-Net: đơn giản, nhanh, dễ huấn luyện.
- Swin UNETR: mạnh hơn về mô hình hóa ngữ cảnh không gian và thông tin toàn cục, nhưng tốn nhiều tài nguyên hơn.

## 13. Đánh giá hiệu năng mô hình

Để đánh giá chất lượng phân đoạn, hệ thống sử dụng các metric sau:

- Dice Score: đo độ trùng khớp giữa mask dự đoán và ground truth.
- IoU: đo độ phủ của vùng dự đoán so với vùng thật.
- Precision: tỷ lệ vùng dự đoán đúng trong số tất cả vùng được dự đoán.
- Recall: tỷ lệ vùng thật được phát hiện thành công.
- HD95: đo khoảng cách giữa các bề mặt dự đoán và ground truth.

## 14. Công nghệ sử dụng

| Thành phần | Công nghệ đề xuất |
|---|---|
| Deep Learning | PyTorch, MONAI |
| Xử lý ảnh y khoa | NiBabel, NumPy |
| Trực quan hóa | Matplotlib, OpenCV |
| Giao diện | Streamlit hoặc Gradio |
| Huấn luyện | Google Colab / máy cá nhân |
| Dataset | BraTS 2023 |

## 15. Cơ sở dữ liệu và lưu trữ

Nếu hệ thống cần lưu lịch sử suy luận, có thể thiết kế bảng dữ liệu đơn giản như sau:

### Bảng Prediction

- ID
- PatientID
- Model
- Dice
- Volume
- Time
- Path

## 16. Kiến trúc triển khai

```text
Browser
  ↓
Streamlit UI
  ↓
Python Backend
  ↓
PyTorch + MONAI
  ↓
Trained Model
  ↓
MRI Dataset
```

## 17. Kế hoạch triển khai

| Giai đoạn | Nội dung |
|---|---|
| Phase 1 | Khảo sát bài toán, EDA dữ liệu, tiền xử lý |
| Phase 2 | Huấn luyện mô hình 3D U-Net |
| Phase 3 | Huấn luyện mô hình Swin UNETR |
| Phase 4 | Đánh giá và so sánh hiệu năng |
| Phase 5 | Xây dựng giao diện demo |
| Phase 6 | Viết báo cáo và chuẩn bị bảo vệ |

## 18. Kết quả mong đợi

Sau khi hoàn thiện, hệ thống kỳ vọng có thể:

- Phân đoạn khối u não từ ảnh MRI một cách tự động.
- Hiển thị kết quả rõ ràng và dễ hiểu.
- Cung cấp cơ sở để so sánh hiệu quả giữa các mô hình AI.
- Hỗ trợ quy trình nghiên cứu và demo sản phẩm một cách chuyên nghiệp.
