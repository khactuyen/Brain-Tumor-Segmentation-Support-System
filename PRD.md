# Product Requirements Document (PRD)

## 1. Tên sản phẩm

Brain Tumor Segmentation Support System

## 2. Mục tiêu sản phẩm

- Hỗ trợ bác sĩ và nhà nghiên cứu tự động phân đoạn khối u não từ ảnh MRI 3D.
- So sánh và cung cấp kết quả từ hai mô hình: U-Net và Swin UNETR.
- Hiển thị trực quan kết quả (mask, overlay), báo cáo chỉ số đánh giá và thể tích khối u.
- Cung cấp giao diện đơn giản để tải dữ liệu, chạy suy luận và xuất báo cáo.

## 3. Người dùng mục tiêu

- Bác sĩ chuyên khoa chẩn đoán hình ảnh.
- Nghiên cứu sinh và nhà nghiên cứu trong lĩnh vực xử lý ảnh y khoa.
- Sinh viên thực hành và demo trong các buổi thuyết trình đề tài.

## 4. Tính năng chính (MVP)

1. Upload dữ liệu MRI (NIfTI) và xem tiền xử lý cơ bản.
2. Chọn mô hình phân đoạn (`U-Net` hoặc `Swin UNETR`).
3. Chạy suy luận trên một case và nhận về `PredictionResult` gồm mask và thời gian tính toán.
4. Hiển thị ảnh gốc, mask và overlay theo các lát cắt (axial/coronal/sagittal).
5. Tính và hiển thị các chỉ số: Dice, IoU, Precision, Recall, HD95.
6. Tính thể tích khối u (cm³) từ mask phân đoạn.
7. Xuất báo cáo PDF/HTML chứa hình ảnh, chỉ số và thông tin case.

## 5. Yêu cầu chức năng chi tiết

- FR1: Hệ thống phải nhận file NIfTI (*.nii, *.nii.gz) và kiểm tra tính toàn vẹn cơ bản.
- FR2: Hệ thống phải chuẩn hóa intensity và resize/crop về kích thước đầu vào của mô hình.
- FR3: Người dùng có thể chọn mô hình và tùy chọn hậu xử lý (lọc nhiễu, morphological).
- FR4: Kết quả suy luận phải lưu thành file checkpointable (mask .nii.gz) và metadata JSON.
- FR5: Tính toán metric phải so sánh với ground truth nếu có sẵn.
- FR6: Giao diện phải hiển thị tiến trình suy luận và ước lượng thời gian hoàn thành.

## 6. Yêu cầu phi chức năng

- NFR1: Hệ thống phải hoạt động cục bộ (offline) trên máy có GPU được hỗ trợ (CUDA) hoặc CPU.
- NFR2: Thời gian suy luận cho một case trung bình không vượt quá 2 phút trên GPU tiêu chuẩn (tùy cấu hình).
- NFR3: Bảo mật file: dữ liệu bệnh nhân không được lưu trữ công khai; cung cấp tùy chọn xóa tạm thời.
- NFR4: Hệ thống phải có logging cho quá trình inference và lỗi.
- NFR5: Tài liệu hướng dẫn sử dụng cơ bản đi kèm.

## 7. Ràng buộc và giả định

- Giả sử dữ liệu input tuân thủ định dạng BraTS (các chuỗi T1, T1ce, T2, FLAIR).
- Có sẵn checkpoint của `U-Net` và `Swin UNETR` cho suy luận. Nếu không, chỉ chạy mô hình baseline.
- Môi trường triển khai có Python 3.8+ và PyTorch tương thích.

## 8. Thành phần hệ thống và giao diện

- Web UI: `Streamlit` hoặc `Gradio` cho demo nhanh.
- Backend: Python + PyTorch + MONAI cho luồng inference.
- Data Layer: thư mục `data/` với `train/`, `val/`, `test/` và `checkpoints/`.
- Output: `results/` chứa mask `.nii.gz`, ảnh PNG overlay, và report PDF/HTML.

## 9. Luồng người dùng (User Flow)

1. Người dùng mở UI và upload file NIfTI (hoặc chọn case có sẵn).
2. Chọn mô hình, cấu hình hậu xử lý và nhấn `Run`.
3. Hệ thống hiển thị tiến trình, sau đó trả về mask và các chỉ số.
4. Người dùng xem overlay, kiểm tra metric và xuất báo cáo.

## 10. Metrics để đánh giá sản phẩm

- Độ chính xác phân đoạn: Dice, IoU, HD95.
- Hiệu năng hệ thống: thời gian suy luận trung bình, tỷ lệ lỗi khi đọc file.
- Trải nghiệm người dùng: thời gian từ upload đến kết quả, tính trực quan của giao diện.

## 11. Kế hoạch phát triển (Roadmap ngắn hạn)

Phase 1 (2 tuần): EDA, data loader, preprocessor, baseline U-Net inference.
Phase 2 (3 tuần): Thêm Swin UNETR inference, evaluation metrics, save/load results.
Phase 3 (2 tuần): Xây dựng UI demo (Streamlit), visualization và export report.
Phase 4 (1 tuần): Testing, docs, và tối ưu inference (GPU/CPU configs).

## 12. Success Criteria

- MVP hoàn chỉnh khi user có thể upload 1 case, chạy inference với 1 trong 2 mô hình và nhận được mask + báo cáo trong vòng 2 phút (trên GPU).
- Metrics (Dice) trung bình tối thiểu 0.6 trên tập validation (tùy dataset).

## 13. Rủi ro và biện pháp giảm thiểu

- Rủi ro: Thiếu checkpoint cho Swin UNETR → Giảm thiểu: cung cấp mô hình thay thế hoặc hướng dẫn huấn luyện.
- Rủi ro: Dữ liệu đầu vào không chính xác → Giảm thiểu: validate file đầu vào và cung cấp hướng dẫn chuẩn hóa.

## 14. Tài liệu và hỗ trợ

- Include `README.md` with setup steps, `requirements.txt` and example commands to run UI and inference.

## 15. Kế hoạch làm việc chi tiết cho dự án

### 15.1 Giai đoạn 1 – Khảo sát và chuẩn bị dữ liệu

Mục tiêu của giai đoạn này là hiểu rõ bộ dữ liệu, chuẩn bị môi trường và xác định cách xử lý dữ liệu phù hợp trước khi triển khai mô hình.

#### 15.1.1 Nội dung công việc

- Tìm hiểu cấu trúc thư mục và định dạng dữ liệu của bộ BraTS 2023.
- Xác định các loại file cần sử dụng: ảnh MRI các chuỗi T1, T1ce, T2, FLAIR; file mask ground truth; file metadata liên quan.
- Kiểm tra số lượng case có sẵn, phân bố train/val/test và độ đầy đủ của dữ liệu.
- Làm quen với thao tác đọc, kiểm tra và trực quan hóa dữ liệu từ file NIfTI bằng thư viện như NiBabel và NumPy.
- Thực hiện EDA ban đầu để thống kê:
  - kích thước ảnh đầu vào,
  - số lượng slice trên mỗi case,
  - giá trị voxel và phạm vi intensity,
  - các case bị thiếu hoặc lỗi.

#### 15.1.2 Công việc cụ thể từng bước

1. Bước 1 – Thiết lập môi trường phát triển
   - Cài đặt Python 3.8+.
   - Cài đặt các thư viện cần thiết: PyTorch, MONAI, NiBabel, NumPy, Matplotlib, OpenCV, Streamlit (hoặc Gradio).
   - Kiểm tra phiên bản CUDA nếu có GPU, hoặc xác nhận chạy trên CPU.
   - Tạo môi trường ảo riêng cho dự án để tránh xung đột thư viện.

2. Bước 2 – Chuẩn bị thư mục làm việc trên Colab
   - Sử dụng Google Drive làm nơi lưu dữ liệu và kết quả chạy.
   - Tạo các thư mục trên Drive như: /content/drive/MyDrive/BraTS2023/raw/, /content/drive/MyDrive/BraTS2023/processed/, /content/drive/MyDrive/BraTS2023/results/.
   - Xác định nơi lưu dữ liệu gốc, dữ liệu tiền xử lý và output của mô hình để dễ theo dõi.
   - Nếu dùng Colab, có thể không cần tạo cấu trúc thư mục phức tạp trên máy local.

3. Bước 3 – Khảo sát cấu trúc bộ dữ liệu BraTS
   - Liệt kê các thư mục case trong bộ dữ liệu.
   - Xác định tên file ảnh cho từng case: T1, T1ce, T2, FLAIR và mask segmentation.
   - Kiểm tra có bao nhiêu case trong train/val/test và cách phân chia dữ liệu.
   - Ghi lại quy ước đặt tên file để viết code đọc dữ liệu tự động.

4. Bước 4 – Đọc và kiểm tra một case mẫu
   - Chọn 1 case đầu tiên từ bộ dữ liệu.
   - Dùng NiBabel đọc các file NIfTI.
   - In ra shape, affine, dtype và thông tin metadata của từng file.
   - Xác nhận rằng dữ liệu có thể đọc đúng và không bị lỗi.

5. Bước 5 – Trực quan hóa dữ liệu MRI và mask
   - Chọn 1 slice ở giữa volume để hiển thị.
   - Vẽ 4 ảnh MRI (T1, T1ce, T2, FLAIR) và mask ground truth trên cùng một hình.
   - Quan sát sự khác biệt giữa các chuỗi MRI và vùng khối u.
   - Ghi nhận các nhận xét ban đầu để hỗ trợ tiền xử lý.

6. Bước 6 – Thống kê dữ liệu ban đầu (EDA)
   - Tính thống kê về kích thước ảnh của các case.
   - Kiểm tra số lượng slice trên mỗi volume.
   - Đo lường phạm vi giá trị voxel và các giá trị ngoại lệ.
   - Xác định các case bị thiếu file hoặc có định dạng bất thường.

7. Bước 7 – Xây dựng pipeline đọc dữ liệu đầu tiên
   - Viết hàm load_case(case_id) để tự động đọc các file liên quan của một case.
   - Trả về danh sách ảnh MRI và mask dưới dạng numpy array.
   - Chuẩn hóa đầu ra để các module sau có thể dùng chung.

8. Bước 8 – Kiểm tra khả năng chuyển dữ liệu thành tensor
   - Chuyển dữ liệu từ numpy sang tensor PyTorch.
   - Đảm bảo shape phù hợp với đầu vào mô hình.
   - Xác định cần thêm batch dimension hay channel dimension không.

9. Bước 9 – Ghi nhận kết quả và chuẩn bị cho giai đoạn 2
   - Lưu lại kết quả kiểm tra một case mẫu.
   - Ghi chú những vấn đề phát sinh khi đọc dữ liệu.
   - Chuẩn bị danh sách công việc tiếp theo cho tiền xử lý và dataset loader.

#### 15.1.3 Kết quả mong đợi của giai đoạn 1

- Hiểu được cấu trúc dữ liệu BraTS 2023.
- Có môi trường phát triển sẵn sàng để triển khai.
- Có pipeline đọc dữ liệu NIfTI đầu tiên hoạt động được trên ít nhất 1 case mẫu.
- Có bản danh sách dữ liệu và kế hoạch tiền xử lý cho các giai đoạn tiếp theo.

### 15.2 Giai đoạn 2 – Xây dựng pipeline tiền xử lý

- Viết module đọc và tải dữ liệu MRI từ thư mục NIfTI.
- Chuẩn hóa voxel intensity cho từng chuỗi MRI.
- Resize/crop ảnh về kích thước phù hợp với mô hình.
- Tạo dataset loader cho train/val/test.
- Xác định cách xử lý dữ liệu đầu vào cho cả U-Net và Swin UNETR.

### 15.3 Giai đoạn 3 – Xây dựng mô hình baseline

- Triển khai mô hình 3D U-Net.
- Huấn luyện hoặc sử dụng checkpoint có sẵn để chạy inference.
- Kiểm tra đầu ra mask và hậu xử lý mask.
- Đánh giá kết quả ban đầu bằng Dice, IoU, Precision, Recall.

### 15.4 Giai đoạn 4 – Tích hợp mô hình Swin UNETR

- Tìm hiểu kiến trúc Swin UNETR và cấu hình phù hợp.
- Tải hoặc huấn luyện mô hình trên dữ liệu tương ứng.
- Chạy inference trên cùng tập dữ liệu để so sánh kết quả với U-Net.
- Ghi chép điểm mạnh, điểm yếu của từng mô hình.

### 15.5 Giai đoạn 5 – Đánh giá và tối ưu hóa

- Tính toán các metric phân đoạn và thể tích khối u.
- So sánh hiệu năng giữa hai mô hình về độ chính xác và thời gian.
- Thử nghiệm các chiến lược hậu xử lý để cải thiện mask.
- Tối ưu cấu hình mô hình nếu có GPU mạnh hơn hoặc dữ liệu lớn hơn.

### 15.6 Giai đoạn 6 – Xây dựng giao diện demo

- Xây dựng giao diện upload file và chọn mô hình.
- Hiển thị ảnh gốc, mask và overlay theo các lát cắt.
- Cho phép người dùng xem kết quả và xuất báo cáo.
- Tối ưu trải nghiệm người dùng để demo dễ hiểu.

### 15.7 Giai đoạn 7 – Hoàn thiện báo cáo và bảo vệ

- Tổng hợp kết quả thực nghiệm.
- Viết phần giới thiệu, phương pháp, kết quả và kết luận.
- Chuẩn bị slide bảo vệ và video demo nếu cần.
- Hoàn thiện tài liệu hướng dẫn sử dụng và cách chạy hệ thống.

## 16. Phân công công việc đề xuất

- Nhóm 1: Thu thập dữ liệu, tiền xử lý, data loader.
- Nhóm 2: Triển khai và huấn luyện mô hình U-Net.
- Nhóm 3: Triển khai và huấn luyện mô hình Swin UNETR.
- Nhóm 4: Xây dựng giao diện và báo cáo kết quả.

## 17. Timeline đề xuất

| Giai đoạn | Thời gian đề xuất |
|---|---| 
| Khảo sát dữ liệu và môi trường | 3–5 ngày |
| Xây dựng tiền xử lý | 4–6 ngày |
| Baseline U-Net | 5–7 ngày |
| Swin UNETR | 7–10 ngày |
| Đánh giá và tối ưu | 3–5 ngày |
| Giao diện demo | 4–5 ngày |
| Báo cáo và bảo vệ | 3–4 ngày |

---

Phiên bản tài liệu: 1.1
Người viết: Nhóm SIC Capstone 2026
