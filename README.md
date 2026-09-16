# Brain Tumor Segmentation Support System

Dự án SIC Capstone xây dựng pipeline phân đoạn u não 3D từ MRI đa modality theo chuẩn BraTS. Repository gồm hai notebook thực nghiệm và ứng dụng web inference.

## Hai notebook

| Notebook | Mô hình | Vai trò | Đánh giá |
|---|---|---|---|
| `SIC_Capstone_v1.ipynb` | 3D U-Net | Baseline, pipeline đầy đủ | 875 train, 188 validation, 188 held-out test; checkpoint epoch 40 |
| `SIC_Capstone_v2.ipynb` | Swin UNETR | Mô hình nâng cấp dùng Transformer | 812 train, 204 validation; train 20 epoch |

Hai notebook không dùng cùng cohort, split và protocol đánh giá. Vì vậy chỉ số Dice của v1 và v2 chưa phải phép so sánh công bằng.

## Dữ liệu và nhãn

Mỗi ca có bốn modality MRI: T1 native, T1 contrast, T2 weighted và T2-FLAIR. Nhãn BraTS gồm background 0, NCR/NET 1, ED 2 và ET 4 trong raw data; pipeline remap raw label 4 thành class nội bộ 3.

- WT: toàn bộ u, `label > 0`
- TC: lõi u, `label ∈ {1, 3}`
- ET: vùng u bắt thuốc, `label = 3`

## Pipeline

`NIfTI → kiểm tra shape/affine/spacing → Z-score theo modality → crop/patch 96³ → augmentation → DataLoader → model 3D → sliding-window inference → hậu xử lý → Dice/IoU/Precision/Recall/HD95`

Notebook v1 có held-out test. Notebook v2 hiện chỉ báo cáo validation.

## Ứng dụng web

Ứng dụng nằm trong `BraTS_Model/`, sử dụng FastAPI/Gradio và checkpoint đã huấn luyện.
Link Gradio App : https://huggingface.co/spaces/vokhactuyen/BraTS_Model
```powershell
cd BraTS_Model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_web.txt
python app.py
```

Các file chính:

- `BraTS_Model/app.py`: điểm khởi chạy ứng dụng
- `BraTS_Model/inference_utils.py`: preprocessing và inference
- `BraTS_Model/best_Swin.pth`: checkpoint Swin UNETR
- `BraTS_Model/unet3d_brats_best.pth`: checkpoint 3D U-Net

## Cài đặt notebook

Notebook được thiết kế cho Google Colab hoặc máy có GPU CUDA:

```powershell
pip install -r requirements.txt
```

Mở notebook tương ứng, cấu hình đường dẫn dataset/checkpoint và chạy theo thứ tự cell. Dataset, checkpoint, cache và kết quả lớn không được đưa vào repository.

## Tài liệu phát hành

Repository chỉ theo dõi ba tài liệu trong `doc/`:

1. `SIC_AI_Capstone Project_Final Report_Brain Tumor Segmentation_Revised.updated.final.docx`
2. `SIC_AI_Chapter 11. Starting an AI Project.pdf`
3. `Report_BraTS-GLI-2026_nzd2b1s9.pdf`

Các bản nháp Word, PDF render, hình kiểm tra và artefact tạo tài liệu đều được ignore.

## Lưu ý khoa học

Dice và HD95 là chỉ số đánh giá nghiên cứu. Ứng dụng hiện tại phục vụ nghiên cứu và minh họa, chưa được kiểm định lâm sàng và không thay thế chẩn đoán của bác sĩ.
