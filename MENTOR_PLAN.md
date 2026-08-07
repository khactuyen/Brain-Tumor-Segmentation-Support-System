# 🎓 MENTOR PLAN — Học Deep Learning qua dự án
**Dự án:** Brain Tumor Segmentation (BraTS 2023) — U-Net & Swin UNETR

**Nguyên tắc học:** Bạn code trước, mentor review + chỉ lỗi + giải thích. Mentor không làm thay.
**Phương thức:** Chạy trên **Google Colab** (GPU miễn phí), dữ liệu từ Google Drive.

---

## Môi trường thực tế của bạn
- Python 3.11, PyTorch (CPU), MONAI 1.6 — có sẵn trên máy local, dùng để học/làm bài tập nhỏ.
- Train thật toàn bộ trên **Colab GPU**.

---

## Lộ trình học (mỗi chặng = 1 khái niệm cốt lõi + code chạy được)

| # | Giai đoạn | Bạn học | Chạy được cái gì |
|---|---|---|---|
| 1 | **NIfTI + MRI** | NiBabel, shape, affine, slice | Đọc & in thông tin 1 case |
| 2 | **EDA** | numpy, matplotlib | Ảnh 4 chuỗi + mask + thống kê |
| 3 | **Preprocessing** | normalization z-score, crop/pad | Preprocessor chuẩn |
| 4 | **Dataset** | PyTorch `Dataset`, tensor, batch | Dataset load được 1 batch |
| 5 | **Model** | U-Net kiến trúc, forward, params | forward pass 1 batch |
| 6 | **Training** | loss, optimizer, loop, AMP | Trainer chạy vài epoch |
| 7 | **Evaluation** | Dice, IoU, Precision, Recall, HD95 | Số liệu thật trên val |
| 8 | **Inference** | sliding window, checkpoint | Engine chạy 1 case |
| 9 | **Model 2** | Swin UNETR, so sánh 2 model | So sánh U-Net vs Swin |
| 10 | **UI + Report** | Streamlit, PDF/HTML | App + báo cáo |

---

## Luồng học đề xuất (Colab)
Nguồn hướng dẫn: dùng notebook `SIC_Capstone_v2.ipynb` trong repo làm tham chiếu — nó đã có đầy đủ pipeline.
Bạn viết lại từ đầu theo cách hiểu của mình, so sánh với bản tham chiếu.

---

## Checklist tuyệt (Definition of Done cho từng chặng)
- [ ] Chạy được đoạn code, không lỗi import/syntax
- [ ] Đọc được code, giải thích được bằng lời
- [ ] Có thông tin/xem được slice từ dữ liệu
- [ ] Code gọn, đúng chuẩn — đó chính là mục tiêu nếu muốn chạy trên máy

---

## Đường dẫn dữ liệu (hỏi bạn khi có thực tế)
BraTS 2023 GLI Training — trên Google Drive bạn. Đường dẫn chính xác sẽ được note tại đây khi bạn cung cấp.
- Chuỗi MRI: `t1n`, `t1c`, `t2w`, `t2f` (ground truth: `seg`)
- Format file: `<case_id>-<modal>.nii.gz`

_Ghi lại thông tin sau mỗi phiên học vào đây._