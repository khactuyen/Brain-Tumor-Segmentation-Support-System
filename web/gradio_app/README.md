# Gradio Web App — Brain Tumor Segmentation

Giao diện web thân thiện cho người dùng, sử dụng Gradio.

## Tính năng

- **🧠 Tab 1 — Phân đoạn khối u**: Upload 4 file MRI NIfTI → nhận kết quả segmentation
  - Visualization overlay 3-view (Axial · Coronal · Sagittal) với màu cho từng vùng
  - Thống kê thể tích chi tiết (cm³, voxels, %) cho WT/TC/NCR/ED/ET
  - Xuất mask NIfTI để dùng tiếp trong phần mềm y tế
  - **Xuất báo cáo PDF** giống notebook (thông tin case + bảng thể tích + ảnh overlay + disclaimer)

- **🔬 Tab 2 — Xem MRI**: 3-view viewer cho từng modality riêng lẻ

- **📖 Tab 3 — Hướng dẫn**: Giải thích quy trình, nhãn phân đoạn, và disclaimer

## Chạy local

```bash
# Cài dependencies
pip install -r requirements.txt

# Chạy app
python app.py
# Mở http://localhost:7860
```

## Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `MODEL_PATH` | `../../models/unet.onnx` | Đường dẫn model |
| `MODEL_FORMAT` | `auto` | `onnx`, `torchscript`, hoặc `auto` |
| `DEMO_MODE` | `1` | `1` = demo (dummy mask), `0` = inference thật |
| `ROI_SIZE` | `128,128,128` | Patch size |
| `OVERLAP` | `0.5` | Sliding window overlap |

## Demo Mode

Nếu `MODEL_PATH` không tồn tại hoặc `DEMO_MODE=1`, app tự động chuyển sang **demo mode**:
- Giao diện đầy đủ hoạt động bình thường
- Mask phân đoạn là hình cầu giả để minh họa
- Cảnh báo demo hiển thị rõ ràng cho người dùng

## Deploy lên Hugging Face Spaces

Xem `README_HF.md` ở root project để biết hướng dẫn đầy đủ.
