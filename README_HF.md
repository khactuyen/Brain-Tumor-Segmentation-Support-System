---
title: Brain Tumor Segmentation
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: hf_app.py
pinned: false
license: mit
short_description: 3D MRI brain tumor segmentation using U-Net / Swin UNETR (BraTS 2023)
---

# 🧠 Brain Tumor Segmentation — SIC Capstone 2026

Hệ thống phân đoạn khối u não 3D từ ảnh MRI đa phương thức (BraTS 2023).

## 🚀 Hướng dẫn deploy lên Hugging Face Spaces

### Bước 1: Tạo Space mới

1. Truy cập [huggingface.co/spaces](https://huggingface.co/spaces)
2. Bấm **Create new Space**
3. Điền thông tin:
   - **Space name**: `brain-tumor-segmentation` (hoặc tên tùy chọn)
   - **SDK**: chọn **Gradio**
   - **SDK version**: `4.44.0`
   - **Visibility**: Public hoặc Private
4. Bấm **Create Space**

### Bước 2: Cấu hình file README_HF.md

HF Spaces đọc metadata từ phần YAML frontmatter ở đầu file `README.md` của Space repo. File này (`README_HF.md`) đã có sẵn metadata đúng định dạng — **đổi tên thành `README.md`** khi push lên Space.

### Bước 3: Push code lên Space

```bash
# Clone Space repo về máy
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>

# Copy các file cần thiết từ project
cp -r /path/to/SIC_Capstone/web/ .
cp /path/to/SIC_Capstone/hf_app.py ./app.py

# Tạo requirements.txt ở root (HF Spaces dùng file này)
cp web/gradio_app/requirements.txt requirements.txt

# Push lên HF
git add .
git commit -m "Initial deploy: Brain Tumor Segmentation Gradio App"
git push
```

### Bước 4: Upload Model weights (nếu có)

Có 2 cách upload model:

**Cách 1: Dùng HF Model Hub (khuyến nghị)**
```bash
# Upload model lên HF Hub
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='models/unet.onnx',
    path_in_repo='unet.onnx',
    repo_id='<username>/<model-repo-name>',
    repo_type='model',
)
"
```

Sau đó trong `hf_app.py` thêm:
```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="<username>/<model-repo>", filename="unet.onnx")
os.environ["MODEL_PATH"] = model_path
```

**Cách 2: Upload trực tiếp vào Space repo**
> ⚠️ Chỉ phù hợp với model < 500MB. Model ONNX BraTS thường 100-400MB.

```bash
git lfs install
git lfs track "*.onnx" "*.pt"
cp /path/to/models/unet.onnx ./models/
git add models/unet.onnx
git commit -m "Add model weights"
git push
```

### Bước 5: Cấu hình biến môi trường (Space Settings)

Trong HF Space → **Settings** → **Variables and secrets**:

| Variable | Giá trị | Mô tả |
|---|---|---|
| `MODEL_PATH` | `/app/models/unet.onnx` | Đường dẫn model |
| `MODEL_FORMAT` | `onnx` | Định dạng model |
| `DEMO_MODE` | `0` | `0` = chạy thật, `1` = demo |
| `ROI_SIZE` | `128,128,128` | Patch size sliding window |
| `OVERLAP` | `0.5` | Sliding window overlap |

---

## 💡 Lưu ý về Hardware

| Tier | Chi phí | Phù hợp |
|---|---|---|
| **CPU Basic** (2 vCPU, 16GB RAM) | Miễn phí | Demo mode hoặc CPU inference |
| **CPU Upgrade** (8 vCPU, 32GB RAM) | ~$0.03/h | Inference thật với ONNX CPU |
| **T4 GPU** (16GB VRAM) | ~$0.05/h | Inference nhanh nhất |
| **ZeroGPU** | Miễn phí (theo hàng đợi) | Phù hợp cho demo public |

> 💡 **Khuyến nghị**: Dùng **ZeroGPU** cho demo public hoặc **CPU Upgrade** cho inference ổn định.

---

## 📁 Cấu trúc file cần thiết trên HF Space

```
HF Space repo/
├── app.py                       ← Entry point (từ hf_app.py)
├── README.md                    ← Metadata YAML + hướng dẫn (từ README_HF.md)
├── requirements.txt             ← Dependencies (từ web/gradio_app/requirements.txt)
├── web/
│   ├── inference_utils.py       ← Inference utilities
│   └── gradio_app/
│       ├── app.py               ← Gradio UI
│       ├── visualization.py     ← MRI visualization
│       └── report_generator.py  ← PDF generator
└── models/
    └── unet.onnx                ← Model weights (upload riêng)
```

---

## 🔗 Links hữu ích

- [HF Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio on HF Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio)
- [ZeroGPU Guide](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Git LFS for large files](https://huggingface.co/docs/hub/repositories-getting-started#adding-files)
