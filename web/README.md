# Web App — README

## Khởi chạy

### 1. Cài dependencies
```bash
pip install -r requirements_web.txt
```

### 2. Đặt model vào thư mục `models/`
Sau khi chạy notebook và export model, copy file vào:
```
SIC_Capstone 2026/
└── models/
    ├── unet.onnx           ← hoặc unet.pt (TorchScript)
    └── swin_unetr.onnx
```

### 3. Chạy server
```bash
# Từ thư mục web/
MODEL_PATH=../models/unet.onnx uvicorn app:app --host 0.0.0.0 --port 8000

# Hoặc với TorchScript model
MODEL_PATH=../models/unet.pt MODEL_FORMAT=torchscript uvicorn app:app --port 8000
```

### 4. Test API
```bash
# Health check
curl http://localhost:8000/health

# Predict (4 NIfTI files)
curl -X POST http://localhost:8000/predict \
  -F "t1n=@path/to/case-t1n.nii.gz" \
  -F "t1c=@path/to/case-t1c.nii.gz" \
  -F "t2w=@path/to/case-t2w.nii.gz" \
  -F "t2f=@path/to/case-t2f.nii.gz"

# Download NIfTI mask
curl -X POST "http://localhost:8000/predict?return_nifti=true" \
  -F "t1n=@..." -F "t1c=@..." -F "t2w=@..." -F "t2f=@..." \
  -o segmentation.nii.gz
```

### 5. API Docs
Truy cập http://localhost:8000/docs để xem Swagger UI
