# ⚙️ WORKFLOW — Học BraTS Segmentation qua Notebook (Colab)
**Bạn làm trên Colab, 1 notebook duy nhất, mỗi step = 1 cell. Mentor review từng step.**

> Cách học: mở Colab, tạo notebook `learn_brats.ipynb`. Mỗi Step bên dưới = 1 cell trong notebook.
> Bạn tự gõ code (đừng copy chữa lệch). Khi 1 cell chạy đạt kết quả mong đợi → đánh `[x]` + báo mentor → sang cell kế.

---

## Setup chung (chạy 1 lần đầu notebook)

**Cell 0 — Mount Drive & tìm dataset:**
```python
from google.colab import drive
drive.mount('/content/drive')
import os
from pathlib import Path

# ← ĐỔI thành đường dẫn dataset trên Drive của bạn
DATA_ROOT = Path("/content/drive/MyDrive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
cases = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
print(f"Số case: {len(cases)}")
print("Sample:", cases[:5])
```
- **Kết quả đạt:** in ra số case (>0) và 5 tên case đầu. Báo mentor.

---

## GIAI ĐOẠN 1 — Đọc dữ liệu NIfTI

**Step 1 (cell 1) — Đọc 1 case:**
```python
import nibabel as nib
import numpy as np

case_path = DATA_ROOT / cases[0]
files = sorted(p.name for p in case_path.glob("*.nii.gz"))
print("Files:", files)

img = nib.load(case_path / f"{cases[0]}-t1n.nii.gz")
print("shape:", img.shape)
print("affine:\n", img.affine)
print("dtype:", img.get_fdata().dtype)
```
- **Kết quả đạt (DONE):** ra `shape (240, 240, 155)`, affine `(4,4)`. Báo mentor.

**Step 2 (cell 2) — Đọc cả case (4 modal + seg):**
```python
modalities = ["t1n", "t1c", "t2w", "t2f"]
data = {}
for m in modalities:
    data[m] = nib.load(case_path / f"{cases[0]}-{m}.nii.gz").get_fdata()
seg = nib.load(case_path / f"{cases[0]}-seg.nii.gz").get_fdata()

for m in modalities:
    print(m, data[m].shape, data[m].min(), data[m].max())
print("seg", seg.shape, "giá trị duy nhất:", np.unique(seg))
```
- **Kết quả đạt:** 4 chuỗi cùng shape, `np.unique(seg)` = `[0 1 2 3]`. Báo mentor.
- 💡 **Số `np.unique(seg)` quan trọng** = số lớp phân đoạn.

---

## GIAI ĐOẠN 2 — EDA (visualization)

**Step 3 (cell 3) — Vẽ 1 lát cắt giữa của FLAIR:**
```python
import matplotlib.pyplot as plt

mid = seg.shape[2] // 2
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(data["t2f"][:, :, mid], cmap="gray"); axes[0].set_title("FLAIR axial")
axes[1].imshow(seg[:, :, mid], cmap="jet");        axes[1].set_title("Mask")
plt.tight_layout(); plt.show()
```
- **Kết quả:** nhìn thấy não + khối u (màu trên mask). Báo mentor.

**Step 4 (cell 4) — 4 chuỗi + mask cùng 1 slice:**
```python
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, m in enumerate(modalities):
    axes[i].imshow(data[m][:, :, mid], cmap="gray")
    axes[i].set_title(m)
axes[4].imshow(seg[:, :, mid], cmap="jet"); axes[4].set_title("seg")
plt.tight_layout(); plt.show()
```
- **Kết quả:** 5 panel. **Ghi ra:** chuỗi nào thấy u rõ nhất? Báo mentor (kèm trả lời).

---

## GIAI ĐOẠN 3 — Preprocessor

**Step 5 (cell 5) — Normalize z-score 1 chuỗi:**
```python
def normalize(arr):
    mean, std = arr.mean(), arr.std()
    return (arr - mean) / (std + 1e-8)

norm = normalize(data["t1n"])
print("mean:", norm.mean().round(4), "std:", norm.std().round(4))
```
- **Kết quả:** `mean ~0.0`, `std ~1.0`. Báo mentor.

**Step 6 (cell 6) — Normalize cả 4 chuỗi, stack thành 1 tensor:**
```python
import torch
stacked = np.stack([normalize(data[m]) for m in modalities], axis=0)  # (4,H,W,D)
tensor = torch.from_numpy(stacked).float()
print("shape:", tensor.shape, tensor.dtype)
```
- **Kết quả:** `(4, 240, 240, 155)`, float32. Báo mentor.

---

## GIAI ĐOẠN 4 — Dataset & DataLoader

**Step 7 (cell 7) — class BraTSDataset3D:**
```python
from torch.utils.data import Dataset

class BraTSDataset3D(Dataset):
    def __init__(self, root, case_ids):
        self.root = Path(root)
        self.case_ids = case_ids
        self.modalities = ["t1n", "t1c", "t2w", "t2f"]

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        data = []
        for m in self.modalities:
            arr = nib.load(self.root / cid / f"{cid}-{m}.nii.gz").get_fdata()
            data.append(normalize(arr))
        img = torch.from_numpy(np.stack(data)).float()
        seg = torch.from_numpy(nib.load(self.root / cid / f"{cid}-seg.nii.gz").get_fdata()).long()
        return img, seg

ds = BraTSDataset3D(DATA_ROOT, cases)
print("len:", len(ds))
```
- **Kết quả:** `len(ds)` = số case. Báo mentor.

**Step 8 (cell 8) — DataLoader + 1 batch:**
```python
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=2, shuffle=True)
img, seg = next(iter(dl))
print("img:", img.shape, "| seg:", seg.shape)
```
- **Kết quả:** `img (2,4,240,240,155)`, `seg (2,240,240,155)`. Báo mentor.
- 💡 Thấy batch dim `2` ở đầu — đó là "batch_size".

---

## GIAI ĐOẠN 5 — Model

**Step 9 (cell 9) — Khởi tạo U-Net (MONAI):**
```python
from monai.networks.nets import UNet
import torch.nn as nn

model = UNet(
    spatial_dims=3,
    in_channels=4, out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
n_params = sum(p.numel() for p in model.parameters())
print("Params:", f"{n_params:,}")
```
- **Kết quả:** n_params ~ hàng triệu (>1M). Báo mentor.

**Step 10 (cell 10) — Forward dummy:**
```python
x = torch.randn(1, 4, 128, 128, 128)   # 1 batch, 4 chuỗi, patch 128³
with torch.no_grad():
    y = model(x)
print("output:", y.shape)
```
- **Kết quả:** `(1, 4, 128, 128, 128)` — khớp input. Báo mentor.
- 💡 Ở cell này input dùng **128³** không phải 240³ (U-Net cần patch nhỏ cho train).

---

## GIAI ĐOẠN 6 — Training

**Step 11 (cell 11) — 1 bước train (loss giảm):**
```python
loss_fn = nn.CrossEntropyLoss()      # đơn giản để học
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(3):
    optimizer.zero_grad()
    y = model(x)                                  # (1,4,128,128,128)
    target = torch.randint(0, 4, (1,128,128,128)) # mask giả
    loss = loss_fn(y, target)
    loss.backward()
    optimizer.step()
    print(f"step {step}: loss {loss.item():.4f}")
```
- **Kết quả:** loss giảm dần sau 3 step. Báo mentor.
- 💡 `target` đang là **mask giả** — thay mask thật sau.

**Step 12 (cell 12) — Vài epoch, dùng mask thật:**
```python
from torch.utils.data import DataLoader
# cắt nhỏ để train nhanh
dl = DataLoader(ds, batch_size=1, shuffle=True)
for epoch in range(3):
    total = 0.0
    for img, seg in dl:
        y = model(img)                             # thay 128→240? model cần patch
        loss = loss_fn(y, seg)
        loss.backward(); optimizer.step()
        total += loss.item()
    print(f"epoch {epoch}: avg loss {total/len(dl):.4f}")
```
- **Kết quả:** loss giảm dần theo epoch (chạy 1-2 epoch cho đủ lệnh). Báo mentor.
- ⚠️ **Sẽ gặp lỗi shape** (model 128³ vs ảnh 240³) → đây là bài học: cần **crop 128³ hoặc sliding window**. Gặp lỗi gì dán nguyên vào đây.

---

## GIAI ĐOẠN 7 — Evaluation

**Step 13 (cell 13) — Dice & IoU:**
```python
def dice(pred, target, eps=1e-8):
    inter = (pred & target).sum()
    return (2*inter + eps) / (pred.sum() + target.sum() + eps)

def iou(pred, target, eps=1e-8):
    inter = (pred & target).sum()
    union = (pred | target).sum()
    return (inter + eps) / (union + eps)

# giả: so sánh 2 mask cùng 1 slice
a = seg[:, :, mid].numpy() if torch.is_tensor(seg) else seg[:, :, mid]
a = a > 0
b = a.copy()
print("dice(gt,gt):", dice(a,b).round(3), "iou:", iou(a,b).round(3))
```
- **Kết quả:** `dice=1.0`, `iou=1.0` (so sánh giống nhau). Báo mentor.

---

## Lộ trình sau khi xong 13 step (sang giai đoạn sau)
- **14:** Merge về repo (`src/` + `scripts/`) — thay code bạn đã viết.
- **15:** Sliding Window Inference cho ảnh 240³.
- **16:** Swin UNETR + so sánh.
- **17:** Streamlit UI + báo cáo PDF.

Bạn báo mình từng cell kết quả. Mình chỉ dẫn, không làm thay file bạn.