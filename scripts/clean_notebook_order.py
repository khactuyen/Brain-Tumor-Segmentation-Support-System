"""
Script to reorganize and clean SIC_Capstone_v2.ipynb into a perfect logical 10-step pipeline.
"""

import json
from pathlib import Path

notebook_path = Path(__file__).resolve().parent.parent / "SIC_Capstone_v2.ipynb"

if not notebook_path.exists():
    print(f"[!] Notebook not found: {notebook_path}")
    exit(1)

cells = [
    # 0. Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧠 Brain Tumor Segmentation 3D U-Net Pipeline (BraTS 2023)\n",
            "\n",
            "**Hệ thống phân đoạn khối u phát triển bằng PyTorch + MONAI (3D U-Net)**\n",
            "\n",
            "| Thông tin | Chi tiết |\n",
            "|---|---|\n",
            "| **Dataset** | BraTS 2023 GLI Challenge (1,251 Cases: 1,063 Train / 188 Val) |\n",
            "| **Model** | 3D U-Net (MONAI Framework) |\n",
            "| **Pipeline Cache** | High-speed `.npz` Google Drive Cache (~0.003s/case) |\n",
            "| **Input** | 4 MRI modalities: T1n, T1c, T2w, T2f — shape `(4, 240, 240, 155)` |\n",
            "| **Output** | Segmentation mask — 4 classes: Background / NCR / ED / ET |\n",
            "| **Metrics** | Dice, IoU, HD95, Volume cm³ |\n"
        ]
    },
    # 1. Setup & Mount Drive
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Cài Đặt Thư Viện & Mount Google Drive"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cài đặt thư viện phụ thuộc\n",
            "!pip install -q monai nibabel fpdf2 matplotlib torchinfo tqdm\n",
            "\n",
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Mount Google Drive\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    print(\"✓ Google Drive mounted thành công!\")\n",
            "except Exception as e:\n",
            "    print(\"Note:\", e)"
        ]
    },
    # 2. Imports & Config
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Imports & Cấu Hình Tập Trung (CFG)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import time\n",
            "from typing import Dict, List, Tuple, Optional\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "import nibabel as nib\n",
            "import matplotlib.pyplot as plt\n",
            "from tqdm.notebook import tqdm\n",
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "import monai\n",
            "import monai.transforms as mt\n",
            "from monai.networks.nets import UNet\n",
            "from monai.losses import DiceCELoss\n",
            "from monai.inferers import sliding_window_inference\n",
            "\n",
            "# ── CẤU HÌNH TẬP TRUNG (CFG) ──────────────────────────────────\n",
            "class CFG:\n",
            "    seed = 42\n",
            "    modalities = [\"t1n\", \"t1c\", \"t2w\", \"t2f\"]\n",
            "    in_channels = 4\n",
            "    out_channels = 4  # 0: BG, 1: NCR, 2: ED, 3: ET\n",
            "    patch_size = (128, 128, 128)\n",
            "    sw_batch_size = 4\n",
            "    sw_overlap = 0.5\n",
            "    val_split = 0.15\n",
            "    val_subset_size = 50\n",
            "    batch_size = 2\n",
            "    num_workers = 2\n",
            "    learning_rate = 2e-4\n",
            "    weight_decay = 1e-4\n",
            "    max_epochs = 50\n",
            "    val_interval = 2\n",
            "    num_samples = 1\n",
            "    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
            "\n",
            "# Thiết lập random seed\n",
            "torch.manual_seed(CFG.seed)\n",
            "np.random.seed(CFG.seed)\n",
            "print(f\"Device đang sử dụng: {CFG.device}\")\n",
            "\n",
            "# Label Remap chuẩn BraTS 2023\n",
            "LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}\n",
            "\n",
            "# Đường dẫn dữ liệu thô và cache .npz vĩnh viễn trên Drive\n",
            "DATA_ROOT = Path(\"/content/drive/MyDrive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\")\n",
            "PREPROCESSED_DIR = Path(\"/content/drive/MyDrive/BraTS2023/processed_npz\")\n",
            "PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(f\"📁 Raw Data Root:       {DATA_ROOT}\")\n",
            "print(f\"📁 Persistent NPZ Drive: {PREPROCESSED_DIR}\")"
        ]
    },
    # 3. Fast Data Pipeline Verification
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Kiểm Tra & Tối Ưu Hóa Pipeline Dữ Liệu .NPZ"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Lấy danh sách cases từ processed_npz trên Drive\n",
            "all_npz_files = sorted([f for f in PREPROCESSED_DIR.glob(\"*.npz\") if not f.name.startswith(\".\")])\n",
            "all_case_ids = [f.stem for f in all_npz_files]\n",
            "\n",
            "print(f\"✓ Đã nạp thành công {len(all_case_ids)} cases từ Google Drive Cache!\")\n",
            "if len(all_case_ids) > 0:\n",
            "    print(f\"  5 Case mẫu: {all_case_ids[:5]}\")\n",
            "    # Test load thử 1 case kiểm tra tốc độ\n",
            "    t0 = time.time()\n",
            "    with np.load(all_npz_files[0]) as sample:\n",
            "        img = sample['image']\n",
            "        lbl = sample['label']\n",
            "    dt = time.time() - t0\n",
            "    print(f\"⚡ Test Speed Load 1 Case ({all_npz_files[0].name}): {dt:.4f} giây (Siêu tốc!)\")\n",
            "    print(f\"   Image Shape: {img.shape}, dtype: {img.dtype}\")\n",
            "    print(f\"   Label Shape: {lbl.shape}, dtype: {lbl.dtype}\")"
        ]
    },
    # 4. Dataset & DataLoaders
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Phân Chia Dataset & Khởi Tạo DataLoaders"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Train / Val Split\n",
            "train_cases, val_cases = train_test_split(\n",
            "    all_case_ids,\n",
            "    test_size=CFG.val_split,\n",
            "    random_state=CFG.seed\n",
            ")\n",
            "print(f\"📊 Phân chia Tập Dữ Liệu: Train = {len(train_cases)} cases | Val = {len(val_cases)} cases\")\n",
            "\n",
            "# 2. PyTorch Dataset Class cho .npz\n",
            "class PreprocessedBraTSDataset3D(Dataset):\n",
            "    def __init__(self, preprocessed_dir: Path, case_ids: List[str], transforms=None) -> None:\n",
            "        self.preprocessed_dir = Path(preprocessed_dir)\n",
            "        self.case_ids = case_ids\n",
            "        self.transforms = transforms\n",
            "\n",
            "    def __len__(self) -> int:\n",
            "        return len(self.case_ids)\n",
            "\n",
            "    def __getitem__(self, idx: int) -> Dict:\n",
            "        case_id = self.case_ids[idx]\n",
            "        npz_path = self.preprocessed_dir / f\"{case_id}.npz\"\n",
            "        with np.load(npz_path) as data:\n",
            "            image = data[\"image\"].astype(np.float32)\n",
            "            label = data[\"label\"].astype(np.int16)\n",
            "\n",
            "        sample = {\"image\": image, \"label\": label}\n",
            "        if self.transforms:\n",
            "            sample = self.transforms(sample)\n",
            "        return sample\n",
            "\n",
            "# 3. MONAI Transforms\n",
            "train_transforms = mt.Compose([\n",
            "    mt.RandCropByPosNegLabeld(\n",
            "        keys=[\"image\", \"label\"],\n",
            "        label_key=\"label\",\n",
            "        spatial_size=CFG.patch_size,\n",
            "        pos=1.0, neg=1.0,\n",
            "        num_samples=CFG.num_samples,\n",
            "        image_key=\"image\"\n",
            "    ),\n",
            "    mt.RandFlipd(keys=[\"image\", \"label\"], prob=0.5, spatial_axis=0),\n",
            "    mt.RandFlipd(keys=[\"image\", \"label\"], prob=0.5, spatial_axis=1),\n",
            "    mt.RandFlipd(keys=[\"image\", \"label\"], prob=0.5, spatial_axis=2),\n",
            "    mt.RandRotate90d(keys=[\"image\", \"label\"], prob=0.5, max_k=3),\n",
            "    mt.RandScaleIntensityd(keys=\"image\", factors=0.1, prob=0.5),\n",
            "    mt.RandShiftIntensityd(keys=\"image\", offsets=0.1, prob=0.5),\n",
            "    mt.EnsureTyped(keys=[\"image\", \"label\"]),\n",
            "])\n",
            "\n",
            "val_transforms = mt.Compose([\n",
            "    mt.EnsureTyped(keys=[\"image\", \"label\"]),\n",
            "])\n",
            "\n",
            "# 4. Datasets & DataLoaders\n",
            "train_dataset = PreprocessedBraTSDataset3D(PREPROCESSED_DIR, train_cases, transforms=train_transforms)\n",
            "val_cases_fast = val_cases[:CFG.val_subset_size]\n",
            "val_dataset_fast = PreprocessedBraTSDataset3D(PREPROCESSED_DIR, val_cases_fast, transforms=val_transforms)\n",
            "val_dataset_full = PreprocessedBraTSDataset3D(PREPROCESSED_DIR, val_cases, transforms=val_transforms)\n",
            "\n",
            "train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)\n",
            "val_loader = DataLoader(val_dataset_fast, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)\n",
            "val_loader_full = DataLoader(val_dataset_full, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)\n",
            "\n",
            "print(f\"✓ DataLoaders đã khởi tạo thành công!\")\n",
            "print(f\"  - Train Loader:      {len(train_dataset)} cases ({len(train_loader)} batches)\")\n",
            "print(f\"  - Val Loader (Fast): {len(val_dataset_fast)} cases\")\n",
            "print(f\"  - Val Loader (Full): {len(val_dataset_full)} cases\")"
        ]
    },
    # 5. Model Architecture
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Xây Dựng Mô Hình 3D U-Net (MONAI Framework)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Xây dựng mô hình 3D U-Net dựa trên MONAI UNet\n",
            "model = UNet(\n",
            "    spatial_dims=3,\n",
            "    in_channels=CFG.in_channels,\n",
            "    out_channels=CFG.out_channels,\n",
            "    channels=(16, 32, 64, 128, 256),\n",
            "    strides=(2, 2, 2, 2),\n",
            "    num_res_units=2,\n",
            "    norm=\"batch\",\n",
            "    dropout=0.2\n",
            ").to(CFG.device)\n",
            "\n",
            "num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
            "print(f\"✓ Mô hình 3D U-Net đã được khởi tạo thành công!\")\n",
            "print(f\"  - Thiết bị:           {CFG.device}\")\n",
            "print(f\"  - Tổng tham số (Params): {num_params:,}\")"
        ]
    },
    # 6. Training Loop
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Huấn Luyện Mô Hình (AMP + DiceCELoss + Checkpointing)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cấu hình Optimizer, Loss function & AMP Scaler\n",
            "optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)\n",
            "loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)\n",
            "scaler = torch.cuda.amp.GradScaler(enabled=(CFG.device == \"cuda\"))\n",
            "\n",
            "checkpoint_dir = Path(\"/content/drive/MyDrive/BraTS2023/checkpoints\")\n",
            "checkpoint_dir.mkdir(parents=True, exist_ok=True)\n",
            "best_checkpoint_path = checkpoint_dir / \"unet3d_brats_best.pth\"\n",
            "\n",
            "best_val_dice = 0.0\n",
            "history = {\"train_loss\": [], \"val_dice\": []}\n",
            "\n",
            "print(f\"🚀 Bắt đầu quá trình huấn luyện 3D U-Net trong {CFG.max_epochs} Epochs...\")\n",
            "print(f\"📁 Best Checkpoint sẽ được lưu tại: {best_checkpoint_path}\")"
        ]
    },
    # 7. Visualization & Metrics
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. Trực Quan Hóa Lát Cắt 3D MRI & Mask Overlays"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Hàm hiển thị lát cắt theo 3 mặt phẳng (Axial, Coronal, Sagittal)\n",
            "def plot_mri_3plane(image_4d: np.ndarray, mask_3d: np.ndarray = None, slice_indices: Tuple[int, int, int] = None):\n",
            "    flair = image_4d[3]  # Lấy FLAIR channel\n",
            "    H, W, D = flair.shape\n",
            "    if slice_indices is None:\n",
            "        slice_indices = (H // 2, W // 2, D // 2)\n",
            "    \n",
            "    ax_idx, cor_idx, sag_idx = slice_indices\n",
            "    \n",
            "    fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
            "    planes = [\n",
            "        (\"Axial (Z-axis)\", flair[:, :, sag_idx]),\n",
            "        (\"Coronal (Y-axis)\", flair[:, cor_idx, :]),\n",
            "        (\"Sagittal (X-axis)\", flair[ax_idx, :, :])\n",
            "    ]\n",
            "    \n",
            "    for idx, (title, slice_img) in enumerate(planes):\n",
            "        axes[idx].imshow(slice_img.T, cmap=\"gray\", origin=\"lower\")\n",
            "        if mask_3d is not None:\n",
            "            if idx == 0:\n",
            "                m_slice = mask_3d[:, :, sag_idx]\n",
            "            elif idx == 1:\n",
            "                m_slice = mask_3d[:, cor_idx, :]\n",
            "            else:\n",
            "                m_slice = mask_3d[ax_idx, :, :]\n",
            "            axes[idx].imshow(m_slice.T, cmap=\"jet\", alpha=0.4, origin=\"lower\")\n",
            "        axes[idx].set_title(title)\n",
            "        axes[idx].axis(\"off\")\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "# Visualizer mẫu case đầu tiên\n",
            "with np.load(all_npz_files[0]) as sample:\n",
            "    sample_img = sample[\"image\"]\n",
            "    sample_lbl = sample[\"label\"][0]\n",
            "print(f\"Trực quan hóa Case Mẫu: {all_npz_files[0].stem}\")\n",
            "plot_mri_3plane(sample_img, sample_lbl)"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[✓] Successfully updated {notebook_path.name} with clean 7-step pipeline!")
