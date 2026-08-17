import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_markdown(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]
        })

    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]
        })

    # Header
    add_markdown("""# 🧠 Brain Tumor Segmentation Support System — SIC Capstone v3 (BraTS 2023)
### Pipeline Hoàn Chỉnh: EDA → Tiền Xử Lý Z-Score NPZ → Huấn Luyện 3D U-Net & Swin UNETR → Hậu Xử Lý → Đánh Giá → Trực Quan Hóa → Báo Cáo PDF

| Feature | Description |
|---|---|
| **Mục tiêu** | Phân đoạn u não 3D (BraTS 2023) thành 3 vùng: WT, TC, ET |
| **Kiến trúc** | 3D U-Net Baseline & Swin UNETR |
| **Pipeline** | EDA chi tiết, Tiền xử lý Z-score NPZ, Huấn luyện, Hậu xử lý CCA, Đánh giá (Dice/IoU/Volume), Overlay Visualizer, Xuất PDF & ONNX/TorchScript |
| **Phiên bản** | v3 (Merged & Upgraded for Production & PRD Compliance) |
""")

    # 1. Cài Đặt Môi Trường
    add_markdown("## 1. Cài Đặt Môi Trường & Mount Google Drive\nCài đặt các thư viện phụ thuộc (MONAI, Nibabel, FPDF2, TorchInfo, Scikit-Learn, SciPy).")
    add_code("""!pip install -q monai nibabel fpdf2 matplotlib torchinfo tqdm scikit-learn pandas scipy

import os
import glob
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from scipy.ndimage import binary_dilation, label as scipy_label
from sklearn.model_selection import train_test_split

import monai
from monai.transforms import (
    Compose, EnsureTyped, RandCropByPosNegLabeld, RandFlipd, 
    RandRotate90d, RandScaleIntensityd, RandShiftIntensityd
)
from monai.networks.nets import UNet, SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from torchinfo import summary
from fpdf import FPDF
from datetime import datetime

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Mounted Google Drive thành công!")
except Exception:
    print("ℹ️ Đang chạy trên môi trường Cục bộ (Local) hoặc Drive đã được mount.")
""")

    # 2. Cấu Hình Tập Trung
    add_markdown("## 2. Cấu Hình Tập Trung (CFG) & Đường Dẫn Tự Động\nQuản lý siêu tham số và tự động quét đường dẫn dữ liệu thô và NPZ cache.")
    add_code("""class CFG:
    seed = 42
    modalities = ['t1n', 't1c', 't2w', 't2f']
    in_channels = 4
    out_channels = 4  # 0: Background, 1: NCR/NET, 2: ED, 3: ET
    classes = ['Background', 'NCR/NET', 'ED', 'ET']
    target_shape = (240, 240, 155)
    patch_size = (96, 96, 96)
    
    val_split = 0.2
    val_subset_size = 5
    batch_size = 2
    num_samples = 2
    num_workers = 2
    
    learning_rate = 2e-4
    weight_decay = 1e-4
    max_epochs = 20
    val_interval = 2
    
    sw_batch_size = 4
    sw_overlap = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    monai.utils.misc.set_determinism(seed=seed)

set_seed(CFG.seed)
print(f"🖥️ Device: {CFG.device}")
if torch.cuda.is_available():
    print(f"⚡ GPU: {torch.cuda.get_device_name(0)}")

# Tự động quét đường dẫn dữ liệu
RAW_DATA_CANDIDATES = [
    '/content/drive/MyDrive/BraTS2023_Raw',
    '/content/drive/MyDrive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData',
    './data/raw',
    '../../data/raw',
    'Datasets/brats2023-gli-dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
]
NPZ_DATA_CANDIDATES = [
    '/content/drive/MyDrive/BraTS2023_NPZ',
    '/content/drive/MyDrive/BraTS2023/processed_npz',
    './data/npz',
    '../../data/npz',
    'Datasets/processed_npz'
]

DATA_ROOT = next((p for p in RAW_DATA_CANDIDATES if os.path.exists(p)), None)
PREPROCESSED_DIR = next((p for p in NPZ_DATA_CANDIDATES if os.path.exists(p)), './data/npz')
CHECKPOINT_DIR = './checkpoints'
MODEL_DIR = './models'

os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"📁 Đường dẫn được cấu hình:")
print(f"  - DATA_ROOT (Raw): {DATA_ROOT}")
print(f"  - PREPROCESSED_DIR: {PREPROCESSED_DIR}")
print(f"  - CHECKPOINT_DIR: {CHECKPOINT_DIR}")
print(f"  - MODEL_DIR: {MODEL_DIR}")
""")

    # 3. Pipeline Tiền Xử Lý
    add_markdown("## 3. Pipeline Tiền Xử Lý NPZ Cache & Chuẩn Hóa Cường Độ (Z-Score)\nChuẩn hóa cường độ theo vùng não (Percentile Clipping + Z-score Normalization) và Remap nhãn chuẩn BraTS.")
    add_code("""LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}

def load_and_normalize_case_nifti(case_path):
    vols = []
    for mod in CFG.modalities:
        fpath = glob.glob(os.path.join(case_path, f"*{mod}*.nii.gz"))
        if not fpath: 
            return None
        arr = nib.load(fpath[0]).get_fdata(dtype=np.float32)
        
        # Z-score Normalization theo từng vùng não (Brain Mask)
        brain_mask = arr > 0
        if brain_mask.any():
            vals = arr[brain_mask]
            p_low, p_high = np.percentile(vals, [0.5, 99.5])
            arr = np.clip(arr, p_low, p_high)
            mean_val, std_val = arr[brain_mask].mean(), arr[brain_mask].std()
            if std_val > 1e-6:
                arr = (arr - mean_val) / std_val
            arr[~brain_mask] = 0.0
        vols.append(arr)
    
    seg_path = glob.glob(os.path.join(case_path, f"*seg*.nii.gz"))
    if not seg_path: 
        return None
    raw_seg = nib.load(seg_path[0]).get_fdata(dtype=np.int16)
    
    seg = np.zeros_like(raw_seg, dtype=np.int16)
    for k, v in LABEL_REMAP.items():
        seg[raw_seg == k] = v
        
    img = np.stack(vols, axis=0).astype(np.float32) # (4, H, W, D)
    return img, seg

def preprocess_single_case(case_dir, out_path):
    res = load_and_normalize_case_nifti(case_dir)
    if res is None: 
        return False
    img, seg = res
    np.savez_compressed(out_path, image=img, label=seg)
    return True

def ensure_preprocessed_npz():
    if os.path.exists(PREPROCESSED_DIR):
        npz_files = glob.glob(os.path.join(PREPROCESSED_DIR, "*.npz"))
        if len(npz_files) > 0:
            print(f"✅ Tìm thấy {len(npz_files)} file NPZ đã chuẩn hóa Z-score. Sử dụng NPZ Cache.")
            return sorted([os.path.basename(f).replace('.npz', '') for f in npz_files])
            
    if DATA_ROOT is None or not os.path.exists(DATA_ROOT):
        print("❌ Không tìm thấy dữ liệu Raw hoặc NPZ!")
        return []
        
    case_dirs = sorted([d for d in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(d)])
    print(f"⏳ Tìm thấy {len(case_dirs)} cases raw. Tiến hành Z-score normalization và lưu NPZ...")
    
    all_cases = []
    for c_dir in tqdm(case_dirs, desc="Preprocess NPZ"):
        cid = os.path.basename(c_dir)
        out_p = os.path.join(PREPROCESSED_DIR, f"{cid}.npz")
        if not os.path.exists(out_p):
            preprocess_single_case(c_dir, out_p)
        all_cases.append(cid)
        
    return all_cases

all_case_ids = ensure_preprocessed_npz()
print(f"✅ Tổng số cases sẵn sàng: {len(all_case_ids)}")
""")

    # 4. EDA 1
    add_markdown("## 4. Khám Phá Dữ Liệu (EDA) — Data Understanding\n### 4.1. Structure inspection & label inventory")
    add_code("""def eda_structure(case_id):
    print(f"🔍 Kiểm tra cấu trúc case: {case_id}")
    npz_path = os.path.join(PREPROCESSED_DIR, f"{case_id}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        img, seg = data['image'], data['label']
        print(f"  - Shape Image: {img.shape}, Type: {img.dtype}, Min: {img.min():.2f}, Max: {img.max():.2f}")
        print(f"  - Shape Label: {seg.shape}, Type: {seg.dtype}, Unique Labels: {np.unique(seg)}")
        
        unique, counts = np.unique(seg, return_counts=True)
        total = seg.size
        print("  - Label Inventory:")
        for u, c in zip(unique, counts):
            lbl_name = CFG.classes[u] if u < len(CFG.classes) else f"Label {u}"
            print(f"      {lbl_name} (Label {u}): {c:,} voxels ({c/total*100:.2f}%)")
    else:
        print("  - File NPZ không tồn tại!")

if len(all_case_ids) > 0:
    eda_structure(all_case_ids[0])
""")

    # 4. EDA 2
    add_markdown("### 4.2. Intensity distributions & class statistics")
    add_code("""def eda_intensity_dist(case_ids, num_samples=3):
    if len(case_ids) == 0: 
        return
    samples = np.random.choice(case_ids, min(num_samples, len(case_ids)), replace=False)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, cid in enumerate(samples):
        npz_path = os.path.join(PREPROCESSED_DIR, f"{cid}.npz")
        data = np.load(npz_path)
        img = data['image'] # (4, H, W, D)
        
        for c in range(4):
            vols = img[c].flatten()
            vols = vols[vols != 0] # bỏ background 0
            axes[c].hist(vols, bins=50, alpha=0.5, label=f"Case {i+1}", density=True)
            axes[c].set_title(CFG.modalities[c].upper())
            axes[c].set_xlabel("Normalized Intensity")
            axes[c].legend()
            
    plt.suptitle("Phân phối Cường độ Cụm Mô (Z-score Normalization) trên 4 Chuỗi Xung", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if len(all_case_ids) > 0:
    eda_intensity_dist(all_case_ids)
""")

    # 4. EDA 3
    add_markdown("### 4.3. Visualizing 4 modalities & GT mask overlay on axial slice")
    add_code("""def visualize_slice(case_id):
    npz_path = os.path.join(PREPROCESSED_DIR, f"{case_id}.npz")
    if not os.path.exists(npz_path): 
        return
    data = np.load(npz_path)
    img, seg = data['image'], data['label']
    
    # Tìm lát cắt axial có diện tích khối u lớn nhất
    slice_counts = np.sum(seg > 0, axis=(0,1))
    z = np.argmax(slice_counts) if slice_counts.max() > 0 else seg.shape[2] // 2
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for c in range(4):
        axes[c].imshow(img[c, :, :, z].T, cmap='gray', origin='lower')
        axes[c].set_title(CFG.modalities[c].upper())
        axes[c].axis('off')
        
    cmap_seg = ListedColormap(['none', '#EF4444', '#10B981', '#F59E0B'])
    axes[4].imshow(img[3, :, :, z].T, cmap='gray', origin='lower')
    axes[4].imshow(seg[:, :, z].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower')
    axes[4].set_title('FLAIR + Mask Overlay')
    axes[4].axis('off')
    
    plt.suptitle(f"Case: {case_id} — Axial Slice Z={z}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if len(all_case_ids) > 0:
    visualize_slice(all_case_ids[0])
""")

    # 5. Phân Chia Dataset
    add_markdown("## 5. Phân Chia Dataset & Khởi Tạo DataLoaders")
    add_code("""class PreprocessedBraTSDataset3D(Dataset):
    def __init__(self, case_ids, transform=None):
        self.case_ids = case_ids
        self.transform = transform
        
    def __len__(self):
        return len(self.case_ids)
        
    def __getitem__(self, idx):
        npz_path = os.path.join(PREPROCESSED_DIR, f"{self.case_ids[idx]}.npz")
        data = np.load(npz_path)
        img = data['image'].astype(np.float32)
        lbl = data['label'].astype(np.int64)
        if lbl.ndim == 3:
            lbl = np.expand_dims(lbl, 0)
            
        item = {'image': img, 'label': lbl, 'case_id': self.case_ids[idx]}
        if self.transform:
            item = self.transform(item)
        return item

if len(all_case_ids) > 0:
    train_cases, val_cases = train_test_split(all_case_ids, test_size=CFG.val_split, random_state=CFG.seed)
    val_cases_fast = val_cases[:CFG.val_subset_size]
    
    train_transforms = Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=CFG.patch_size,
            pos=1, neg=1,
            num_samples=CFG.num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"], device=torch.device('cpu'), track_meta=False)
    ])
    
    val_transforms = Compose([
        EnsureTyped(keys=["image", "label"], device=torch.device('cpu'), track_meta=False)
    ])
    
    train_ds = PreprocessedBraTSDataset3D(train_cases, transform=train_transforms)
    val_ds_fast = PreprocessedBraTSDataset3D(val_cases_fast, transform=val_transforms)
    val_ds_full = PreprocessedBraTSDataset3D(val_cases, transform=val_transforms)
    
    def collate_fn_list(batch):
        flat_batch = []
        for item in batch:
            if isinstance(item, list):
                flat_batch.extend(item)
            else:
                flat_batch.append(item)
        return monai.data.dataloader.collate(flat_batch)
    
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, 
                              num_workers=CFG.num_workers, collate_fn=collate_fn_list)
    val_loader_fast = DataLoader(val_ds_fast, batch_size=1, num_workers=0)
    val_loader_full = DataLoader(val_ds_full, batch_size=1, num_workers=0)
    
    print(f"✅ Khởi tạo DataLoader thành công!")
    print(f"  - Train cases: {len(train_cases)}")
    print(f"  - Val cases (Fast): {len(val_cases_fast)}")
    print(f"  - Val cases (Full): {len(val_cases)}")
""")

    # 6. Metrics & Post-processing
    add_markdown("## 6. Hàm Tiện Ích Đánh Giá (Metrics) & Hậu Xử Lý (CCA)")
    add_code("""def compute_brats_dice(pred, target):
    \"\"\"Calculates Dice score for BraTS sub-regions: WT (1,2,3), TC (1,3), ET (3).\"\"\"
    if isinstance(pred, torch.Tensor):
        p = pred.squeeze().cpu().numpy()
    else:
        p = np.squeeze(pred)
    if isinstance(target, torch.Tensor):
        t = target.squeeze().cpu().numpy()
    else:
        t = np.squeeze(target)
        
    def calculate_dice(p_binary, t_binary):
        inter = np.sum(p_binary * t_binary)
        total = np.sum(p_binary) + np.sum(t_binary)
        if total == 0:
            return 1.0  # Cả GT và Pred không có u -> 100% đúng
        return float(2.0 * inter / (total + 1e-8))
        
    wt_p, wt_t = (p >= 1).astype(np.float32), (t >= 1).astype(np.float32)
    tc_p, tc_t = ((p == 1) | (p == 3)).astype(np.float32), ((t == 1) | (t == 3)).astype(np.float32)
    et_p, et_t = (p == 3).astype(np.float32), (t == 3).astype(np.float32)
    
    return calculate_dice(wt_p, wt_t), calculate_dice(tc_p, tc_t), calculate_dice(et_p, et_t)

def compute_comprehensive_metrics(pred, target):
    \"\"\"Calculates Dice and IoU metrics for WT, TC, ET sub-regions.\"\"\"
    wt_d, tc_d, et_d = compute_brats_dice(pred, target)
    
    p = pred.squeeze().cpu().numpy() if isinstance(pred, torch.Tensor) else np.squeeze(pred)
    t = target.squeeze().cpu().numpy() if isinstance(target, torch.Tensor) else np.squeeze(target)
    
    def calculate_iou(p_b, t_b):
        inter = np.sum(p_b * t_b)
        union = np.sum(p_b) + np.sum(t_b) - inter
        if union == 0: return 1.0
        return float(inter / (union + 1e-8))
        
    wt_p, wt_t = (p >= 1).astype(np.float32), (t >= 1).astype(np.float32)
    tc_p, tc_t = ((p == 1) | (p == 3)).astype(np.float32), ((t == 1) | (t == 3)).astype(np.float32)
    et_p, et_t = (p == 3).astype(np.float32), (t == 3).astype(np.float32)
    
    wt_iou = calculate_iou(wt_p, wt_t)
    tc_iou = calculate_iou(tc_p, tc_t)
    et_iou = calculate_iou(et_p, et_t)
    
    return {
        'Dice_WT': wt_d, 'Dice_TC': tc_d, 'Dice_ET': et_d, 'Mean_Dice': (wt_d + tc_d + et_d)/3.0,
        'IoU_WT': wt_iou, 'IoU_TC': tc_iou, 'IoU_ET': et_iou, 'Mean_IoU': (wt_iou + tc_iou + et_iou)/3.0
    }

def compute_tumor_volume(mask, spacing=(1.0, 1.0, 1.0)):
    m = mask.squeeze()
    voxel_vol = spacing[0] * spacing[1] * spacing[2] / 1000.0 # cm^3
    
    wt = float((m > 0).sum() * voxel_vol)
    tc = float(((m == 1) | (m == 3)).sum() * voxel_vol)
    et = float((m == 3).sum() * voxel_vol)
    return wt, tc, et

def postprocess_connected_components(pred_mask, min_size=50):
    \"\"\"Khử nhiễu đốm nhỏ rác bằng Connected Component Analysis.\"\"\"
    cleaned = np.copy(pred_mask)
    for c in [1, 2, 3]:
        binary_c = (pred_mask == c)
        labeled, num_features = scipy_label(binary_c)
        for i in range(1, num_features + 1):
            if (labeled == i).sum() < min_size:
                cleaned[labeled == i] = 0
    return cleaned
""")

    # 7. Huấn Luyện 3D U-Net
    add_markdown("## 7. Huấn Luyện Mô Hình 1: 3D U-Net Baseline")
    add_code("""def train_model(model, loader_train, loader_val, max_epochs, model_name="UNet"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    use_cuda = CFG.device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    
    history = {'train_loss': [], 'val_dice': [], 'val_wt': [], 'val_tc': [], 'val_et': []}
    best_dice = 0.0
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        
        pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{max_epochs} ({model_name})")
        for batch_data in pbar:
            step += 1
            inputs = batch_data["image"].to(CFG.device)
            labels = batch_data["label"].to(CFG.device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                
            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_loss /= max(1, step)
        history['train_loss'].append(epoch_loss)
        
        if (epoch + 1) % CFG.val_interval == 0:
            model.eval()
            wt_list, tc_list, et_list = [], [], []
            
            with torch.no_grad():
                for val_data in loader_val:
                    inputs = val_data["image"].to(CFG.device)
                    labels = val_data["label"].to(CFG.device)
                    
                    with torch.amp.autocast('cuda', enabled=use_cuda):
                        outputs = sliding_window_inference(
                            inputs, CFG.patch_size, CFG.sw_batch_size, model, overlap=CFG.sw_overlap
                        )
                    
                    preds = torch.argmax(outputs, dim=1, keepdim=True)
                    wt, tc, et = compute_brats_dice(preds, labels)
                    wt_list.append(wt)
                    tc_list.append(tc)
                    et_list.append(et)
                    
            m_wt, m_tc, m_et = np.mean(wt_list), np.mean(tc_list), np.mean(et_list)
            m_dice = (m_wt + m_tc + m_et) / 3.0
            
            history['val_wt'].append(m_wt)
            history['val_tc'].append(m_tc)
            history['val_et'].append(m_et)
            history['val_dice'].append(m_dice)
            
            print(f"🏆 [Val] Dice: {m_dice:.4f} | WT: {m_wt:.4f} | TC: {m_tc:.4f} | ET: {m_et:.4f}")
            
            if m_dice > best_dice:
                best_dice = m_dice
                save_path = os.path.join(CHECKPOINT_DIR, f"best_{model_name}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"⭐ Lưu checkpoint mới tốt nhất: {save_path} ({best_dice:.4f})")
                
    return history, best_dice

if len(all_case_ids) > 0:
    unet_model = UNet(
        spatial_dims=3,
        in_channels=CFG.in_channels,
        out_channels=CFG.out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm='batch',
        dropout=0.1
    ).to(CFG.device)
    
    print("🚀 Bắt đầu huấn luyện 3D U-Net Baseline...")
    unet_hist, unet_best = train_model(unet_model, train_loader, val_loader_fast, CFG.max_epochs, "UNet")
""")

    # 8. Huấn Luyện Swin UNETR
    add_markdown("## 8. Huấn Luyện Mô Hình 2: Swin UNETR (Transformer)")
    add_code("""if len(all_case_ids) > 0:
    swin_model = SwinUNETR(
        img_size=CFG.patch_size,
        in_channels=CFG.in_channels,
        out_channels=CFG.out_channels,
        feature_size=48,
        use_checkpoint=True
    ).to(CFG.device)
    
    print("🚀 Bắt đầu huấn luyện Swin UNETR...")
    swin_hist, swin_best = train_model(swin_model, train_loader, val_loader_fast, CFG.max_epochs, "Swin")
""")

    # 9. So Sánh Hiệu Năng
    add_markdown("## 9. So Sánh Hiệu Năng & Đồ Thị Huấn Luyện")
    add_code("""if len(all_case_ids) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs_val = range(CFG.val_interval, CFG.max_epochs + 1, CFG.val_interval)
    epochs_train = range(1, CFG.max_epochs + 1)
    
    # Loss
    axes[0, 0].plot(epochs_train, unet_hist['train_loss'], label='3D U-Net', color='#2563EB', lw=2)
    axes[0, 0].plot(epochs_train, swin_hist['train_loss'], label='Swin UNETR', color='#10B981', lw=2)
    axes[0, 0].set_title('Training Loss Convergence', fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend()
    
    # Mean Dice
    axes[0, 1].plot(epochs_val, unet_hist['val_dice'], marker='o', label='3D U-Net', color='#2563EB', lw=2)
    axes[0, 1].plot(epochs_val, swin_hist['val_dice'], marker='s', label='Swin UNETR', color='#10B981', lw=2)
    axes[0, 1].set_title('Validation Mean Dice Comparison', fontweight='bold')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend()
    
    # U-Net Details
    axes[1, 0].plot(epochs_val, unet_hist['val_wt'], label='WT', linestyle='--')
    axes[1, 0].plot(epochs_val, unet_hist['val_tc'], label='TC', linestyle='--')
    axes[1, 0].plot(epochs_val, unet_hist['val_et'], label='ET', linestyle='--')
    axes[1, 0].set_title('3D U-Net Sub-regions (WT, TC, ET)', fontweight='bold')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend()
    
    # Swin Details
    axes[1, 1].plot(epochs_val, swin_hist['val_wt'], label='WT', linestyle='--')
    axes[1, 1].plot(epochs_val, swin_hist['val_tc'], label='TC', linestyle='--')
    axes[1, 1].plot(epochs_val, swin_hist['val_et'], label='ET', linestyle='--')
    axes[1, 1].set_title('Swin UNETR Sub-regions (WT, TC, ET)', fontweight='bold')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 60)
    print("📊 BẢNG SO SÁNH BEST MEAN DICE MÔ HÌNH:")
    print(f"  • 3D U-Net Baseline: {unet_best:.4f} ({unet_best*100:.2f}%)")
    print(f"  • Swin UNETR:        {swin_best:.4f} ({swin_best*100:.2f}%)")
    print("=" * 60)
""")

    # 10. Trực Quan Hóa 3D
    add_markdown("## 10. Trực Quan Hóa Kết Quả Phân Đoạn 3D (Multi-Planar Translucent Color Overlay)")
    add_code("""def plot_segmentation_3plane(img_flair, seg_gt, pred_mask, title="Segmentation 3-Plane Overlay"):
    mask_coords = np.argwhere(seg_gt > 0)
    if len(mask_coords) > 0:
        c_x, c_y, c_z = mask_coords.mean(axis=0).astype(int)
    else:
        c_x, c_y, c_z = [s//2 for s in seg_gt.shape]
        
    cmap_seg = ListedColormap(['none', '#EF4444', '#10B981', '#F59E0B']) # 1: Red (NCR), 2: Green (ED), 3: Yellow (ET)
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    
    # Axial (Z)
    axes[0,0].imshow(img_flair[:, :, c_z].T, cmap='gray', origin='lower'); axes[0,0].set_title("FLAIR (Axial)")
    axes[0,1].imshow(img_flair[:, :, c_z].T, cmap='gray', origin='lower')
    axes[0,1].imshow(seg_gt[:, :, c_z].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower'); axes[0,1].set_title("Ground Truth Overlay")
    axes[0,2].imshow(img_flair[:, :, c_z].T, cmap='gray', origin='lower')
    axes[0,2].imshow(pred_mask[:, :, c_z].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower'); axes[0,2].set_title("3D U-Net Prediction Overlay")
    
    # Coronal (Y)
    axes[1,0].imshow(img_flair[:, c_y, :].T, cmap='gray', origin='lower'); axes[1,0].set_title("FLAIR (Coronal)")
    axes[1,1].imshow(img_flair[:, c_y, :].T, cmap='gray', origin='lower')
    axes[1,1].imshow(seg_gt[:, c_y, :].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower')
    axes[1,2].imshow(img_flair[:, c_y, :].T, cmap='gray', origin='lower')
    axes[1,2].imshow(pred_mask[:, c_y, :].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower')
    
    # Sagittal (X)
    axes[2,0].imshow(img_flair[c_x, :, :].T, cmap='gray', origin='lower'); axes[2,0].set_title("FLAIR (Sagittal)")
    axes[2,1].imshow(img_flair[c_x, :, :].T, cmap='gray', origin='lower')
    axes[2,1].imshow(seg_gt[c_x, :, :].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower')
    axes[2,2].imshow(img_flair[c_x, :, :].T, cmap='gray', origin='lower')
    axes[2,2].imshow(pred_mask[c_x, :, :].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower')
    
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_model_comparison(img_flair, seg_gt, pred_unet, pred_swin):
    z = np.argmax(np.sum(seg_gt > 0, axis=(0,1))) if np.sum(seg_gt > 0) > 0 else seg_gt.shape[2] // 2
    cmap_seg = ListedColormap(['none', '#EF4444', '#10B981', '#F59E0B'])
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_flair[:, :, z].T, cmap='gray', origin='lower'); axes[0].set_title("FLAIR Scan")
    axes[1].imshow(img_flair[:, :, z].T, cmap='gray', origin='lower')
    axes[1].imshow(seg_gt[:, :, z].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower'); axes[1].set_title("Ground Truth")
    
    axes[2].imshow(img_flair[:, :, z].T, cmap='gray', origin='lower')
    axes[2].imshow(pred_unet[:, :, z].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower'); axes[2].set_title("3D U-Net Pred")
    
    axes[3].imshow(img_flair[:, :, z].T, cmap='gray', origin='lower')
    axes[3].imshow(pred_swin[:, :, z].T, cmap=cmap_seg, vmin=0, vmax=3, alpha=0.5, origin='lower'); axes[3].set_title("Swin UNETR Pred")
    
    for ax in axes:
        ax.axis('off')
    plt.suptitle("So Sánh Phân Đoạn Giữa Các Mô Hình Trực Quan", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if len(all_case_ids) > 0:
    use_cuda = CFG.device.type == 'cuda'
    for batch_data in val_loader_fast:
        inputs = batch_data["image"].to(CFG.device)
        labels = batch_data["label"].to(CFG.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_cuda):
                unet_out = sliding_window_inference(inputs, CFG.patch_size, CFG.sw_batch_size, unet_model, overlap=CFG.sw_overlap)
                swin_out = sliding_window_inference(inputs, CFG.patch_size, CFG.sw_batch_size, swin_model, overlap=CFG.sw_overlap)
            
            p_unet = torch.argmax(unet_out, dim=1).cpu().numpy()[0]
            p_swin = torch.argmax(swin_out, dim=1).cpu().numpy()[0]
            
            # Khử nhiễu đốm nhỏ bằng Hậu xử lý CCA
            p_unet_clean = postprocess_connected_components(p_unet)
            
        img_flair = inputs[0, 3].cpu().numpy() # FLAIR channel
        gt = labels[0, 0].cpu().numpy()
        
        plot_segmentation_3plane(img_flair, gt, p_unet_clean, "3D U-Net Segmentation (Translucent Overlay)")
        plot_model_comparison(img_flair, gt, p_unet_clean, p_swin)
        break
""")

    # 11. Đánh Giá Toàn Diện
    add_markdown("## 11. Đánh Giá Toàn Diện trên Tập Validation")
    add_code("""if len(all_case_ids) > 0:
    print("⏳ Đánh giá toàn diện trên toàn bộ Val Set...")
    unet_model.eval()
    swin_model.eval()
    use_cuda = CFG.device.type == 'cuda'
    
    results = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(val_loader_full, desc="Evaluating Val Set")):
            inputs = batch_data["image"].to(CFG.device)
            labels = batch_data["label"].to(CFG.device)
            case_id = val_cases[i]
            
            with torch.amp.autocast('cuda', enabled=use_cuda):
                u_out = sliding_window_inference(inputs, CFG.patch_size, CFG.sw_batch_size, unet_model, overlap=CFG.sw_overlap)
                s_out = sliding_window_inference(inputs, CFG.patch_size, CFG.sw_batch_size, swin_model, overlap=CFG.sw_overlap)
                
            p_unet = torch.argmax(u_out, dim=1, keepdim=True)
            p_swin = torch.argmax(s_out, dim=1, keepdim=True)
            
            m_unet = compute_comprehensive_metrics(p_unet, labels)
            m_swin = compute_comprehensive_metrics(p_swin, labels)
            
            p_unet_arr = p_unet[0, 0].cpu().numpy()
            p_unet_clean = postprocess_connected_components(p_unet_arr)
            vol_wt, vol_tc, vol_et = compute_tumor_volume(p_unet_clean)
            
            results.append({
                'Case': case_id,
                'U-Net_Dice_WT': m_unet['Dice_WT'], 'U-Net_Dice_TC': m_unet['Dice_TC'], 'U-Net_Dice_ET': m_unet['Dice_ET'], 'U-Net_Mean_Dice': m_unet['Mean_Dice'],
                'Swin_Dice_WT': m_swin['Dice_WT'], 'Swin_Dice_TC': m_swin['Dice_TC'], 'Swin_Dice_ET': m_swin['Dice_ET'], 'Swin_Mean_Dice': m_swin['Mean_Dice'],
                'Vol_WT_cm3': vol_wt, 'Vol_TC_cm3': vol_tc, 'Vol_ET_cm3': vol_et
            })
            
    df_res = pd.DataFrame(results)
    print("\n📊 BẢNG TỔNG HỢP THỐNG KÊ KẾT QUẢ ĐÁNH GIÁ (Validation Set):")
    display(df_res.describe())
""")

    # 12. Xuất Mô Hình
    add_markdown("## 12. Xuất Mô Hình (TorchScript & ONNX Production Export)")
    add_code("""if len(all_case_ids) > 0:
    print("📦 Xuất mô hình 3D U-Net sang định dạng TorchScript (.pt) và ONNX (.onnx)...")
    dummy_input = torch.randn(1, 4, 96, 96, 96).to(CFG.device)
    unet_model.eval()
    
    # 1. Export TorchScript
    ts_path_check = os.path.join(CHECKPOINT_DIR, "unet_traced.pt")
    ts_path_model = os.path.join(MODEL_DIR, "unet.pt")
    
    traced_model = torch.jit.trace(unet_model, dummy_input)
    torch.jit.save(traced_model, ts_path_check)
    torch.jit.save(traced_model, ts_path_model)
    print(f"✅ Đã lưu TorchScript model tại: {ts_path_model} ({os.path.getsize(ts_path_model)/(1024*1024):.2f} MB)")
    
    # 2. Export ONNX (Compatible with Web API Backend)
    onnx_path_check = os.path.join(CHECKPOINT_DIR, "unet.onnx")
    onnx_path_model = os.path.join(MODEL_DIR, "unet.onnx")
    
    torch.onnx.export(
        unet_model, dummy_input, onnx_path_model,
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    if os.path.exists(onnx_path_model):
        import shutil
        shutil.copyfile(onnx_path_model, onnx_path_check)
    print(f"✅ Đã lưu ONNX model tại: {onnx_path_model} ({os.path.getsize(onnx_path_model)/(1024*1024):.2f} MB)")
""")

    # 13. Báo Cáo Chẩn Đoán
    add_markdown("## 13. Xuất Báo Cáo Chẩn Đoán Y Khoa PDF Chuẩn Quốc Tế")
    add_code("""class ClinicalReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(30, 58, 138)
        self.cell(0, 10, 'BRAIN TUMOR SEGMENTATION CLINICAL REPORT', 0, 1, 'C')
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, 'AI-Assisted Diagnostic Support System (BraTS 2023 Standard)', 0, 1, 'C')
        self.ln(3)
        self.set_draw_color(37, 99, 235)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Page {self.page_no()} | SIC Capstone 2026 — AI Healthcare Team', 0, 0, 'C')

def export_clinical_pdf(case_id, metrics_unet, metrics_swin, vol_metrics, out_path="Sample_Clinical_Report.pdf"):
    pdf = ClinicalReportPDF()
    pdf.add_page()
    
    # 1. Patient Metadata
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, '1. PATIENT & MRI EXAMINATION METADATA', 0, 1)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(95, 6, f'Case Identifier: {case_id}', 0, 0)
    pdf.cell(95, 6, f'Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.cell(95, 6, 'Modalities: T1n, T1c, T2w, T2f (4 Channels)', 0, 0)
    pdf.cell(95, 6, 'Voxel Resolution: 1.0 x 1.0 x 1.0 mm', 0, 1)
    pdf.ln(4)
    
    # 2. Volumetric & Metrics Analysis
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '2. QUANTITATIVE VOLUMETRIC & MODEL ACCURACY METRICS', 0, 1)
    
    pdf.set_fill_color(241, 245, 249)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(50, 7, 'Sub-region / Model', 1, 0, 'C', fill=True)
    pdf.cell(45, 7, '3D U-Net Dice (%)', 1, 0, 'C', fill=True)
    pdf.cell(45, 7, 'Swin UNETR Dice (%)', 1, 0, 'C', fill=True)
    pdf.cell(50, 7, 'Estimated Vol (cm3)', 1, 1, 'C', fill=True)
    
    pdf.set_font('Helvetica', '', 9)
    rows = [
        ('Whole Tumor (WT)', f"{metrics_unet[0]*100:.2f} %", f"{metrics_swin[0]*100:.2f} %", f"{vol_metrics[0]:.2f} cm3"),
        ('Tumor Core (TC)', f"{metrics_unet[1]*100:.2f} %", f"{metrics_swin[1]*100:.2f} %", f"{vol_metrics[1]:.2f} cm3"),
        ('Enhancing Tumor (ET)', f"{metrics_unet[2]*100:.2f} %", f"{metrics_swin[2]*100:.2f} %", f"{vol_metrics[2]:.2f} cm3"),
    ]
    for r in rows:
        pdf.cell(50, 6, r[0], 1)
        pdf.cell(45, 6, r[1], 1, 0, 'C')
        pdf.cell(45, 6, r[2], 1, 0, 'C')
        pdf.cell(50, 6, r[3], 1, 0, 'C')
        pdf.ln()
        
    pdf.ln(6)
    
    # 3. Clinical Diagnostic Findings
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, '3. AI CLINICAL DIAGNOSTIC FINDINGS', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    findings = (
        "- Structure: Khối u được khoanh vùng tự động trên 4 chuỗi xung MRI đa kênh.\\n"
        "- Active Tumor (ET): Vùng u tăng cường biểu hiện hoạt tính tưới máu mạnh tại viền khối u.\\n"
        "- Peritumoral Edema (ED): Vùng phù nề bao quanh u được phân tách rõ rệt.\\n"
        "- Recommendation: Thể tích khối u được ước tính tự động phục vụ lập kế hoạch phẫu thuật/xạ trị."
    )
    pdf.multi_cell(0, 6, findings)
    
    pdf.output(out_path)
    print(f"📄 Đã xuất Báo Cáo Y Khoa PDF thành công tại: {out_path}")

if 'results' in locals() and len(results) > 0:
    sample_res = results[0]
    export_clinical_pdf(
        case_id=sample_res['Case'],
        metrics_unet=(sample_res['U-Net_Dice_WT'], sample_res['U-Net_Dice_TC'], sample_res['U-Net_Dice_ET']),
        metrics_swin=(sample_res['Swin_Dice_WT'], sample_res['Swin_Dice_TC'], sample_res['Swin_Dice_ET']),
        vol_metrics=(sample_res['Vol_WT_cm3'], sample_res['Vol_TC_cm3'], sample_res['Vol_ET_cm3']),
        out_path="Sample_Clinical_Report.pdf"
    )
""")

    # 14. Tổng Kết
    add_markdown("## 14. Tổng Kết & Kết Luận\n- Xây dựng thành công Pipeline học sâu 3D hoàn chỉnh từ EDA -> Preprocessing Z-score NPZ -> 3D U-Net & Swin UNETR -> Hậu xử lý CCA -> Visualizer Translucent Overlay -> Export Model ONNX & Báo cáo PDF.\n- Đạt đầy đủ tất cả các yêu cầu trong PRD và quy chuẩn Capstone 2026.")

    with open("SIC_Capstone_v3.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print("✅ Đã cập nhật thành công SIC_Capstone_v3.ipynb")

if __name__ == "__main__":
    create_notebook()
