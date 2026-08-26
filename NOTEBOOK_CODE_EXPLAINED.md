# SIC Capstone v2 - Giai thich code notebook

Tai lieu nay doc theo dung thu tu 22 code cell trong `SIC_Capstone_v2.ipynb`. Moi muc gom code chinh cua cell va giai thich theo tung dong/nhom dong lien tiep. Cac dong trong code block duoc giu theo y nghia va thu tu cua notebook.

## Tong quan pipeline

1. Cai dat thu vien va import dependency.
2. Phat hien Colab, chon device, seed va duong dan.
3. Doc/preprocess BraTS NIfTI thanh NPZ co image, label, affine va spacing.
4. Chia train/validation/test theo case, tao transform va DataLoader.
5. Tinh Dice, IoU, Precision, Recall, HD95 cho WT, TC va ET.
6. Tao U-Net/Swin UNETR, train, luu checkpoint va chon model bang validation.
7. Ve learning curve, in bang ket qua, luu NIfTI prediction va xuat PDF research report.

## Code dung de lam gi?

Bang nay tra loi muc dich thuc te cua tung code cell. Khi doc phan giai thich dong lenh ben duoi, hay gan moi dong vao muc tieu cua cell tuong ung.

| Cell | Muc dich su dung trong pipeline |
|---|---|
| 1 | Cai dat dung cac thu vien can thiet de notebook co the doc du lieu y khoa, train model, tinh metric va tao report. |
| 2 | Nap cac module vao bo nho de cac cell sau co the dung chung mot moi truong Python. |
| 3 | Cho phep cung mot notebook chay duoc ca tren Google Colab co Google Drive va tren may local. |
| 4 | Co dinh ngau nhien va chon thiet bi tinh toan de ket qua co the lap lai va biet model dang chay o CPU hay GPU. |
| 5 | Tap trung toan bo duong dan, label mapping va hyperparameter vao mot noi de de dieu chinh thuc nghiem. |
| 6 | Chuyen tung case BraTS tu NIfTI sang cache NPZ sach, nhanh va giu du thong tin khong gian cho output. |
| 7 | Tai su dung cache san co, preprocess phan con thieu va dong bo cache sang SSD de giam thoi gian doc du lieu. |
| 8 | Kiem tra nhanh cache dau vao truoc khi train: shape, dtype, label, spacing va affine co hop ly khong. |
| 9 | Tao ba tap train/validation/test khong bi trung case, dong thoi tao tap validation nhanh de theo doi training. |
| 10 | Dong goi cache thanh giao dien Dataset cua PyTorch, bao dam moi sample co cung schema cho MONAI. |
| 11 | Tao augmentation cho train va giu evaluation deterministic de model hoc duoc bien doi nhung metric van cong bang. |
| 12 | Tao cac DataLoader de doc train, fast validation, full validation va held-out test voi cau hinh phu hop. |
| 13 | Dinh nghia cach do chat luong phan doan va cach suy luan full volume bang sliding window. |
| 14 | Tao model theo ten cau hinh, cho phep so sanh U-Net 3D va Swin UNETR trong cung contract. |
| 15 | Dinh nghia ham loss dung trong training va khoi tao noi luu ket qua cua cac run. |
| 16 | Thuc hien training, validation dinh ky, luu checkpoint tot nhat va chi danh gia test sau khi chon model. |
| 17 | Trinh bay loss va Dice theo epoch de nhan biet model dang hoc, qua fit hay khong hoi tu. |
| 18 | In bang ket qua cuoi theo model va split de so sanh validation voi held-out test. |
| 19 | Chon slice dai dien va ve overlay 3 mat phang de con nguoi kiem tra hinh dang prediction. |
| 20 | Chay model da chon tren mot case test, tinh metric, luu mask NIfTI va hien thi ket qua. |
| 21 | Dinh nghia mau PDF research report co thong tin case, volume, metric va gioi han su dung. |
| 22 | Tao report cho case mau sau khi prediction va metric da san sang. |

---

## Code cell 1 - Cai dat thu vien

```python
# Install into the active notebook kernel. Version ranges avoid unreviewed major upgrades.
%pip install -q "monai>=1.3,<2.0" "nibabel>=5.2,<6" "fpdf2>=2.7,<3" \
  "matplotlib>=3.7,<4" "tqdm>=4.66,<5" "scikit-learn>=1.3,<2" "scipy>=1.11,<2"
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| `# Install into the active notebook kernel...` | Ghi chu rang thu vien duoc cai vao kernel dang chay va version duoc gioi han de tranh thay doi API bat ngo. |
| `%pip install -q "monai>=1.3,<2.0" ... "scipy>=1.11,<2"` | Cai cac thu vien de doc NIfTI, train model, tinh metric, hien progress, ve hinh va tao PDF. `-q` chi giam log; dau `\\` noi cau lenh tren nhieu dong. |

## Code cell 2 - Import

```python
import inspect
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

import monai.transforms as mt
from monai.data import DataLoader as MonaiDataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR, UNet
from monai.utils import set_determinism
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| `import inspect` ... `from zipfile import BadZipFile` | Nap cong cu phan tich signature, xu ly file, ngau nhien, path, typing va bat loi file NPZ/ZIP. |
| `import matplotlib.pyplot as plt` ... `from tqdm.notebook import tqdm` | Nap cong cu doc/luu anh, mang so, deep learning, phep toan mask, chia dataset va progress bar. |
| `import monai.transforms as mt` ... `from monai.utils import set_determinism` | Nap cac thanh phan MONAI cho augmentation, DataLoader, suy luan sliding window, loss, model va reproducibility. |

## Code cell 3 - Colab mount

```python
IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive", force_remount=False)
    if not Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Google Drive mount failed: /content/drive/MyDrive is unavailable")
    print("Google Drive is ready")
except ImportError:
    print("Running outside Google Colab")
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| `IN_COLAB = False` | Dat mac dinh la dang chay tren may local. |
| `try:` va `from google.colab import drive` | Thu nhan dien moi truong Colab; tren may local import nay se that bai va khong lam notebook dung. |
| `IN_COLAB = True` | Danh dau notebook dang chay trong Colab de cac cell sau chon duong dan phu hop. |
| `if not Path("/content/drive/MyDrive").exists():` ... `drive.mount(...)` | Kiem tra Drive da mount chua; neu chua thi mount vao Colab. |
| `raise RuntimeError(...)` | Dung chuong trinh voi thong bao ro rang neu mount khong thanh cong. |
| `print("Google Drive is ready")` | Xac nhan Drive san sang. |
| `except ImportError:` ... `print("Running outside Google Colab")` | Xu ly truong hop local va bo qua logic mount. |

## Code cell 4 - Seed va device

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_determinism(seed=SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / (1024 ** 3):.2f} GB")
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| `SEED = 42` va cac lenh `random.seed(...)`, `np.random.seed(...)`, `torch.manual_seed(...)` | Co dinh cac so ngau nhien de chia data, augmentation va khoi tao model co the lap lai. |
| `torch.cuda.manual_seed_all(SEED)` | Dat seed cho moi GPU neu may co CUDA. |
| `set_determinism(seed=SEED)` | Yeu cau MONAI transforms/DataLoader su dung seed da dat. |
| `torch.backends.cudnn.benchmark = False` va `.deterministic = True` | Uu tien ket qua lap lai thay vi tu dong chon thuat toan nhanh nhat. |
| `device = torch.device("cuda" if ... else "cpu")` | Chon GPU neu co, neu khong thi dung CPU de model va tensor biet noi tinh toan. |
| `torch.cuda.get_device_properties(0)` va cac lenh `print` | In ten GPU va VRAM de kiem tra kha nang chay model 3D. |

## Code cell 5 - Duong dan va cau hinh

```python
raw_dir = Path("/content/drive/MyDrive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
npz_dir = Path("/content/drive/MyDrive/BraTS2023/processed_npz")
data_root = raw_dir if raw_dir.exists() else None
drive_npz_dir = npz_dir if npz_dir.exists() and len(list(npz_dir.glob("*.npz"))) > 0 else None
local_npz_dir = Path("/content/processed_npz") if IN_COLAB else npz_dir
checkpoints_dir = Path("/content/drive/MyDrive/BraTS2023/checkpoints") if IN_COLAB else Path("checkpoints")
reports_dir = Path("/content/drive/MyDrive/BraTS2023/reports") if IN_COLAB else Path("reports")
predictions_dir = Path("/content/drive/MyDrive/BraTS2023/predictions") if IN_COLAB else Path("predictions")
checkpoints_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)
predictions_dir.mkdir(parents=True, exist_ok=True)

PREPROCESS_VERSION = "brats_npz_v3_affine_spacing_strict_labels"
FORCE_REBUILD_CACHE = False
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
ALLOWED_RAW_LABELS = set(LABEL_MAP)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
VAL_SUBSET_SIZE = 20
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 1
NUM_SAMPLES = 1
NUM_WORKERS = 2
STEPS_PER_EPOCH = 150
MAX_EPOCHS = 30
FAST_VAL_INTERVAL = 2
FULL_VAL_INTERVAL = 10
SW_BATCH_SIZE = 1
SW_OVERLAP = 0.5
LR = 2e-4
WEIGHT_DECAY = 1e-4
RUN_SWIN_UNETR = False
MODELS_TO_TRAIN = ["unet3d"] + (["swin_unetr"] if RUN_SWIN_UNETR else [])
```

### Giai thich tung dong: duong dan va cau hinh

Voi moi dong ben duoi, cot trai la code, cot phai la viec dong do dung de lam trong chuong trinh.

Quy uoc cua tai lieu: phan giai thich luon dat **doan code nguyen van** o cot trai, sau do moi noi doan code do dung de lam gi. Vi du, thay vi ghi `Dong 18-33`, tai lieu ghi truc tiep cac dong nhu `VAL_SPLIT = 0.15` va `TEST_SPLIT = 0.15`.

| Dong code | Dong nay dung de lam gi? |
|---|---|
| `raw_dir = Path("...TrainingData")` | Luu duong dan den thu muc chua cac file MRI NIfTI goc cua BraTS. Ham preprocess se tim cac case trong thu muc nay. |
| `npz_dir = Path("...processed_npz")` | Luu duong dan den noi chua cache NPZ da xu ly, de khong phai doc va chuan hoa lai NIfTI moi lan chay. |
| `data_root = raw_dir if raw_dir.exists() else None` | Neu thu muc raw ton tai thi dung no; neu khong thi dat `None` de biet may khong co raw data. |
| `drive_npz_dir = npz_dir if ... else None` | Chi danh dau cache Drive la san sang khi thu muc ton tai va co it nhat mot file `.npz`. |
| `local_npz_dir = Path("/content/processed_npz") if IN_COLAB else npz_dir` | Tren Colab dung SSD local de doc nhanh; tren may local dung truc tiep thu muc NPZ da cau hinh. |
| `checkpoints_dir = ...` | Chi dinh noi luu trong so model tot nhat sau training. |
| `reports_dir = ...` | Chi dinh noi luu cac bao cao PDF nghien cuu. |
| `predictions_dir = ...` | Chi dinh noi luu mask phan doan NIfTI duoc model tao ra. |
| `checkpoints_dir.mkdir(...)` | Tao thu muc checkpoint neu chua co; `exist_ok=True` cho phep chay lai ma khong loi neu thu muc da ton tai. |
| `reports_dir.mkdir(...)` | Tao thu muc report truoc khi ghi PDF. |
| `predictions_dir.mkdir(...)` | Tao thu muc prediction truoc khi ghi NIfTI. |
| `PREPROCESS_VERSION = "..."` | Dat ten phien ban cua quy tac tien xu ly; checkpoint khac version se bi tu choi de tranh dung sai cache. |
| `FORCE_REBUILD_CACHE = False` | Khong preprocess lai cache da co. Dat `True` khi muon tao lai toan bo NPZ tu NIfTI. |
| `MODALITIES = ("t1n", "t1c", "t2w", "t2f")` | Xac dinh 4 loai anh MRI se doc cho moi case va thu tu kenh dau vao model. |
| `LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}` | Quy doi nhan raw sang nhan model; label raw 4 duoc gop vao lop 3. |
| `ALLOWED_RAW_LABELS = set(LABEL_MAP)` | Tao tap `{0,1,2,3,4}` de kiem tra va phat hien nhan la trong segmentation. |
| `VAL_SPLIT = 0.15` | Danh rieng **15% so case** lam tap validation. Tap nay dung de theo doi chat luong va chon checkpoint; no khong duoc dung de cap nhat trong so model. Vi du co 100 case thi khoang 15 case vao validation. |
| `TEST_SPLIT = 0.15` | Danh rieng 15% case lam held-out test. Tap nay chi duoc mo ra de bao cao cuoi sau khi model da duoc chon. |
| `VAL_SUBSET_SIZE = 20` | Gioi han fast validation toi da 20 case de kiem tra nhanh trong luc train. |
| `PATCH_SIZE = (96, 96, 96)` | Moi mau train la mot khoi 3D co 96 voxel moi truc, tong `884,736` voxel. Day khong phai la so case. |
| `BATCH_SIZE = 1` | Moi lan model cap nhat trong so chi xu ly **1 patch**. Vi du image batch co shape `(1, 4, 96, 96, 96)`: `1` la so patch, `4` la so modality MRI. |
| `NUM_SAMPLES = 1` | Moi case duoc transform sinh 1 patch trong mot lan goi. Tham so nay khac `BATCH_SIZE`: no quyet dinh so patch duoc tao ra, khong phai so patch trong batch. |
| `NUM_WORKERS = 2` | Tao 2 worker phu de doc du lieu song song. Hai worker khong co nghia batch size bang 2. |
| `STEPS_PER_EPOCH = 150` | Moi epoch toi da xu ly 150 batch va cap nhat model 150 lan; neu DataLoader co it batch hon thi dung som hon. |
| `MAX_EPOCHS = 30` | Lap qua toi da 30 epoch; mot epoch la mot luot model di qua cac batch duoc phep cua tap train. |
| `FAST_VAL_INTERVAL = 2` | Cu moi 2 epoch chay fast validation mot lan de theo doi nhanh. |
| `FULL_VAL_INTERVAL = 10` | Cu moi 10 epoch chay full validation; ket qua nay duoc dung de chon checkpoint tot nhat. |
| `SW_BATCH_SIZE = 1` | Khi suy luan full volume, moi lan sliding window xu ly 1 cua so 3D. Day la batch cho inference, khac batch train. |
| `SW_OVERLAP = 0.5` | Hai cua so lien tiep chong len nhau 50%; cac vung lap duoc tron de giam bien dut giua patch. |
| `LR = 2e-4` | Learning rate bang `0.0002`, quyet dinh muc do lon cua moi lan optimizer dieu chinh trong so. |
| `WEIGHT_DECAY = 1e-4` | Muc regularization `0.0001`, giup han che trong so tang qua lon va giam overfitting. |
| `RUN_SWIN_UNETR = False` | Tat viec train Swin UNETR trong cau hinh hien tai; chi train U-Net 3D. |
| `MODELS_TO_TRAIN = ["unet3d"] + (["swin_unetr"] if RUN_SWIN_UNETR else [])` | Tao danh sach model thuc su se chay; neu bat Swin thi danh sach co them `swin_unetr`. |

### Vi du cu the ve cac tham so

- `PATCH_SIZE = (96, 96, 96)`: moi lan crop se lay mot khoi 3D co `96 x 96 x 96 = 884,736` voxel. Day la kich thuoc khong gian cua mot patch, khong phai so luong mau.
- `BATCH_SIZE = 1`: moi buoc cap nhat model chi nhan **1 patch**. Neu image co 4 modality thi mot batch image thuong co shape `(1, 4, 96, 96, 96)`; batch label co shape gan `(1, 1, 96, 96, 96)`. So `1` dau tien la so patch trong batch, `4` la so kenh MRI.
- Vi du, neu DataLoader lay 8 case va moi case sinh 1 patch, `BATCH_SIZE=1` tao 8 batch: batch 1 cap nhat model xong moi den batch 2. No khong co nghia la chi train tren 1 case duy nhat; nhieu batch van duoc xu ly trong mot epoch.
- Ly do dung batch 1 la patch 3D chua gan 3.5 trieu gia tri float32 cho image 4 kenh, chua tinh activation cua U-Net; batch lon hon de nhanh het VRAM. Doi lai gradient nhieu noise hon va toc do co the cham hon.
- `NUM_SAMPLES = 1`: moi case trong mot lan goi transform sinh 1 patch. Neu dat `2`, mot case co the tra ve 2 patch va DataLoader se collate thanh nhieu sample; day khac voi `BATCH_SIZE`.
- `NUM_WORKERS = 2`: co 2 tien trinh phu doc/preprocess sample song song. No khong lam batch co 2 sample.
- `STEPS_PER_EPOCH = 150`: moi epoch toi da cap nhat trong 150 batch. Neu `len(train_loader)` nho hon 150 thi chi chay het so batch thuc te.
- `LR = 2e-4`: moi lan optimizer step dieu chinh trong so voi buoc co do lon khoang `0.0002`, sau khi gradient da duoc tinh.

## Code cell 6 - Doc cache va preprocess case

```python
CACHE_ERRORS = (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile)

def load_validated_cache(npz_path: Path) -> Dict[str, np.ndarray]:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        img_key = next((k for k in ("image", "img", "data", "mri") if k in data), None)
        if img_key is None:
            if len(data.files) > 0:
                img_key = data.files[0]
            else:
                raise ValueError(f"No array found in {npz_path}")
        image = np.asarray(data[img_key])
        if image.ndim == 4 and image.shape[-1] in (1, 4) and image.shape[0] not in (1, 4):
            image = np.moveaxis(image, -1, 0)
        elif image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        lbl_key = next((k for k in ("label", "seg", "mask", "target", "labels") if k in data), None)
        if lbl_key is not None:
            label = np.asarray(data[lbl_key])
            if label.ndim == 4 and label.shape[-1] == 1:
                label = np.squeeze(label, axis=-1)
            if label.ndim == 3:
                label = np.expand_dims(label, axis=0)
            if np.any(label == 4):
                label = label.copy()
                label[label == 4] = 3
        else:
            label = np.zeros((1,) + image.shape[1:], dtype=np.uint8)
        if "affine" in data and hasattr(data["affine"], "shape") and data["affine"].shape == (4, 4):
            affine = np.asarray(data["affine"], dtype=np.float64)
        else:
            affine = np.eye(4, dtype=np.float64)
        if "spacing" in data and hasattr(data["spacing"], "__len__") and len(data["spacing"]) == 3:
            spacing = np.asarray(data["spacing"], dtype=np.float32)
        else:
            spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return {"image": image.astype(np.float32, copy=False), "label": label.astype(np.uint8, copy=False), "affine": affine, "spacing": spacing}
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1 | Gom cac loi co the xuat hien khi NPZ hong, sai schema hoac khong doc duoc. |
| 3-5 | Chuan hoa path, mo NPZ cho phep pickle, va giu file mo trong context manager. |
| 6-11 | Tim ten image theo nhieu schema; neu khong co ten chuan thi dung array dau tien, neu rong thi bao loi. |
| 12-16 | Chuan hoa image ve dang channel-first `(C,H,W,D)`; mo rong kenh cho volume 3D. |
| 17-25 | Tim label theo nhieu ten, xu ly singleton dimension, them channel va remap label legacy 4 -> 3. |
| 26-28 | Neu cache khong co label thi tao mask rong, phu hop voi cache suy luan. |
| 29-35 | Doc affine hop le, neu thieu thi dung ma tran don vi; lam tuong tu cho spacing voi gia tri mac dinh 1 mm. |
| 36 | Tra ve dtype on dinh: image float32, label uint8, affine float64, spacing float32. |

Cell nay con co hai ham trong notebook: `cache_is_current` kiem tra file ton tai, khong rong va `np.load` doc duoc; `preprocess_case` doc 4 modality NIfTI, kiem tra shape/affine/spacing/non-finite, clip percentile 0.5-99.5, z-score trong brain mask, doc segmentation, tu choi label la, va ghi NPZ tam theo cach atomic. `label[None, ...]` tao channel label; `temp_path.replace(out_path)` tranh de lai file dang ghi neu qua trinh bi gian doan.

## Code cell 7 - Chuan bi dataset

```python
def prepare_dataset(drive_dir: Path, local_dir: Path, raw_dir: Path) -> List[str]:
    drive_dir = Path(drive_dir)
    local_dir = Path(local_dir)
    raw_dir = Path(raw_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    existing_drive_npz = {f.stem: f for f in drive_dir.glob("*.npz") if not f.name.startswith(".") and f.stat().st_size > 0}
    existing_local_npz = {f.stem: f for f in local_dir.glob("*.npz") if not f.name.startswith(".") and f.stat().st_size > 0}
    raw_cases = sorted(directory.name for directory in raw_dir.iterdir() if directory.is_dir() and (directory.name.startswith("BraTS") or not directory.name.startswith("."))) if raw_dir.exists() else []
    reusable_cases = [cid for cid in raw_cases if cid in existing_drive_npz or cid in existing_local_npz] if raw_cases else list(existing_drive_npz.keys() or existing_local_npz.keys())
    cases_to_build = raw_cases if FORCE_REBUILD_CACHE else [cid for cid in raw_cases if cid not in reusable_cases]
    if cases_to_build:
        for case_id in tqdm(cases_to_build, desc="Preprocess missing cases"):
            preprocess_case(case_id, raw_dir, drive_dir, force=FORCE_REBUILD_CACHE)
    drive_files = [f for f in sorted(drive_dir.glob("*.npz")) if not f.name.startswith(".") and f.stat().st_size > 0]
    if not drive_files and existing_local_npz:
        drive_files = [f for f in sorted(local_dir.glob("*.npz")) if not f.name.startswith(".") and f.stat().st_size > 0]
    if not drive_files:
        raise RuntimeError(f"No NPZ files found in {drive_dir} or {local_dir}, and no raw data available in {raw_dir}.")
    if local_dir.resolve() != drive_dir.resolve() and existing_drive_npz:
        for source in drive_files:
            destination = local_dir / source.name
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                shutil.copy2(source, destination)
        target_dir = local_dir
    else:
        target_dir = drive_dir if drive_files else local_dir
    case_ids = sorted(f.stem for f in target_dir.glob("*.npz") if not f.name.startswith(".") and f.stat().st_size > 0)
    print(f"Loaded {len(case_ids)} preprocessed cases from {target_dir}")
    return case_ids
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-6 | Chuan hoa path, tao thu muc, lap danh sach cache Drive/local va loc file an/file rong. |
| 7-9 | Lay danh sach raw case; xac dinh case da tai su dung va case can build, phu thuoc `FORCE_REBUILD_CACHE`. |
| 10-12 | Preprocess tung case thieu co progress bar. |
| 13-17 | Thu thap NPZ, fallback sang local, va dung loi ro rang neu khong co du lieu. |
| 18-24 | Dong bo file sang SSD local khi hai path khac nhau; chi copy file thieu hoac sai kich thuoc. |
| 25-27 | Tao danh sach case ID tu cache hop le, in thong tin va return. |

## Code cell 8 - Nap va kiem tra mau

```python
all_cases = prepare_dataset(drive_npz_dir, local_npz_dir, data_root)
print(f"Valid cases: {len(all_cases)}")
sample_path = local_npz_dir / f"{all_cases[0]}.npz"
if not sample_path.exists() and drive_npz_dir.exists():
    sample_path = drive_npz_dir / f"{all_cases[0]}.npz"
start = time.time()
sample_data = load_validated_cache(sample_path)
read_duration = time.time() - start
sample_img = sample_data["image"]
sample_lbl = sample_data["label"]
sample_spacing = sample_data["spacing"]
sample_affine = sample_data["affine"]
print(f"Read time: {read_duration:.5f}s")
print(f"Image: {sample_img.shape} {sample_img.dtype}")
print(f"Label: {sample_lbl.shape} labels={np.unique(sample_lbl).tolist()}")
print(f"Spacing: {sample_spacing.tolist()} | affine determinant={np.linalg.det(sample_affine[:3, :3]):.4f}")
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| `all_cases = prepare_dataset(...)` | Chay pipeline tim/preprocess cache va tao danh sach case ID de train. |
| `sample_path = local_npz_dir / f"{all_cases[0]}.npz"` va `if not sample_path.exists(): ...` | Chon file mau o local; neu khong co thi thu tim file tren Drive. |
| `start = time.time()` va `read_duration = time.time() - start` | Do thoi gian doc cache de biet toc do I/O. |
| `sample_img = ...`, `sample_lbl = ...`, `sample_spacing = ...`, `sample_affine = ...` | Tach cac thanh phan mau de kiem tra schema dau ra cua cache. |
| `print(...)` va `np.linalg.det(sample_affine[:3, :3])` | In shape, dtype, label, spacing va determinant affine de phat hien du lieu loi truoc khi train. |

## Code cell 9 - Chia train/validation/test

```python
if len(all_cases) < 20:
    raise RuntimeError(f"At least 20 labeled cases are required, found {len(all_cases)}")
train_val_cases, test_cases = train_test_split(all_cases, test_size=TEST_SPLIT, random_state=SEED, shuffle=True)
relative_val_size = VAL_SPLIT / (1.0 - TEST_SPLIT)
train_cases, val_cases = train_test_split(train_val_cases, test_size=relative_val_size, random_state=SEED, shuffle=True)
rng = np.random.default_rng(SEED)
fast_size = min(VAL_SUBSET_SIZE, len(val_cases))
val_cases_fast = sorted(rng.choice(val_cases, size=fast_size, replace=False).tolist())
assert set(train_cases).isdisjoint(val_cases)
assert set(train_cases).isdisjoint(test_cases)
assert set(val_cases).isdisjoint(test_cases)
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| `if len(all_cases) < 20: raise RuntimeError(...)` | Chan dataset qua nho vi ba tap train/validation/test se khong du dai dien. |
| `train_val_cases, test_cases = train_test_split(..., test_size=TEST_SPLIT, ...)` | Tach test truoc de test khong bi dung trong train hay validation. |
| `relative_val_size = ...` va `train_cases, val_cases = train_test_split(...)` | Tinh ti le validation tren phan train-val con lai va tao tap validation. |
| `rng = np.random.default_rng(SEED)` ... `val_cases_fast = ...` | Lay ngau nhien toi da 20 case validation de kiem tra nhanh, nhung lap lai duoc nho seed. |
| `assert set(...).isdisjoint(...)` | Kiem tra train, validation va test khong trung case; day la split theo patient/case. |

## Code cell 10 - Dataset class

```python
class BraTSDataset3D(Dataset):
    def __init__(self, npz_dir, case_list, transforms=None):
        self.npz_dir = Path(npz_dir)
        self.case_list = list(case_list)
        self.transforms = transforms

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        case_id = self.case_list[index]
        npz_path = self.npz_dir / f"{case_id}.npz"
        if not npz_path.exists() and drive_npz_dir.exists():
            fallback_path = drive_npz_dir / f"{case_id}.npz"
            if fallback_path.exists():
                npz_path = fallback_path
        try:
            cached = load_validated_cache(npz_path)
        except Exception as exc:
            raise RuntimeError(f"Invalid or unreadable cache: {npz_path}") from exc
        sample = {"image": cached["image"], "label": cached["label"].astype(np.int64), "case_id": case_id, "affine": cached["affine"], "spacing": cached["spacing"]}
        return self.transforms(sample) if self.transforms else sample
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| `class BraTSDataset3D(Dataset)` va `__init__(...)` | Tao lop dataset PyTorch, luu noi chua NPZ, danh sach case va transform. |
| `__len__` va `__getitem__` | Cho DataLoader biet co bao nhieu mau va cach lay mot mau theo index. |
| `case_id = self.case_list[index]` ... `fallback_path = ...` | Xac dinh case can doc, uu tien cache local va fallback sang Drive. |
| `cached = load_validated_cache(npz_path)` va `sample = {...}` | Doc cache da validate va dong goi thanh dictionary co key image/label/affine/spacing. |
| `return self.transforms(sample) if ...` | Ap dung augmentation/tensor conversion neu co; label int64 phu hop CrossEntropy. |

## Code cell 11 - Transforms

```python
train_transforms = mt.Compose([
    mt.RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE, pos=3.0, neg=1.0, num_samples=NUM_SAMPLES, allow_smaller=False),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    mt.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    mt.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    mt.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    mt.EnsureTyped(keys=["image", "label"], track_meta=False),
])
eval_transforms = mt.Compose([mt.EnsureTyped(keys=["image", "label"], track_meta=False)])
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| `train_transforms = mt.Compose([...])` | Tao chuoi bien doi ngau nhien chi dung cho train de tang da dang du lieu. |
| `mt.RandCropByPosNegLabeld(... spatial_size=PATCH_SIZE, pos=3.0, neg=1.0 ...)` | Cat patch 96^3 va uu tien patch co tumor theo ti le duong/am 3:1. |
| `mt.RandFlipd(...)` va `mt.RandRotate90d(...)` | Lat/ xoay dong thoi image va label, tranh lam mask lech khoi anh. |
| `mt.RandScaleIntensityd(...)` va `mt.RandShiftIntensityd(...)` | Mo phong thay doi cuong do MRI; chi tac dong len image, khong doi label. |
| `mt.EnsureTyped(...)` va `eval_transforms = ...` | Chuyen du lieu sang tensor; evaluation khong co augmentation de ket qua cong bang. |

## Code cell 12 - DataLoader

```python
train_dataset = BraTSDataset3D(local_npz_dir, train_cases, transforms=train_transforms)
val_dataset_fast = BraTSDataset3D(local_npz_dir, val_cases_fast, transforms=eval_transforms)
val_dataset_full = BraTSDataset3D(local_npz_dir, val_cases, transforms=eval_transforms)
test_dataset = BraTSDataset3D(local_npz_dir, test_cases, transforms=eval_transforms)
loader_kwargs = {"num_workers": NUM_WORKERS, "pin_memory": torch.cuda.is_available()}
if NUM_WORKERS > 0:
    loader_kwargs["prefetch_factor"] = 1
train_loader = MonaiDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, persistent_workers=NUM_WORKERS > 0, **loader_kwargs)
val_loader_fast = MonaiDataLoader(val_dataset_fast, batch_size=1, shuffle=False, persistent_workers=False, **loader_kwargs)
val_loader_full = MonaiDataLoader(val_dataset_full, batch_size=1, shuffle=False, persistent_workers=False, **loader_kwargs)
test_loader = MonaiDataLoader(test_dataset, batch_size=1, shuffle=False, persistent_workers=False, **loader_kwargs)
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-4 | Tao dataset cho train, fast validation, full validation va held-out test. |
| 5-7 | Cau hinh worker, pin memory khi co GPU va prefetch nho de giam RAM. |
| 8-11 | Train shuffle; validation/test khong shuffle va batch 1 de danh gia tung case. `persistent_workers` chi bat cho train. |

## Code cell 13 - Metrics va inference

```python
REGION_BUILDERS = {
    "wt": lambda array: array > 0,
    "tc": lambda array: np.isin(array, [1, 3]),
    "et": lambda array: array == 3,
}

def _binary_metrics(prediction, target, spacing, include_surface=True):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    intersection = int(np.logical_and(prediction, target).sum())
    union = pred_count + target_count - intersection
    if pred_count == 0 and target_count == 0:
        dice = iou = precision = recall = 1.0
        hd95 = 0.0
    else:
        dice = 2.0 * intersection / max(pred_count + target_count, 1)
        iou = intersection / max(union, 1)
        precision = intersection / pred_count if pred_count else 0.0
        recall = intersection / target_count if target_count else 0.0
        hd95 = float("nan")
        if include_surface:
            if pred_count == 0 or target_count == 0:
                hd95 = float(np.linalg.norm(np.asarray(prediction.shape) * np.asarray(spacing)))
            else:
                structure = np.ones((3, 3, 3), dtype=bool)
                pred_surface = np.logical_xor(prediction, binary_erosion(prediction, structure=structure, border_value=0))
                target_surface = np.logical_xor(target, binary_erosion(target, structure=structure, border_value=0))
                distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
                distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
                distances = np.concatenate([distance_to_target[pred_surface], distance_to_pred[target_surface]])
                hd95 = float(np.percentile(distances, 95))
    voxel_volume_cm3 = float(np.prod(spacing) / 1000.0)
    result = {"dice": float(dice), "iou": float(iou), "precision": float(precision), "recall": float(recall), "volume_cm3": float(pred_count * voxel_volume_cm3)}
    if include_surface:
        result["hd95"] = float(hd95)
    return result
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-5 | Dinh nghia WT = label > 0, TC = 1/3, ET = 3; tach logic vung khoi u khoi ham metric. |
| 7-13 | Chuyen mask sang bool, dem voxel prediction/target, tinh intersection va union. |
| 14-17 | Truong hop ca hai rong dat metric bang 1; nguoc lai tinh Dice, IoU, precision, recall va khoi tao HD95. |
| 18-29 | HD95: neu mot mask rong dung duong cheo volume; neu khong thi lay surface bang erosion, tinh distance transform theo spacing, lay percentile 95. |
| 30-34 | Doi voxel sang cm3, dong goi metric va chi them HD95 khi duoc yeu cau. |

Cell 13 con co `compute_case_metrics`: doi Tensor ve NumPy, squeeze channel/batch, kiem tra shape, lap qua ba region, dat key co hau to region va tinh cac mean. `predict_volume` dung `sliding_window_inference` voi `PATCH_SIZE`, overlap 0.5 va gaussian blending, sau do `argmax` kenh lop. `evaluate_model` dat model eval, duyet loader, suy luan tung case, lay spacing, tinh metric, roi dung `nanmean` de tao summary.

## Code cell 14 - Model factory

```python
def build_model(model_name):
    if model_name == "unet3d":
        return UNet(spatial_dims=3, in_channels=4, out_channels=4, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="instance", dropout=0.2)
    if model_name == "swin_unetr":
        kwargs = {"in_channels": 4, "out_channels": 4, "feature_size": 24, "use_checkpoint": True, "spatial_dims": 3}
        if "img_size" in inspect.signature(SwinUNETR).parameters:
            kwargs["img_size"] = PATCH_SIZE
        return SwinUNETR(**kwargs)
    raise ValueError(f"Unsupported model: {model_name}")

for model_name in MODELS_TO_TRAIN:
    parameter_count = sum(parameter.numel() for parameter in build_model(model_name).parameters())
    print(f"{model_name}: {parameter_count:,} parameters")
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-3 | Factory tra ve 3D U-Net voi 4 kenh vao va 4 lop ra; cac stride va channels tao encoder-decoder. |
| 4-8 | Nhanh Swin UNETR; dung checkpoint de giam memory; kiem tra signature de tuong thich nhieu phien ban MONAI. |
| 9 | Tu choi ten model khong hop le thay vi am tham tra ve sai. |
| 11-13 | Khoi tao tung model va dem tong so parameter de kiem tra quy mo. |

## Code cell 15 - Loss

```python
criterion = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False,
    lambda_dice=1.0,
    lambda_ce=1.0,
)
MODEL_RUNS = {}
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-7 | Ket hop Dice loss va Cross Entropy; label duoc one-hot, logits duoc softmax; bo qua background de tap trung tumor. |
| 9 | Dictionary luu history, checkpoint va ket qua cua moi model. |

## Code cell 16 - Training va chon checkpoint

```python
def create_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

def save_checkpoint_atomic(state, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-5 | Tao AMP GradScaler theo API moi, fallback API cu khi MONAI/PyTorch moi truong cu. |
| 8-11 | Luu checkpoint vao file tam roi rename atomic; checkpoint cu khong bi thay the nua chung. |

Ham `train_one_model` trong cell nay thuc hien cac buoc sau theo dung thu tu:

| Buoc/dong logic | Giai thich chi tiet |
|---|---|
| Khoi tao model, AdamW, scaler | Dua model len `device`, dung learning rate/weight decay da cau hinh. |
| Tao `checkpoint_path`, `best_full_val_dice`, `history` | Theo doi loss, fast validation, full validation va epoch tot nhat. |
| Vong `for epoch` | Moi epoch dat model train, reset loss/count va tao progress bar. |
| Vong batch | Dung toi da `STEPS_PER_EPOCH`; chuyen image/label sang device, xoa gradient. |
| `torch.autocast` | Bat float16 chi tren CUDA; CPU giu precision binh thuong. |
| forward, loss, backward, optimizer step | Tinh logits, DiceCE, scale gradient, cap nhat tham so va scaler. |
| Fast validation | Chay theo interval hoac epoch cuoi; chi dung fast set de theo doi, khong chon checkpoint. |
| Full validation | Chay theo interval/epoch cuoi; dung `dice_mean` lam tieu chi chon checkpoint. |
| Save checkpoint | Luu state model/optimizer/scaler, epoch, summary, history, preprocessing version va patch size. |
| Nap checkpoint | Bao loi neu khong co checkpoint hoac version cache/checkpoint khac nhau; nap lai model tot nhat. |
| Final validation/test | Danh gia voi surface metrics; test chi chay sau khi model da duoc chon bang validation. |
| Cleanup va return | Dua model ve CPU, giai phong CUDA, tra ve ket qua va epoch tot nhat. |

Phan cuoi lap qua `MODELS_TO_TRAIN`, dien `MODEL_RUNS`, chon `best_model_name` theo validation Dice, nap checkpoint tuong ung, tao `best_model` va dat eval. Diem quan trong: test metric khong tham gia model selection.

### Mot batch di qua training nhu the nao?

Gia su `BATCH_SIZE=1`, DataLoader tra ve mot batch image co shape `(1, 4, 96, 96, 96)`. Model nhan batch nay va tao logits co shape `(1, 4, 96, 96, 96)`: 4 kenh output la xac suat/diem cho background, label 1, label 2 va label 3 tai moi voxel. Label muc tieu co shape `(1, 1, 96, 96, 96)` va chua so lop nguyen.

1. `optimizer.zero_grad(...)` xoa gradient cua batch truoc.
2. `outputs = model(images)` chay forward de tao du doan.
3. `loss = criterion(outputs, labels)` so sanh du doan voi mask that.
4. `loss.backward()` tinh gradient cua loss theo tung parameter.
5. `optimizer.step()` cap nhat trong so; day moi la luc model hoc tu batch nay.
6. Batch tiep theo lai lap quy trinh tren. Vi vay batch size 1 khong bo qua viec hoc, ma chi lam moi lan cap nhat dua tren mot patch.

Notebook hien khong dung gradient accumulation. Neu muon mo phong batch lon hon tren GPU nho, co the tich luy gradient cua nhieu batch truoc khi goi `optimizer.step`, nhung do la thay doi thuat toan va can sua code rieng.

## Code cell 17 - Ve learning curves

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
colors = {"unet3d": "#1f77b4", "swin_unetr": "#2ca02c"}
for model_name, run in MODEL_RUNS.items():
    history = run["history"]
    color = colors.get(model_name)
    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label=f"{model_name} train loss", color=color)
    axes[1].plot(history["fast_val_epochs"], history["fast_val_dice"], marker="o", linestyle="--", alpha=0.7, label=f"{model_name} fast val", color=color)
    axes[1].plot(history["full_val_epochs"], history["full_val_dice"], marker="s", linewidth=2, label=f"{model_name} full val", color=color)
axes[0].set_title("Training loss")
axes[1].set_title("Validation Dice")
axes[1].set_ylim(0, 1)
for axis in axes:
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
plt.tight_layout()
plt.show()
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-2 | Tao hai subplot va bang mau on dinh cho hai model. |
| 3-7 | Ve train loss, fast validation va full validation; marker/linestyle giup phan biet hai loai validation. |
| 8-10 | Dat tieu de, gioi han Dice 0-1, bat grid/legend. |
| 11-12 | Can chinh layout va hien thi hinh. |

## Code cell 18 - Bang tong ket

```python
print("=" * 116)
print(f"{'MODEL':<14} | {'SPLIT':<10} | {'DICE':>7} | {'WT':>7} | {'TC':>7} | {'ET':>7} | {'IOU':>7} | {'HD95 MM':>9} | {'BEST EPOCH':>10}")
print("=" * 116)
for model_name, run in MODEL_RUNS.items():
    for split_name in ("val", "test"):
        summary = run[f"{split_name}_summary"]
        print(f"{model_name:<14} | {split_name:<10} | {summary['dice_mean']:>7.4f} | {summary['dice_wt']:>7.4f} | {summary['dice_tc']:>7.4f} | {summary['dice_et']:>7.4f} | {summary['iou_mean']:>7.4f} | {summary['hd95_mean']:>9.3f} | {run['best_epoch']:>10}")
print("=" * 116)
print("Model selection used validation only; test metrics are final reporting metrics.")
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-3 | In duong ke va header co do rong co dinh de bang can cot. |
| 4-7 | Duyet model va hai split; lay summary tu dictionary va format Dice/HD95. |
| 8-9 | Ket thuc bang va nhac lai quy tac validation-only. |

## Code cell 19 - Truc quan hoa 3 mat phang

```python
def visualize_slices(image_4d, prediction, case_id, model_name, ground_truth=None):
    flair = image_4d[3]
    pred = np.asarray(prediction).squeeze()
    gt = None if ground_truth is None else np.asarray(ground_truth).squeeze()
    selection_mask = gt if gt is not None and np.any(gt > 0) else pred
    z_sum = np.sum(selection_mask > 0, axis=(0, 1))
    y_sum = np.sum(selection_mask > 0, axis=(0, 2))
    x_sum = np.sum(selection_mask > 0, axis=(1, 2))
    z_idx = int(np.argmax(z_sum)) if z_sum.max() > 0 else flair.shape[2] // 2
    y_idx = int(np.argmax(y_sum)) if y_sum.max() > 0 else flair.shape[1] // 2
    x_idx = int(np.argmax(x_sum)) if x_sum.max() > 0 else flair.shape[0] // 2
    from matplotlib.colors import ListedColormap
    segmentation_cmap = ListedColormap(["none", "#e41a1c", "#4daf4a", "#ff7f00"])
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-4 | Nhan image 4D, chon FLAIR kenh 3, ep prediction/ground truth ve 3D. |
| 5 | Uu tien vi tri co ground truth tumor; neu khong co thi dung prediction. |
| 6-11 | Cong mask theo tung truc de tim slice co nhieu tumor nhat; fallback ve slice giua neu mask rong. |
| 12-14 | Tao colormap trong suot cho background va mau rieng cho cac label 1/2/3. |

Phan con lai cua ham tao ba tuple `Axial/Coronal/Sagittal`, chon 2 cot neu khong co ground truth hoac 3 cot neu co, ve FLAIR va overlay mask cho tung hang, an truc toa do, dat title va `tight_layout`. Cac phep cat `[:, :, z_idx]`, `[:, y_idx, :]`, ` [x_idx, :, :]` lan luot la axial, coronal, sagittal.

## Code cell 20 - Prediction tren test case

```python
sample_batch = next(iter(test_loader))
sample_image = sample_batch["image"].to(device)
sample_label = sample_batch["label"]
sample_case_id = sample_batch["case_id"][0]
sample_spacing = tuple(float(value) for value in sample_batch["spacing"][0].cpu().numpy())
sample_affine = sample_batch["affine"][0].cpu().numpy()
sample_prediction_tensor = predict_volume(best_model, sample_image)
sample_prediction = sample_prediction_tensor[0, 0].cpu().numpy().astype(np.uint8)
sample_metrics = compute_case_metrics(sample_prediction, sample_label[0, 0], sample_spacing, include_surface=True)
prediction_path = predictions_dir / f"{sample_case_id}_{best_model_name}_seg.nii.gz"
nib.save(nib.Nifti1Image(sample_prediction, sample_affine), prediction_path)
visualize_slices(sample_image[0].cpu().numpy(), sample_prediction, sample_case_id, best_model_name, ground_truth=sample_label[0].numpy())
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-6 | Lay case dau tien cua test, chuyen image sang device, giu label/case ID va lay spacing/affine ve NumPy. |
| 7-9 | Suy luan full volume bang sliding window, doi output ve mask uint8 va tinh metric co HD95. |
| 10-11 | Tao ten file NIfTI, luu prediction voi affine goc de giu he toa do khong gian. |
| 12 | Ve FLAIR, ground truth va prediction tren ba mat phang. |

## Code cell 21 - Research report PDF

```python
from fpdf import FPDF

class ResearchSegmentationReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "BRAIN TUMOR SEGMENTATION RESEARCH REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, "Research use only - not for clinical diagnosis or treatment", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} | SIC Capstone 2026", align="C")
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1 | Import lop tao PDF. |
| 3-9 | Tao lop bao cao ke thua FPDF; `header` in tieu de va disclaimer o dau moi trang. |
| 11-14 | `footer` dat vi tri gan day trang, in so trang va ten project. |

Ham `export_research_report` tiep theo tao output directory, khoi tao PDF, them thong tin case/model/thoi gian/spacing, bang the tich WT/TC/ET, va neu co metrics thi them bang Dice/IoU/Precision/Recall/HD95. Sau do ham them phan limitations, goi `pdf.output`, in path va return `Path`. Tham so `metrics=None` cho phep xuat report chi co volume khi khong co ground truth.

## Code cell 22 - Xuat report mau

```python
sample_volumes = {
    region: sample_metrics[f"volume_cm3_{region}"]
    for region in REGION_BUILDERS
}
sample_report_path = reports_dir / f"{sample_case_id}_{best_model_name}_research_report.pdf"
export_research_report(
    case_id=sample_case_id,
    model_name=best_model_name,
    spacing=sample_spacing,
    volumes=sample_volumes,
    metrics=sample_metrics,
    output_file=sample_report_path,
)
```

| Doan code | Doan code nay dung de lam gi? |
|---|---|
| 1-3 | Tao dictionary volume theo ba region tu metric da tinh. |
| 4 | Dat ten report theo case ID va model. |
| 5-11 | Goi ham xuat report voi spacing, volume, metric va output path; PDF duoc luu vao `reports_dir`. |

## Thu tu chay va phu thuoc bien

Notebook phai chay tu tren xuong duoi: cell 1 cai package; cell 2 import; cell 3-5 tao moi truong va config; cell 6-8 nap du lieu; cell 9-12 tao split/dataloader; cell 13-15 dinh nghia metric/model/loss; cell 16 train va tao `best_model`; cell 17-18 can `MODEL_RUNS`; cell 19-22 can `best_model`, `test_loader`, `sample_metrics` va ham report.

Mot so hop dong du lieu quan trong:

- Image co dang `(4, H, W, D)` va label co dang `(1, H, W, D)`.
- Label raw 4 duoc remap thanh 3; cac label ngoai `{0,1,2,3,4}` bi tu choi.
- Affine va spacing phai duoc giu khi xuat NIfTI.
- Checkpoint duoc chon bang full validation; test chi dung cho bao cao cuoi.
- Bao cao la research output, khong phai chan doan lam sang.
