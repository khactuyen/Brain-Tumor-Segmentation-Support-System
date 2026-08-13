"""
Script to update SIC_Capstone_v2.ipynb notebook to save preprocessed .npz files directly to Google Drive permanently.
"""

import json
from pathlib import Path

notebook_path = Path(__file__).resolve().parent.parent / "SIC_Capstone_v2.ipynb"

if not notebook_path.exists():
    print(f"[!] Notebook not found: {notebook_path}")
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

updated_cells = 0

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        source_str = "".join(source_lines)
        
        # 1. Update preprocess_case function to use np.asarray for fast reading
        if "def preprocess_case" in source_str:
            new_source = [
                "def preprocess_case(case_id: str, data_root: Path, out_dir: Path) -> Path:\n",
                "    \"\"\"Đọc 4 chuỗi MRI + mask từ NIfTI, Z-score normalize, remap label và lưu thành file .npz duy nhất trên Google Drive.\"\"\"\n",
                "    out_path = out_dir / f\"{case_id}.npz\"\n",
                "    if out_path.exists():\n",
                "        return out_path\n",
                "\n",
                "    case_dir = data_root / case_id\n",
                "    channels = []\n",
                "    for mod in CFG.modalities:\n",
                "        fpath = case_dir / f\"{case_id}-{mod}.nii.gz\"\n",
                "        nii = nib.load(fpath)\n",
                "        arr = np.asarray(nii.dataobj, dtype=np.float32)\n",
                "        mask = arr > 0\n",
                "        if mask.any():\n",
                "            arr[mask] = (arr[mask] - arr[mask].mean()) / (arr[mask].std() + 1e-8)\n",
                "        channels.append(arr)\n",
                "    image = np.stack(channels, axis=0)  # (4, H, W, D)\n",
                "\n",
                "    seg_path = case_dir / f\"{case_id}-seg.nii.gz\"\n",
                "    if seg_path.exists():\n",
                "        seg_nii = nib.load(seg_path)\n",
                "        seg = np.asarray(seg_nii.dataobj, dtype=np.int16)\n",
                "        label = np.zeros_like(seg, dtype=np.int16)\n",
                "        for src, dst in LABEL_REMAP.items():\n",
                "            label[seg == src] = dst\n",
                "        label = np.expand_dims(label, axis=0)  # (1, H, W, D)\n",
                "    else:\n",
                "        label = np.zeros((1,) + image.shape[1:], dtype=np.int16)\n",
                "\n",
                "    np.savez_compressed(out_path, image=image, label=label)\n",
                "    return out_path\n",
                "\n",
                "\n",
                "def preprocess_all_cases(data_root: Path, out_dir: Path, case_ids: List[str]):\n",
                "    \"\"\"Tiền xử lý toàn bộ các case sang file .npz trên Google Drive (Lưu vĩnh viễn, không phải làm lại).\"\"\"\n",
                "    out_dir = Path(out_dir)\n",
                "    out_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "    missing_cases = [cid for cid in case_ids if not (out_dir / f\"{cid}.npz\").exists()]\n",
                "    if missing_cases:\n",
                "        print(f\"🔄 Đang tiền xử lý {len(missing_cases)} cases sang Google Drive ({out_dir})...\")\n",
                "        for cid in tqdm(missing_cases, desc=\"Preprocessing Drive\"):\n",
                "            preprocess_case(cid, data_root, out_dir)\n",
                "        print(\"✅ Tiền xử lý hoàn tất! File đã được lưu vĩnh viễn trên Google Drive.\")\n",
                "    else:\n",
                "        print(f\"⚡ Dữ liệu tiền xử lý .npz đã có sẵn vĩnh viễn trên Google Drive tại {out_dir}! Không cần load lại.\")\n"
            ]
            cell["source"] = new_source
            updated_cells += 1

        # 2. Update PREPROCESSED_DIR path to persistent Google Drive path
        if "PREPROCESSED_DIR = Path" in source_str:
            new_lines = []
            for line in source_lines:
                if "PREPROCESSED_DIR = Path" in line:
                    new_lines.append('PREPROCESSED_DIR = Path(getattr(CFG, "preprocessed_dir", "/content/drive/MyDrive/BraTS2023/processed_npz"))\n')
                else:
                    new_lines.append(line)
            cell["source"] = new_lines
            updated_cells += 1

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[✓] Updated {updated_cells} cells in {notebook_path.name} to persistent Google Drive storage!")
