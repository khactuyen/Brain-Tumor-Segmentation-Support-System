import json
from pathlib import Path

nb_path = Path(r"D:\SIC_Capstone 2026\SIC_Capstone_v2.ipynb")

def to_source_lines(source_str: str) -> list:
    lines = source_str.split('\n')
    out = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            out.append(line + '\n')
        else:
            if line:
                out.append(line)
    return out

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

translations = {
    4: [
        ("LOGGING SETUP", "THIẾT LẬP LOG"),
        ("REPRODUCIBILITY", "TÁI LẬP KẾT QUẢ"),
    ],
    13: [
        ("# Load & normalize 4 modalities", "# Tải & chuẩn hóa 4 chuỗi MRI"),
        ("# Load & remap segmentation mask", "# Tải & ánh xạ lại label phân đoạn"),
    ],
    18: [
        ("# Per-class Dice (NCR / ED / ET)", "# Dice theo từng lớp (NCR / ED / ET)"),
        ("# Mean Dice (trung bình 3 classes)", "# Dice trung bình (3 classes)"),
    ],
    25: [
        ("# Load best model", "# Tải model tốt nhất"),
    ],
}

changed = False
for ci, repls in translations.items():
    cell = nb['cells'][ci]
    src = ''.join(cell['source'])
    orig = src
    for old, new in repls:
        if old in src:
            src = src.replace(old, new)
    if src != orig:
        cell['source'] = to_source_lines(src)
        changed = True

# Drop heavy outputs from cell 10 (1.3MB of embedded plots)
nb['cells'][10]['outputs'] = []
nb['cells'][10]['execution_count'] = None

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print('comment translations applied:', changed)
print('cell10 outputs cleared: True')
print('new file size: %.2f MB' % (nb_path.stat().st_size / 1024 / 1024))