import json
import re

path = r"D:\SIC_Capstone 2026\SIC_Capstone_v2.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changed_cells = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            new_source.append(line)
            if "label    = self._remap_label(label)" in line or "label = self._remap_label(label)" in line:
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else "        "
                expand_line = f"{indent}label = np.expand_dims(label, axis=0)  # (1, H, W, D)\n"
                new_source.append(expand_line)
                
        # Remove consecutive duplicate np.expand_dims just in case it was run twice
        cleaned_source = []
        for i, line in enumerate(new_source):
            if "np.expand_dims(label, axis=0)" in line:
                if i > 0 and "np.expand_dims(label, axis=0)" in new_source[i-1]:
                    continue
            cleaned_source.append(line)
                
        if cell["source"] != cleaned_source:
            cell["source"] = cleaned_source
            changed_cells += 1

if changed_cells > 0:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"✅ Đã thêm np.expand_dims cho label trong {changed_cells} cells của {path}!")
else:
    print("☑️ Không cần sửa, file đã có sẵn phần mở rộng chiều cho label.")
