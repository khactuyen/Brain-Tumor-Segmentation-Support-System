import json
import re

path = r"D:\SIC_Capstone 2026\SIC_Capstone_v2.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changed_cells = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        skip_mode = False
        for line in cell["source"]:
            # string replacement
            line = line.replace("logger.info(", "print(")
            line = line.replace("logger.warning(", "print(")
            line = line.replace("logger.error(", "print(")
            
            # strip spaces for condition checks
            stripped = line.strip()
            
            if "import logging" in stripped:
                continue
            if "logger = logging.getLogger" in stripped:
                continue
                
            # block removal for logging.basicConfig(...)
            if "logging.basicConfig(" in stripped:
                skip_mode = True
                # If it's a one-liner
                if ")" in stripped:
                    skip_mode = False
                continue
                
            if skip_mode:
                if ")" in stripped:
                    skip_mode = False
                continue
                
            new_source.append(line)
            
        if cell["source"] != new_source:
            cell["source"] = new_source
            changed_cells += 1

if changed_cells > 0:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"✅ Đã dọn dẹp logger trong {changed_cells} cells của {path}!")
else:
    print("☑️ Không tìm thấy logger nào cần dọn dẹp thêm.")
