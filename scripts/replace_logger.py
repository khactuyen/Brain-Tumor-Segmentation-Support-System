import json
from pathlib import Path

def replace_logger_in_notebook(notebook_path: str):
    path = Path(notebook_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            # join to a single string to do block replace easily
            source_str = "".join(cell["source"])
            original_str = source_str
            
            source_str = source_str.replace("logger.info(", "print(")
            source_str = source_str.replace("logger.warning(", "print(")
            source_str = source_str.replace("logger.error(", "print(")
            
            # Remove the logging setup block
            import re
            source_str = re.sub(r'import logging\n', '', source_str)
            source_str = re.sub(r'logging\.basicConfig\([\s\S]*?\)\n', '', source_str)
            source_str = re.sub(r'logger = logging\.getLogger\(.*?\)\n', '', source_str)
            
            if source_str != original_str:
                changed = True
                # split back to lines for Jupyter format
                # keeping \n at the end of each line
                lines = []
                for i, line in enumerate(source_str.split('\n')):
                    if i < len(source_str.split('\n')) - 1:
                        lines.append(line + '\n')
                    else:
                        if line: # if there is something without \n at EOF
                            lines.append(line)
                cell["source"] = lines

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"✅ Successfully updated {path.name} to use print instead of logger.")
    else:
        print("No logger usages found to replace.")

if __name__ == "__main__":
    replace_logger_in_notebook(r"D:\SIC_Capstone 2026\SIC_Capstone_v2.ipynb")
