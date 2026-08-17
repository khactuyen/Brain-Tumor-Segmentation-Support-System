"""Extract all code cells from a Jupyter notebook and save to a text file."""
import json
import sys
import os

def extract_notebook(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(f"=== Notebook: {os.path.basename(notebook_path)} ===\n")
        out.write(f"Total cells: {len(cells)}\n\n")
        
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type', 'unknown')
            source = ''.join(cell.get('source', []))
            out.write(f"{'='*80}\n")
            out.write(f"CELL {i} [{cell_type.upper()}]\n")
            out.write(f"{'='*80}\n")
            out.write(source)
            out.write("\n\n")
    
    print(f"Extracted {len(cells)} cells to {output_path}")

if __name__ == '__main__':
    base = r"t:\AI SAMSUNG\DoAn4\Brain-Tumor-Segmentation-Support-System"
    extract_notebook(
        os.path.join(base, "SIC_Capstone.ipynb"),
        os.path.join(base, "scripts", "_v1_extracted.txt")
    )
    extract_notebook(
        os.path.join(base, "SIC_Capstone_v2.ipynb"),
        os.path.join(base, "scripts", "_v2_extracted.txt")
    )
