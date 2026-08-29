"""
app.py (root) — Hugging Face Spaces entry point.

HF Spaces tìm file `app.py` ở root repo và chạy nó.
File này chỉ import và launch Gradio app từ web/gradio_app/.
"""

import sys
from pathlib import Path

# Thêm thư mục chứa Gradio app vào path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "web" / "gradio_app"))
sys.path.insert(0, str(_ROOT / "web"))

from web.gradio_app.app import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
