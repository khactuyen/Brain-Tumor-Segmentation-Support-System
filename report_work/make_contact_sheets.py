from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\SIC_Capstone 2026\report_work\final-render-pages")
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"D:\SIC_Capstone 2026\report_work\contact-sheets")
OUTPUT.mkdir(parents=True, exist_ok=True)

pages = sorted(SOURCE.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
thumb_w, thumb_h = 320, 453
gap, label_h = 24, 32
cols, rows = 3, 3

for sheet_index in range(0, len(pages), cols * rows):
    chunk = pages[sheet_index:sheet_index + cols * rows]
    canvas = Image.new(
        "RGB",
        (gap + cols * (thumb_w + gap), gap + rows * (thumb_h + label_h + gap)),
        "#D8DCE6",
    )
    draw = ImageDraw.Draw(canvas)
    for offset, page_path in enumerate(chunk):
        with Image.open(page_path) as src:
            thumb = src.convert("RGB")
            thumb.thumbnail((thumb_w, thumb_h))
        row, col = divmod(offset, cols)
        x = gap + col * (thumb_w + gap)
        y = gap + row * (thumb_h + label_h + gap)
        canvas.paste(thumb, (x, y))
        draw.text((x, y + thumb_h + 5), f"Trang {int(page_path.stem.split('-')[-1])}", font=font, fill="#171717")
    out = OUTPUT / f"sheet-{sheet_index // (cols * rows) + 1:02d}.png"
    canvas.save(out)
    print(out)
