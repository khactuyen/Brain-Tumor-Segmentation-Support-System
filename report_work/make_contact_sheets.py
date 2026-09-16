from pathlib import Path
from PIL import Image, ImageDraw


source = Path(r"D:\SIC_Capstone 2026\report_work\render_v1_v2_qa")
pages = sorted(source.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
out_dir = source / "contact_sheets"
out_dir.mkdir(exist_ok=True)

for sheet_index in range(0, len(pages), 4):
    batch = pages[sheet_index:sheet_index + 4]
    opened = [Image.open(path).convert("RGB") for path in batch]
    page_w = max(img.width for img in opened)
    page_h = max(img.height for img in opened)
    label_h = 36
    canvas = Image.new("RGB", (page_w * 2, (page_h + label_h) * 2), "#b7b7b7")
    draw = ImageDraw.Draw(canvas)
    for offset, (path, img) in enumerate(zip(batch, opened)):
        col = offset % 2
        row = offset // 2
        x = col * page_w
        y = row * (page_h + label_h)
        draw.rectangle((x, y, x + page_w, y + label_h), fill="#202020")
        draw.text((x + 12, y + 9), path.stem, fill="white")
        canvas.paste(img, (x, y + label_h))
    first_page = sheet_index + 1
    last_page = sheet_index + len(batch)
    canvas.save(out_dir / f"sheet_{first_page:02d}_{last_page:02d}.jpg", quality=90, subsampling=0)

print(f"Created {len(list(out_dir.glob('sheet_*.jpg')))} contact sheets for {len(pages)} pages")
