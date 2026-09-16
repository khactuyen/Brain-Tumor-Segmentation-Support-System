from pathlib import Path
from PIL import Image, ImageDraw

src = Path(r"D:\SIC_Capstone 2026\report_work\v1_doc_render")
out = Path(r"D:\SIC_Capstone 2026\report_work\v1_doc_contact_sheets")
out.mkdir(exist_ok=True)
files = sorted(src.glob("page-*.png"))
for batch_index in range(0, len(files), 4):
    batch = files[batch_index:batch_index + 4]
    thumbs = []
    for file in batch:
        image = Image.open(file).convert("RGB")
        image.thumbnail((620, 880))
        thumbs.append((file, image.copy()))
    sheet = Image.new("RGB", (1300, 1840), "#d5d8dc")
    draw = ImageDraw.Draw(sheet)
    positions = [(20, 50), (660, 50), (20, 960), (660, 960)]
    for (file, image), (x, y) in zip(thumbs, positions):
        draw.text((x, y - 28), file.stem, fill="black")
        sheet.paste(image, (x, y))
    sheet.save(out / f"contact-{batch_index // 4 + 1:02d}.jpg", quality=88)
print(f"pages={len(files)} sheets={(len(files) + 3) // 4}")
