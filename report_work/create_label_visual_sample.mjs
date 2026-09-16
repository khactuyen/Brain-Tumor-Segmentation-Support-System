import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const sharp = require("C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp");

const root = "D:/SIC_Capstone 2026";
const source = `${root}/report_work/assets/v2_cell9_output0.png`;
const output = `${root}/report_work/brats_label_visual_sample.png`;

const flairBox = { left: 1184, top: 484, width: 376, height: 372 };
const overlayBox = { left: 1574, top: 484, width: 376, height: 372 };

const labelColors = {
  ed: [60, 174, 79, 176],
  ncr: [225, 41, 47, 176],
  et: [255, 135, 13, 176],
};

function classifyHue(r, g, b) {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;
  const saturation = max === 0 ? 0 : delta / max;
  if (saturation <= 0.22 || delta === 0) return null;

  let hue;
  if (max === rn) hue = ((gn - bn) / delta) % 6;
  else if (max === gn) hue = (bn - rn) / delta + 2;
  else hue = (rn - gn) / delta + 4;
  hue = ((hue * 60) + 360) % 360;

  if (hue <= 16 || hue >= 347) return "ncr";
  if (hue >= 72 && hue <= 173) return "ed";
  if (hue > 16 && hue < 72) return "et";
  return null;
}

async function maskedPanel(flair, overlay, wanted) {
  const { data, info } = await sharp(overlay).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const mask = Buffer.alloc(info.width * info.height * 4);
  let count = 0;
  for (let i = 0; i < info.width * info.height; i += 1) {
    const label = classifyHue(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
    const offset = i * 4;
    if (label === wanted) {
      const color = labelColors[wanted];
      mask[offset] = color[0];
      mask[offset + 1] = color[1];
      mask[offset + 2] = color[2];
      mask[offset + 3] = color[3];
      count += 1;
    }
  }
  if (count < 50) throw new Error(`Không tách được mask ${wanted}: ${count} pixels`);
  const maskPng = await sharp(mask, { raw: { width: info.width, height: info.height, channels: 4 } }).png().toBuffer();
  const panel = await sharp(flair).composite([{ input: maskPng }]).png().toBuffer();
  return { panel, count };
}

function asDataUri(buffer) {
  return `data:image/png;base64,${buffer.toString("base64")}`;
}

const escapeXml = (text) => text
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;");

async function main() {
  const base = sharp(source);
  const flair = await base.clone().extract(flairBox).png().toBuffer();
  const overlay = await base.clone().extract(overlayBox).png().toBuffer();
  const ed = await maskedPanel(flair, overlay, "ed");
  const ncr = await maskedPanel(flair, overlay, "ncr");
  const et = await maskedPanel(flair, overlay, "et");

  const images = {
    flair: asDataUri(flair),
    overlay: asDataUri(overlay),
    ed: asDataUri(ed.panel),
    ncr: asDataUri(ncr.panel),
    et: asDataUri(et.panel),
  };

  const width = 1800;
  const height = 1220;
  const panels = [
    { x: 55, y: 170, image: images.flair, title: "1. Ảnh T2-FLAIR gốc" },
    { x: 625, y: 170, image: images.overlay, title: "2. Ground truth phủ lên MRI" },
    { x: 55, y: 705, image: images.ed, title: "Label 2 – ED: phù nề quanh u", color: "#249647" },
    { x: 625, y: 705, image: images.ncr, title: "Label 1 – NCR/NET: lõi hoại tử/không bắt thuốc", color: "#C81E2A" },
    { x: 1195, y: 705, image: images.et, title: "ET – nhãn gốc 4 → class mô hình 3", color: "#EA6D00" },
  ];

  const imageMarkup = panels.map((p) => `
    <g>
      <rect x="${p.x - 15}" y="${p.y - 48}" width="520" height="478" rx="18" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="2"/>
      <text x="${p.x + 245}" y="${p.y - 13}" text-anchor="middle" class="panel-title" fill="${p.color ?? "#0F172A"}">${escapeXml(p.title)}</text>
      <image x="${p.x}" y="${p.y}" width="490" height="385" preserveAspectRatio="xMidYMid meet" href="${p.image}"/>
    </g>`).join("");

  const svg = `
  <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
    <style>
      text { font-family: "Segoe UI", Arial, sans-serif; }
      .panel-title { font-size: 23px; font-weight: 700; }
      .region-title { font-size: 24px; font-weight: 700; }
      .region-sub { font-size: 20px; }
    </style>
    <rect width="100%" height="100%" fill="#F8FAFC"/>
    <rect x="0" y="0" width="1800" height="18" fill="#1E40AF"/>
    <text x="900" y="67" text-anchor="middle" font-size="36" font-weight="800" fill="#0F172A">Ví dụ trực quan nhãn BraTS trên một lát cắt MRI axial</text>
    <text x="900" y="105" text-anchor="middle" font-size="22" fill="#475569">Ảnh thật trích từ EDA notebook v2 – cùng một lát cắt T2-FLAIR và ground truth</text>
    ${imageMarkup}

    <g>
      <rect x="1180" y="122" width="550" height="478" rx="18" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="2"/>
      <text x="1455" y="165" text-anchor="middle" class="panel-title" fill="#0F172A">3. Cách gộp vùng khi đánh giá</text>
      <rect x="1225" y="205" width="460" height="330" rx="28" fill="#DCFCE7" stroke="#16A34A" stroke-width="4"/>
      <text x="1250" y="245" class="region-title" fill="#166534">WT – Whole Tumor</text>
      <text x="1250" y="278" class="region-sub" fill="#166534">label 1 + 2 + 4</text>
      <rect x="1290" y="310" width="330" height="180" rx="25" fill="#FEE2E2" stroke="#DC2626" stroke-width="4"/>
      <text x="1315" y="352" class="region-title" fill="#991B1B">TC – Tumor Core</text>
      <text x="1315" y="385" class="region-sub" fill="#991B1B">label 1 + 4</text>
      <rect x="1380" y="408" width="150" height="62" rx="18" fill="#FFEDD5" stroke="#F97316" stroke-width="4"/>
      <text x="1455" y="437" text-anchor="middle" font-size="21" font-weight="700" fill="#C2410C">ET: raw 4 → class 3</text>
      <text x="1455" y="572" text-anchor="middle" font-size="19" fill="#334155">ET ⊂ TC ⊂ WT</text>
    </g>

    <text x="900" y="1165" text-anchor="middle" font-size="19" font-weight="700" fill="#334155">Nhãn gốc BraTS: {0, 1, 2, 4}  →  class đưa vào mô hình: {0, 1, 2, 3}; chỉ có ET được đổi 4 → 3.</text>
    <text x="900" y="1195" text-anchor="middle" font-size="17" fill="#475569">Label 0 = nền/ngoài vùng u được gán nhãn. Đây là một lát cắt 2D của ảnh MRI 3D.</text>
  </svg>`;

  await sharp(Buffer.from(svg)).png().toFile(output);
  console.log(output);
  console.log(JSON.stringify({ ED: ed.count, "NCR/NET": ncr.count, ET: et.count }));
}

await main();
