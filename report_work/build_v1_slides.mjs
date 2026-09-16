import fs from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const { Presentation, PresentationFile } = await import(pathToFileURL('C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);

const workspaceDir = 'D:/SIC_Capstone 2026';
const SKILL_DIR = 'C:/Users/LENOVO/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11809/skills/presentations';
const TMP_DIR = path.join(workspaceDir, 'report_work', 'v1_slides_build');
const FINAL_PPTX = path.join(workspaceDir, 'doc', 'SIC_Capstone_v1_Thuyet_trinh_10_slide.pptx');
await fs.mkdir(TMP_DIR, { recursive: true });
const { resolvePresentationFont } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);
const family = resolvePresentationFont({ preferred: ['Aptos', 'Arial'] });

const W = 1280, H = 720;
const C = { bg:'#F7F9FC', navy:'#E8EFF6', blue:'#1F4E79', teal:'#176B73', orange:'#B56B16', white:'#17324D', muted:'#587086', panel:'#EAF1F7', red:'#A44A4A' };
const assets = {
  labels: path.join(workspaceDir,'report_work','brats_label_visual_for_report.png'),
  eda: path.join(workspaceDir,'report_work','notebook_v1_explained_assets','v1_eda_raw_nifti.png'),
  curves: path.join(workspaceDir,'report_work','notebook_v1_explained_assets','v1_training_curves.png'),
  pred: path.join(workspaceDir,'report_work','notebook_v1_explained_assets','v1_prediction_three_planes.png'),
};
const p = Presentation.create({ slideSize: { width: W, height: H } });

function rect(slide, x,y,w,h, fill, radius=0) {
  const s = slide.shapes.add({ geometry: 'rect', position:{left:x,top:y,width:w,height:h}, fill, line:{fill:fill === C.bg ? C.bg : '#D3DEE8',width:fill === C.bg ? 0 : 1} });
  return s;
}
function text(slide, content, x,y,w,h, size=24, color=C.white, opts={}) {
  if (typeof size === 'string' && size.startsWith('#')) {
    const legacyColor = size;
    const legacyOpts = (color && typeof color === 'object') ? color : {};
    size = legacyOpts.size ?? 24;
    color = legacyColor;
    opts = legacyOpts;
  }
  const s = slide.shapes.add({ geometry:'textbox', position:{left:x,top:y,width:w,height:h}, fill:'none', line:{fill:'none',width:0} });
  s.text = content;
  s.text.style = { typeface:family, fontSize:size, color, bold:!!opts.bold, italic:!!opts.italic, alignment:opts.align||'left', verticalAlignment:opts.valign||'top', autoFit:'shrink' };
  return s;
}
function title(slide, num, heading, kicker='BRAIN TUMOR SEGMENTATION') {
  text(slide, kicker, 64, 32, 720, 22, C.teal, {size:14,bold:true});
  text(slide, `${num}  ${heading}`, 64, 62, 1120, 60, 34, C.white, {bold:true});
  rect(slide,64,130,1152,3,C.teal);
}
function footer(slide, n) { text(slide, `SIC Capstone 2026  |  Notebook v1`, 64, 684, 800, 18, C.muted, {size:12}); text(slide, String(n).padStart(2,'0'), 1175, 684, 40, 18, C.muted, {size:12,align:'right'}); }
function base(n, heading, kicker) { const s=p.slides.add(); s.background.fill=C.bg; title(s,n,heading,kicker); footer(s,n); return s; }
function notes(slide, t) { slide.speakerNotes.textFrame.setText(t); }
function image(slide, file, x,y,w,h, alt) {
  const dataUrl = `data:image/png;base64,${readFileSync(file).toString('base64')}`;
  slide.images.add({ dataUrl, alt, fit:'contain', position:{left:x,top:y,width:w,height:h} });
}
function bulletText(items) { return items.map(x=>`• ${x}`).join('\n'); }

// 1 cover
{ const s=p.slides.add(); s.background.fill=C.bg; rect(s,0,0,460,H,C.navy); text(s,'SIC CAPSTONE 2026',70,72,330,28,16,C.teal,{bold:true}); text(s,'Brain Tumor\nSegmentation',70,150,520,150,48,C.white,{bold:true}); text(s,'So sánh notebook v1 và v2',72,335,470,55,24,C.muted); rect(s,70,425,220,4,C.orange); text(s,'3D U-Net  |  Swin UNETR',72,457,420,28,18,C.orange,{bold:true}); image(s,assets.pred,535,95,660,500,'Ví dụ dự đoán segmentation theo ba mặt phẳng'); text(s,'Trình bày dự án',72,620,300,25,16,C.muted); notes(s,'Bộ slide trình bày hai notebook: v1 dùng 3D U-Net, v2 dùng Swin UNETR.'); }

// 2 problem
{ const s=base(2,'Bối cảnh và bài toán','WHY THIS MATTERS'); text(s,'Từ ảnh MRI 3D đến mask vùng u có thể kiểm tra được',64,164,720,46,28,C.orange,{bold:true}); text(s,bulletText(['Phân đoạn thủ công tốn thời gian và phụ thuộc người đọc ảnh.','Mỗi ca có bốn modality MRI, mỗi modality bổ sung một tín hiệu mô học.','Notebook v1 tạo baseline với 3D U-Net; notebook v2 thử Swin UNETR.']),80,240,570,180,23,C.white); rect(s,760,190,440,320,C.panel,18); text(s,'INPUT',800,225,150,26,15,C.teal,{bold:true}); text(s,'4 modality MRI\nT1  •  T1ce  •  T2  •  T2-FLAIR',800,260,330,82,25,C.white,{bold:true}); rect(s,800,375,320,3,C.orange); text(s,'OUTPUT',800,400,150,26,15,C.teal,{bold:true}); text(s,'Mask 3D\nWT  •  TC  •  ET',800,435,330,65,25,C.white,{bold:true}); notes(s,'Mục tiêu là segmentation voxel-level; hai notebook dùng hai mô hình khác nhau.'); }

// 3 dataset
{ const s=base(3,'Dataset, modality và nhãn','DATA'); text(s,'1.251 ca BraTS, chuẩn hóa về một schema 3D',64,160,700,42,27,C.orange,{bold:true}); text(s,bulletText(['Volume ảnh: (4, 240, 240, 155), float32.','Label: (1, 240, 240, 155), các giá trị sau remap là 0, 1, 2, 3.','Raw label 4 được đưa về class nội bộ 3 để huấn luyện.']),72,220,560,145,21,C.white); rect(s,690,175,500,360,C.panel,18); const rows=[['Raw','Model','Ý nghĩa'],['0','0','Background'],['1','1','NCR/NET'],['2','2','ED'],['4','3','ET']]; const y0=210; rows.forEach((r,i)=>{ const yy=y0+i*58; rect(s,720,yy,440,48,i===0?C.blue:(i%2? '#F3F6F9':'#EAF1F7')); text(s,r[0],740,yy+12,80,24,i===0?'#FFFFFF':C.white, {size:16,bold:i===0}); text(s,r[1],830,yy+12,80,24,i===0?'#FFFFFF':C.white,{size:16,bold:i===0}); text(s,r[2],920,yy+12,220,24,i===0?'#FFFFFF':C.white,{size:16,bold:i===0}); }); text(s,'WT = label > 0   |   TC = {1,3}   |   ET = label 3',720,485,450,40,16,C.teal,{bold:true}); image(s,assets.labels,65,415,540,220,'Minh họa vị trí các nhãn BraTS'); notes(s,'Cần nhấn mạnh sự khác nhau giữa raw label 4 và class nội bộ 3.'); }

// 4 pipeline
{ const s=base(4,'Pipeline xử lý dữ liệu','PIPELINE'); text(s,'Mỗi bước biến dữ liệu thô thành một phép đo có thể audit',64,160,850,40,26,C.orange,{bold:true}); const steps=[['01','NIfTI','Đọc 4 modality + mask'],['02','QC','Shape, affine, spacing, intensity'],['03','PREPROCESS','Z-score, clip percentile, remap label'],['04','SPLIT','70% train / 15% val / 15% test'],['05','PATCH','Crop 96³ + augmentation'],['06','INFERENCE','Sliding window + TTA + postprocess']]; steps.forEach((st,i)=>{const x=72+(i%3)*380, y=235+Math.floor(i/3)*175; rect(s,x,y,330,120,C.panel,16); text(s,st[0],x+20,y+17,52,28,18,C.teal,{bold:true}); text(s,st[1],x+85,y+15,220,28,20,C.white,{bold:true}); text(s,st[2],x+20,y+59,285,45,16,C.muted);}); notes(s,'Pipeline đi từ file NIfTI đến prediction volume và hình trực quan hóa.'); }

// 5 architecture
{ const s=base(5,'Hai mô hình 3D','MODEL'); text(s,'v1 dùng 3D U-Net, v2 dùng Swin UNETR',64,160,940,42,26,C.orange,{bold:true}); // editable diagram blocks
  const bx=[90,285,480,675,870]; const labels=[['Input','4 × 96³'],['Encoder','3D U-Net'],['Bottleneck','U-Net: 128 → 256'],['Decoder','Skip connections'],['Output','4 classes']]; bx.forEach((x,i)=>{rect(s,x,285,145,120,i===2?C.orange:C.blue,14); text(s,labels[i][0],x+10,310,125,26,18,C.white,{bold:true,align:'center'}); text(s,labels[i][1],x+10,350,125,30,15,C.white,{align:'center'}); if(i<4) { rect(s,x+150,340,48,4,C.teal); text(s,'▶',x+162,320,30,30,20,C.teal,{bold:true}); }}); text(s,'v1: 3D U-Net, 4.81M tham số',90,460,470,30,20,C.muted); text(s,'v2: Swin UNETR, 15.71M tham số\nTransformer encoder, mạnh hơn nhưng cần protocol riêng.',650,455,470,70,20,C.white); notes(s,'v1 là baseline 3D U-Net; v2 dùng Swin UNETR. Vì cohort và protocol khác nhau, cần thận trọng khi so sánh trực tiếp.'); }

// 6 training
{ const s=base(6,'Huấn luyện và checkpoint','TRAINING'); text(s,'Hai notebook đều chọn checkpoint trên validation, nhưng protocol khác nhau',64,160,1080,42,25,C.orange,{bold:true}); image(s,assets.curves,65,230,590,355,'Training loss và validation Dice của v1'); rect(s,710,215,470,345,C.panel,18); text(s,bulletText(['v1: 3D U-Net, resume tới epoch 50, checkpoint epoch 40.','v2: Swin UNETR, feature_size 24, đã train đủ 20 epoch.','v1 có held-out test; v2 chỉ báo cáo validation.','Mixed precision + GradScaler dùng ở cả pipeline.']),745,250,395,250,20,C.white); text(s,'Đường cong hiển thị là của notebook v1.',745,510,370,25,15,C.teal,{bold:true}); notes(s,'Không gọi đây là so sánh công bằng vì v1 và v2 không cùng cohort, split và protocol đánh giá.'); }

// 7 metrics
{ const s=base(7,'Đánh giá mô hình','METRICS'); text(s,'Một metric không đủ để mô tả chất lượng segmentation',64,160,900,42,26,C.orange,{bold:true}); const cards=[['Dice','Độ chồng lấp\nCàng cao càng tốt'],['IoU','Giao / hợp\nKhắt khe hơn Dice'],['Precision','Dự đoán u đúng\nÍt false positive'],['Recall','Tìm đủ vùng u\nÍt false negative'],['HD95','Khoảng cách biên\nCàng thấp càng tốt']]; cards.forEach((c,i)=>{const x=65+i*235; rect(s,x,250,205,190,i===4?C.orange:C.panel,16); text(s,c[0],x+20,282,165,32,25,i===4?C.bg:C.teal,{bold:true,align:'center'}); text(s,c[1],x+20,345,165,65,18,i===4?C.bg:C.white,{align:'center'});}); text(s,'Quy ước empty-empty trong notebook: Dice/IoU/Precision/Recall = 1, HD95 = 0.',100,520,1060,35,18,C.muted,{italic:true,align:'center'}); notes(s,'HD95 tính theo khoảng cách vật lý có spacing, không chỉ khoảng cách voxel.'); }

// 8 results
{ const s=base(8,'Kết quả của v1 và v2','RESULTS'); text(s,'Các con số chỉ nên đọc cùng với cohort và protocol tương ứng',64,160,1070,42,24,C.orange,{bold:true}); const headers=['Model / split','Mean Dice','WT','TC','ET','HD95 (mm)']; const vals=[['v1 validation','0.8052','0.8308','0.8125','0.7724','29.683'],['v1 held-out test','0.7858','0.8158','0.7938','0.7476','31.117']]; const x0=100, widths=[220,180,150,150,150,180]; let x=x0; headers.forEach((h,i)=>{rect(s,x,235,widths[i],54,C.blue); text(s,h,x+8,253,widths[i]-16,25,16,'#FFFFFF',{bold:true,align:'center'}); x+=widths[i];}); vals.forEach((row,r)=>{x=x0; row.forEach((v,i)=>{rect(s,x,289+r*58,widths[i],58,r===0?'#EAF1F7':'#F3F6F9'); text(s,v,x+8,308+r*58,widths[i]-16,25,18,C.white,{align:'center',bold:i===0}); x+=widths[i];});}); rect(s,100,420,1030,84,C.panel); text(s,'v2 Swin UNETR validation (204 ca): Mean Dice 0.8653  |  WT 0.8498  |  TC 0.7461  |  ET 1.0000*',125,438,980,25,18,C.white,{bold:true}); text(s,'*ET = 1.0 ở toàn bộ thống kê do quy ước empty-empty; v2 không có held-out test/HD95 trong output này.',125,470,980,22,14,C.muted); text(s,'Không kết luận v2 tốt hơn v1 nếu chưa chạy cùng cohort, split và protocol.',110,550,980,30,19,C.orange,{bold:true,align:'center'}); notes(s,'v1 có validation và held-out test. v2 chỉ có validation 204 ca, Mean Dice .8653 và ET luôn 1.0 theo output thống kê.'); }

// 9 prediction
{ const s=base(9,'Trực quan hóa prediction','EVIDENCE'); text(s,'Một ca mẫu cho thấy prediction bám khá sát ground truth ở cả ba mặt phẳng',64,160,1050,42,24,C.orange,{bold:true}); image(s,assets.pred,75,225,760,410,'Prediction theo axial coronal sagittal'); rect(s,885,230,305,340,C.panel,18); text(s,'Case',915,262,80,20,15,C.teal,{bold:true}); text(s,'BraTS-GLI-01021-000',915,295,230,32,17,C.white,{bold:true}); text(s,'Ba cột chính:\nFLAIR • Ground truth • U-Net3d',915,365,230,70,18,C.white); text(s,'Hình minh họa giúp kiểm tra lỗi biên, vùng thiếu hoặc vùng dự đoán thừa.',915,475,230,64,16,C.muted); notes(s,'Hình chỉ là một ca thuận lợi để kiểm tra định tính; metric toàn test mới là bằng chứng chính.'); }

// 10 conclusion
{ const s=base(10,'Kết luận và điểm cần audit','TAKEAWAYS'); text(s,'Hai mô hình cho thấy tiến trình phát triển, chưa phải một phép so sánh công bằng',64,160,1100,42,24,C.orange,{bold:true}); rect(s,75,235,520,320,C.panel,18); text(s,'Đã làm được',105,268,220,28,20,C.teal,{bold:true}); text(s,bulletText(['v1: 3D U-Net baseline, có held-out test.','v2: Swin UNETR, validation Mean Dice 0.8653.','Cả hai dùng bốn modality MRI và mask 3D.','Đã kiểm tra định tính bằng hình prediction.']),105,320,430,180,21,C.white); rect(s,650,235,540,320,C.panel,18); text(s,'Cần trả lời khi bảo vệ',680,268,350,28,20,C.red,{bold:true}); text(s,bulletText(['Cohort và split khác nhau, không so sánh Dice trực tiếp.','v2 chỉ có validation, chưa có held-out test.','NIfTI output cần reverse-map class 3 → raw label 4 nếu theo chuẩn BraTS.','Kết quả chưa phải chẩn đoán lâm sàng.']),680,320,455,200,21,C.white); text(s,'Thông điệp cuối: so sánh đúng protocol trước khi so sánh con số.',75,610,1080,32,21,C.orange,{bold:true,align:'center'}); notes(s,'Kết luận cần phân biệt cải tiến mô hình với thay đổi protocol đánh giá.'); }

const candidate = path.join(TMP_DIR,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
for (let i=0;i<p.slides.items.length;i++) { const png=await p.export({slide:p.slides.items[i],format:'png',scale:1}); await fs.writeFile(path.join(TMP_DIR,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer())); }
console.log(candidate);
