const fs = require('fs');

const path = 'SIC_Capstone.ipynb';
const notebook = JSON.parse(fs.readFileSync(path, 'utf8'));

function cell(cell_type, text) {
  return {
    cell_type,
    metadata: {},
    source: text.split('\n').map((line, index, lines) => line + (index < lines.length - 1 ? '\n' : '')),
    ...(cell_type === 'code' ? { outputs: [], execution_count: null } : {})
  };
}

const newCells = [
  cell('markdown', '## 4. Data Understanding: Load, Info, and Describe\n\nThis section builds a manifest of the BraTS cases, checks file completeness, and profiles image and label volumes before training.'),
  cell('code', `import pandas as pd
from IPython.display import display

MODALITIES = ['t1n', 't1c', 't2w', 't2f']
SEG_SUFFIX = 'seg'
PROFILE_CASES = 5

def expected_case_files(case_id):
    case_dir = Path(DATA_ROOT) / case_id
    return {name: case_dir / f'{case_id}-{name}.nii.gz' for name in MODALITIES + [SEG_SUFFIX]}

def build_case_manifest(data_root, case_ids):
    rows = []
    for case_id in sorted(case_ids):
        files = expected_case_files(case_id)
        row = {'case_id': case_id}
        for name, file_path in files.items():
            row[f'{name}_exists'] = file_path.exists()
        row['complete_images'] = all(row[f'{name}_exists'] for name in MODALITIES)
        row['has_segmentation'] = row['seg_exists']
        row['missing_files'] = ', '.join(name for name in files if not row[f'{name}_exists'])
        rows.append(row)
    return pd.DataFrame(rows)

case_manifest = build_case_manifest(DATA_ROOT, all_cases)
if case_manifest.empty:
    print('No cases found. Set BRATS_DATA_ROOT or DATA_ROOT to the BraTS training directory, then rerun this section.')
else:
    display(case_manifest.head())
    print(f'Cases: {len(case_manifest)} | Complete image sets: {case_manifest.complete_images.sum()} | With segmentation: {case_manifest.has_segmentation.sum()}')`),
  cell('code', `def nifti_basic_info(file_path):
    nii = nib.load(str(file_path))
    return {
        'file': str(file_path),
        'shape': tuple(nii.shape),
        'spacing_mm': tuple(round(float(x), 3) for x in nii.header.get_zooms()[:3]),
        'dtype': str(nii.get_data_dtype()),
        'affine_diag': tuple(round(float(x), 3) for x in np.diag(nii.affine)[:3]),
    }

valid_cases = case_manifest[case_manifest.complete_images].case_id.tolist() if not case_manifest.empty else []
sample_case_id = valid_cases[0] if valid_cases else None
if sample_case_id:
    print(f'Sample case: {sample_case_id}')
    display(pd.DataFrame([nifti_basic_info(path) for path in expected_case_files(sample_case_id).values() if path.exists()]))
else:
    print('No complete case is available for NIfTI inspection.')`),
  cell('code', `def describe_volume(file_path, label=False):
    data = nib.load(str(file_path)).get_fdata()
    nonzero = data[data != 0] if label is False else data
    return {
        'shape': tuple(data.shape),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(nonzero)) if nonzero.size else 0.0,
        'std': float(np.std(nonzero)) if nonzero.size else 0.0,
        'nonzero_voxels': int(np.count_nonzero(data)),
        'unique_labels': ', '.join(map(str, np.unique(data).astype(int))) if label else '',
    }

profiled_cases = valid_cases[:PROFILE_CASES]
image_rows, label_rows = [], []
for case_id in profiled_cases:
    files = expected_case_files(case_id)
    for modality in MODALITIES:
        image_rows.append({'case_id': case_id, 'modality': modality, **describe_volume(files[modality])})
    if files[SEG_SUFFIX].exists():
        label_rows.append({'case_id': case_id, **describe_volume(files[SEG_SUFFIX], label=True)})

image_profile_df = pd.DataFrame(image_rows)
label_profile_df = pd.DataFrame(label_rows)
if not image_profile_df.empty:
    display(image_profile_df)
    display(image_profile_df.groupby('modality')[['min', 'max', 'mean', 'std']].describe().round(3))
if not label_profile_df.empty:
    display(label_profile_df)
    display(label_profile_df[['case_id', 'unique_labels', 'nonzero_voxels']])`),
  cell('markdown', '## 5. Exploratory Data Analysis (EDA)\n\nThe plots below check dataset completeness, modality intensity ranges, label distribution, and a representative MRI slice with its tumor mask.'),
  cell('code', `if not case_manifest.empty:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    case_manifest[[f'{m}_exists' for m in MODALITIES]].sum().plot.bar(ax=axes[0], color='#4575b4', title='Available modality files')
    case_manifest['has_segmentation'].value_counts().sort_index().plot.bar(ax=axes[1], color=['#91bfdb', '#d73027'], title='Segmentation availability')
    missing = case_manifest['missing_files'].replace('', 'none').value_counts().head(10)
    missing.plot.bar(ax=axes[2], color='#74add1', title='Missing-file patterns')
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

if not image_profile_df.empty:
    image_profile_df.boxplot(column='mean', by='modality', ax=plt.subplots(figsize=(8, 4))[1], grid=False)
    plt.suptitle('Non-zero voxel mean by modality')
    plt.title('')
    plt.ylabel('Mean intensity')
    plt.show()

if not label_profile_df.empty:
    label_profile_df.plot.bar(x='case_id', y='nonzero_voxels', legend=False, figsize=(8, 4), title='Tumor voxels per profiled case')
    plt.ylabel('Voxel count')
    plt.tight_layout()
    plt.show()`),
  cell('code', `def choose_slice_index(volume, axis=2):
    foreground = np.argwhere(volume > 0)
    if foreground.size == 0:
        return volume.shape[axis] // 2
    return int(np.median(foreground[:, axis]))

def load_case_arrays(case_id):
    files = expected_case_files(case_id)
    images = [nib.load(str(files[m])).get_fdata().astype(np.float32) for m in MODALITIES]
    mask = nib.load(str(files[SEG_SUFFIX])).get_fdata() if files[SEG_SUFFIX].exists() else np.zeros_like(images[0])
    return images, mask

if sample_case_id:
    images, mask = load_case_arrays(sample_case_id)
    z = choose_slice_index(mask)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, image, modality in zip(axes[:4], images, MODALITIES):
        ax.imshow(image[:, :, z].T, cmap='gray', origin='lower')
        ax.set_title(modality.upper())
        ax.axis('off')
    axes[4].imshow(images[3][:, :, z].T, cmap='gray', origin='lower')
    axes[4].imshow(np.ma.masked_where(mask[:, :, z].T == 0, mask[:, :, z].T), cmap='turbo', alpha=0.55, origin='lower')
    axes[4].set_title('FLAIR + mask')
    axes[4].axis('off')
    plt.suptitle(f'EDA sample: {sample_case_id}, axial slice {z}')
    plt.tight_layout()
    plt.show()`),
  cell('markdown', '## 6. Data Preprocessing\n\nPreprocessing includes non-zero z-score normalization, label remapping, and center crop/pad to a stable tensor shape for model input.'),
  cell('code', `TARGET_SHAPE = (240, 240, 160)

def zscore_nonzero(volume):
    volume = volume.astype(np.float32, copy=True)
    mask = volume != 0
    if mask.any():
        values = volume[mask]
        volume[mask] = (values - values.mean()) / max(values.std(), 1e-8)
    return volume

def normalize_modalities(images):
    return np.stack([zscore_nonzero(image) for image in images], axis=0).astype(np.float32)

def center_crop_or_pad_3d(volume, target_shape, fill_value=0):
    result = np.full(target_shape, fill_value, dtype=volume.dtype)
    source_slices, target_slices = [], []
    for current, target in zip(volume.shape, target_shape):
        copy_size = min(current, target)
        source_start = (current - copy_size) // 2
        target_start = (target - copy_size) // 2
        source_slices.append(slice(source_start, source_start + copy_size))
        target_slices.append(slice(target_start, target_start + copy_size))
    result[tuple(target_slices)] = volume[tuple(source_slices)]
    return result

def remap_brats_labels(mask):
    mask = mask.astype(np.int16, copy=True)
    mask[mask == 4] = 3
    return mask

def preprocess_case(case_id, target_shape=TARGET_SHAPE):
    images, mask = load_case_arrays(case_id)
    reference = nib.load(str(expected_case_files(case_id)[MODALITIES[0]]))
    image = normalize_modalities(images)
    image = np.stack([center_crop_or_pad_3d(channel, target_shape) for channel in image], axis=0)
    mask = center_crop_or_pad_3d(remap_brats_labels(mask), target_shape)
    return {'case_id': case_id, 'image': image, 'mask': mask, 'spacing': reference.header.get_zooms()[:3], 'affine': reference.affine, 'original_shape': images[0].shape, 'target_shape': target_shape}

if sample_case_id:
    processed_sample = preprocess_case(sample_case_id)
    print(f"{sample_case_id}: images {processed_sample['image'].shape}, mask {processed_sample['mask'].shape}")
    print('Normalized image mean/std:', float(processed_sample['image'].mean()), float(processed_sample['image'].std()))
    print('Processed labels:', np.unique(processed_sample['mask']).tolist())`),
  cell('code', `if sample_case_id:
    z = choose_slice_index(processed_sample['mask'])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(images[3][:, :, choose_slice_index(mask)].T, cmap='gray', origin='lower')
    axes[0].set_title(f'Original FLAIR: {images[3].shape}')
    axes[1].imshow(processed_sample['image'][3, :, :, z].T, cmap='gray', origin='lower')
    axes[1].imshow(np.ma.masked_where(processed_sample['mask'][:, :, z].T == 0, processed_sample['mask'][:, :, z].T), cmap='turbo', alpha=0.55, origin='lower')
    axes[1].set_title(f'Preprocessed: {processed_sample["image"].shape[1:]}')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()`),
];

if (notebook.cells.some(c => (c.source || []).join('').includes('## 4. Data Understanding: Load, Info, and Describe'))) {
  notebook.cells.splice(7, 0, ...newCells);
  const install = notebook.cells.find(c => (c.source || []).join('').includes('!pip install'));
  if (install) install.source = install.source.map(line => line.replace('pyyaml', 'pyyaml pandas'));
  for (const c of notebook.cells) {
    c.source = (c.source || []).map(line => line.replace(/^## ([4-9])\./, (_, n) => `## ${Number(n) + 3}.`));
  }
  fs.writeFileSync(path, JSON.stringify(notebook, null, 2) + '\n');
  console.log(`Inserted ${newCells.length} cells.`);
} else {
  console.log('Data understanding cells already exist; no changes made.');
}
