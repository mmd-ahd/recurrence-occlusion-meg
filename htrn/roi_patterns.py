"""Model ROI patterns for RSA: activations per stimulus, PCA per ROI, and per-subject assembly.

Every unique stimulus image is passed through the model once. Activations of the layers assigned
to an ROI are flattened, concatenated and reduced with PCA (99% variance, fitted on all stimuli),
and the resulting rows are then looked up in each subject's trial order to build the per-subject
pattern matrices that the RSA step compares against MEG.
"""
import glob
import json
import logging
import os

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

from htrn.config import CATEGORIES, DATASET_DIR, STIMULI_DIR, SUBJECT_IDS

log = logging.getLogger(__name__)

SEQUENCE_FILE = os.path.join(DATASET_DIR, 'preprocessed_image_sequence_order.json')
PCA_TARGET_VARIANCE = 0.99
RANDOM_SEED = 715
OCCLUSION_LEVELS = ['0', '60', '80']
MASK_CONDITIONS = ['nomask', 'mask']


def load_sequence_data(path=SEQUENCE_FILE):
    """Per-subject trial order: ``{sub: {'<mask>-<level>-<category>': [image ids]}}``."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Trial-order file not found: {path}')
    with open(path) as f:
        return json.load(f)


def discover_stimuli(stimuli_dir=STIMULI_DIR, occlusion_levels=OCCLUSION_LEVELS):
    """Every stimulus on disk as ``(category, level, image_id, path)``.

    The same 768 images (4 categories x 3 levels x 64) are shown to all subjects; mask/no-mask only
    changes trial bookkeeping, so each image needs to go through the model only once.
    """
    stimuli = []
    for cat in CATEGORIES:
        for level in occlusion_levels:
            folder = os.path.join(stimuli_dir, cat, f'level_{level.zfill(2)}')
            files = sorted(glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg')))
            for path in files:
                try:
                    uid = int(os.path.basename(path).split('_', 1)[0])
                except ValueError:
                    continue
                stimuli.append((cat, level, uid, path))
    return stimuli


def roi_features(activations, areas):
    """Concatenate the flattened activations of ``areas`` into a ``(batch, features)`` array."""
    parts = []
    for name in areas:
        feat = activations[name].detach().cpu()
        parts.append(feat.reshape(feat.shape[0], -1).numpy())
    return np.concatenate(parts, axis=1)


def build_stimulus_cache(activation_fn, roi_areas, transform, device, batch_size,
                         stimuli_dir=STIMULI_DIR, occlusion_levels=OCCLUSION_LEVELS):
    """PCA-reduced ROI features of every stimulus.

    Args:
        activation_fn: Maps a batch of images to ``{area_name: activation tensor}``.
        roi_areas: ``{roi_name: [area_name, ...]}`` selecting the layers pooled into each ROI.
        transform: Image preprocessing applied to each stimulus.
        device: Device the batches are moved to.
        batch_size: Number of images per forward pass.

    Returns:
        ``cache[roi][(category, level, image_id)] -> PCA-transformed feature row``. Each ROI gets
        its own PCA fitted on all stimuli.
    """
    stimuli = discover_stimuli(stimuli_dir, occlusion_levels)
    log.info('%d unique stimulus images', len(stimuli))
    keys = [(cat, level, uid) for cat, level, uid, _ in stimuli]
    paths = [s[3] for s in stimuli]

    raw_data = {roi: [] for roi in roi_areas}
    for i in range(0, len(paths), batch_size):
        tensors = [transform(Image.open(p).convert('RGB')) for p in paths[i:i + batch_size]]
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            activations = activation_fn(batch)
        for roi, areas in roi_areas.items():
            raw_data[roi].append(roi_features(activations, areas))

    cache = {}
    for roi in roi_areas:
        X_raw = np.vstack(raw_data[roi]).astype(np.float32, copy=False)
        pca = PCA(n_components=PCA_TARGET_VARIANCE, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_raw)
        log.info('PCA %s: %d components from %d stimuli', roi, pca.n_components_, X_raw.shape[0])
        cache[roi] = dict(zip(keys, X_pca))
    return cache


def save_subject_patterns(cache, sequence_data, output_dir, condition_subdir,
                          subject_ids=SUBJECT_IDS, mask_conditions=MASK_CONDITIONS,
                          occlusion_levels=OCCLUSION_LEVELS):
    """Write one feature matrix per subject, mask condition, occlusion level and ROI.

    Rows follow the subject's trial order (categories concatenated). A subject/level combination is
    skipped if any trial's image is missing from the cache.

    Files: ``<output_dir>/sub-XX/<condition_subdir>/sub-XX_occlusion[-mask]-<level>_AllCat_ROI-<roi>_t0_features.npy``
    """
    for sub_id in subject_ids:
        sub_str = f'sub-{sub_id:02d}'
        cond_out_dir = os.path.join(output_dir, sub_str, condition_subdir)
        os.makedirs(cond_out_dir, exist_ok=True)

        for mask_cond in mask_conditions:
            for level in occlusion_levels:
                trials = []
                for cat in CATEGORIES:
                    key = f'{mask_cond}-{level}-{cat}'
                    if sub_str in sequence_data and key in sequence_data[sub_str]:
                        trials.extend((cat, uid) for uid in sequence_data[sub_str][key])
                if not trials:
                    continue
                if not all((cat, level, uid) in cache[roi] for cat, uid in trials for roi in cache):
                    continue

                prefix = f'occlusion-{level}' if mask_cond == 'nomask' else f'occlusion-{mask_cond}-{level}'
                for roi in cache:
                    matrix = np.stack([cache[roi][(cat, level, uid)] for cat, uid in trials], axis=0)
                    np.save(os.path.join(cond_out_dir, f'{sub_str}_{prefix}_AllCat_ROI-{roi}_t0_features.npy'), matrix)
