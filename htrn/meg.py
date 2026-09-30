"""MEG helpers: condition names, sensor info, epoch file names and merged HCP-MMP1 ROI labels."""
import logging
import os

import mne

from htrn.config import CATEGORIES, DATASET_DIR, ROIS_TO_MERGE, SAMPLE_RAW

log = logging.getLogger(__name__)

# Condition name -> [index into the .mat file's MEG_Signals], ordered by category, mask, level.
EVENT_ID_MAP = {
    f'{mask}-{level}-{cat}': [6 * c + 3 * m + l]
    for c, cat in enumerate(CATEGORIES)
    for m, mask in enumerate(['mask', 'nomask'])
    for l, level in enumerate(['0', '60', '80'])
}


def load_sample_info(path=SAMPLE_RAW):
    """MEG-only ``mne.Info`` (no bad channels) read from a raw recording."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Info source file not found at '{path}'")
    raw = mne.io.read_raw_fif(path, preload=False, on_split_missing='ignore')
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, meg=True, eeg=False))
    info['bads'] = []
    raw.close()
    return info


def epochs_path(sub_str, condition, dataset_dir=DATASET_DIR):
    """Cleaned epochs file of one subject and condition (e.g. ``'nomask-60-camel'``)."""
    return os.path.join(dataset_dir, sub_str, 'meg',
                        f'{sub_str}_task-objectrecognition_occlusion-{condition}_meg.fif')


def merged_roi_labels(per_hemisphere=False):
    """HCP-MMP1 parcels merged into the coarse ROIs of ``htrn.config.ROIS_TO_MERGE``.

    Args:
        per_hemisphere: If True, return one label per ROI and hemisphere, named
            ``'<ROI>-lh'`` / ``'<ROI>-rh'``; otherwise one bi-hemispheric label per ROI.

    Returns:
        ``{label name: mne.Label}``.
    """
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    try:
        mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
    except Exception as exc:
        log.warning('Could not fetch the HCP-MMP1 parcellation (%s); using the local copy', exc)
    all_labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

    merged = {}
    for name, parts in ROIS_TO_MERGE.items():
        for hemi in (['lh', 'rh'] if per_hemisphere else [None]):
            labels = [lbl for lbl in all_labels
                      if any(f'_{p}_' in lbl.name for p in parts) and (hemi is None or lbl.hemi == hemi)]
            if not labels:
                continue
            label = labels[0]
            for lbl in labels[1:]:
                label += lbl
            label.name = f'{name}-{hemi}' if hemi else name
            merged[label.name] = label
    return merged
