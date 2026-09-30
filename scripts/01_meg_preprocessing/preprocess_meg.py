"""Clean the raw MEG epochs: ICA artifact removal, filtering and amplitude rejection.

For each subject, the per-condition epochs stored in ``subject<N>.mat`` are loaded, an extended
infomax ICA is fitted on the no-mask epochs only (1-100 Hz), the subject's hand-picked artifact
components are removed from all conditions, the data are band-passed to 1-45 Hz and bad epochs are
dropped by amplitude. Output:
``Megocclusion/sub-XX/meg/sub-XX_task-objectrecognition_occlusion-<mask>-<level>-<category>_meg.fif``

Usage:
    python scripts/01_meg_preprocessing/preprocess_meg.py
"""
import gc
import logging
import os

import h5py
import mne
import numpy as np
from mne.preprocessing import ICA

from htrn import setup_logging
from htrn.config import DATASET_DIR
from htrn.meg import EVENT_ID_MAP, load_sample_info

log = logging.getLogger(__name__)

TMIN = -0.2
REJECT = dict(mag=4e-12, grad=4000e-13)

# ICA components removed per subject (also listed in Megocclusion/Excluded Components.txt)
EXCLUDED_COMPONENTS = {
    1: [11, 17, 22, 32, 33, 37],
    2: [20, 22, 23, 26, 28, 30],
    3: [11, 19, 22, 39],
    4: [15, 18, 20, 36],
    5: [0, 1, 4, 8, 13, 15, 16, 17, 18, 22, 27, 28, 33, 36],
    6: [13, 19, 20, 22, 33, 35],
    7: [6, 29, 38, 42, 43, 44],
    8: [14, 16, 23, 24, 26, 29, 30],
    9: [4, 17, 18, 29, 35],
    10: [1, 6, 20, 21, 30, 31, 43],
    12: [5, 6, 16, 18, 22, 25, 26, 27, 29, 30, 31, 37],
    13: [21, 26, 34, 35],
    14: [10, 13, 16, 32, 34, 40, 41],
    15: [8, 22, 24, 29, 34],
}


def load_epochs(mat_path, info):
    """Read every condition of a subject's .mat file into an ``mne.EpochsArray``."""
    all_epochs = {}
    with h5py.File(mat_path, 'r') as f:
        meg_signals_ref = f['Data']['MEG_Signals']
        for condition, event_ids in EVENT_ID_MAP.items():
            data = [f[meg_signals_ref[cond, 0]][()]
                    for cond in event_ids if cond < meg_signals_ref.shape[0]]
            if not data:
                continue

            data_for_epochs = np.vstack(data).transpose(0, 2, 1)
            event_marker = event_ids[0]
            events = np.array([[i, 0, event_marker] for i in range(data_for_epochs.shape[0])])
            all_epochs[condition] = mne.EpochsArray(
                data_for_epochs, info, events=events, tmin=TMIN, event_id={condition: event_marker})
            del data, data_for_epochs, events
            gc.collect()
    return all_epochs


def main():
    """Clean and save the epochs of every subject."""
    setup_logging()
    info = load_sample_info()

    for sub_id, excluded in EXCLUDED_COMPONENTS.items():
        sub_bids_id = f'sub-{sub_id:02d}'
        mat_path = os.path.join(DATASET_DIR, f'subject{sub_id}.mat')
        if not os.path.isfile(mat_path):
            log.warning('Skipping %s: %s not found', sub_bids_id, mat_path)
            continue

        log.info('%s: loading epochs', sub_bids_id)
        all_epochs = load_epochs(mat_path, info)

        # ICA is fitted on no-mask data only and then applied to every condition
        nomask = [ep for cond, ep in all_epochs.items() if 'nomask' in cond]
        if not nomask:
            log.warning('%s: no no-mask data, skipping ICA', sub_bids_id)
            continue

        ica_epochs = mne.concatenate_epochs(nomask)
        ica_epochs.filter(l_freq=1, h_freq=100, method='iir', n_jobs=-1, verbose=False)
        ica = ICA(n_components=0.95, method='infomax', fit_params=dict(extended=True), random_state=715)
        ica.fit(ica_epochs, verbose=False)
        ica.exclude = excluded
        log.info('%s: ICA fitted, excluding components %s', sub_bids_id, ica.exclude)
        del nomask, ica_epochs
        gc.collect()

        out_dir = os.path.join(DATASET_DIR, sub_bids_id, 'meg')
        os.makedirs(out_dir, exist_ok=True)

        for condition, epochs in all_epochs.items():
            cleaned = epochs.copy().filter(l_freq=1, h_freq=45, method='iir', n_jobs=-1, verbose=False)
            ica.apply(cleaned, exclude=ica.exclude)
            cleaned.drop_bad(reject=REJECT, verbose=False)
            cleaned.save(os.path.join(
                out_dir, f'{sub_bids_id}_task-objectrecognition_occlusion-{condition}_meg.fif'),
                overwrite=True, verbose=False)
            del cleaned
            gc.collect()

        del all_epochs, ica
        gc.collect()


if __name__ == '__main__':
    main()
