"""Record which epochs were rejected during preprocessing.

Reads the drop log of every cleaned epochs file and writes
``Megocclusion/rejected_indices_database.json`` with, per subject and condition, the dropped trial
indices and the raw / retained epoch counts. ``build_trial_order.py`` uses it to align the image
order with the retained MEG trials.

Usage:
    python scripts/01_meg_preprocessing/log_dropped_epochs.py
"""
import json
import logging
import os

import mne
import numpy as np

from htrn import setup_logging
from htrn.config import DATASET_DIR, SUBJECT_IDS
from htrn.meg import EVENT_ID_MAP, epochs_path

log = logging.getLogger(__name__)


def main():
    """Scan all epoch files and write the drop database."""
    setup_logging()
    output_path = os.path.join(DATASET_DIR, 'rejected_indices_database.json')

    database = {}
    for sub in SUBJECT_IDS:
        sub_str = f'sub-{sub:02d}'
        database[sub_str] = {}

        for condition in EVENT_ID_MAP:
            fname = epochs_path(sub_str, condition)
            if not os.path.exists(fname):
                log.warning('Missing %s', fname)
                continue

            try:
                epochs = mne.read_epochs(fname, preload=False, verbose=False)
            except Exception as exc:
                log.error('%s %s: %s', sub_str, condition, exc)
                continue

            # drop_log has one entry per original epoch; non-empty entries mark rejected epochs
            dropped = np.where([len(entry) > 0 for entry in epochs.drop_log])[0].tolist()
            original_count = len(epochs.drop_log)
            database[sub_str][condition] = {
                'dropped_indices': dropped,
                'retained_count': len(epochs),
                'original_count': original_count,
                'drop_percentage': round(len(dropped) / original_count * 100, 2) if original_count > 0 else 0,
            }

    with open(output_path, 'w') as f:
        json.dump(database, f, indent=4)
    log.info('Saved %s', output_path)


if __name__ == '__main__':
    main()
