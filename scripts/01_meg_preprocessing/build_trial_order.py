"""Align the presented image order with the retained MEG trials of each subject.

The .mat file holds the planned image sequence per subject and condition. This script truncates it
to the number of recorded trials, removes the epochs rejected during preprocessing (see
``log_dropped_epochs.py``) and writes ``Megocclusion/preprocessed_image_sequence_order.json``:
``{sub: {'<mask>-<level>-<category>': [image ids in trial order]}}``.

Usage:
    python scripts/01_meg_preprocessing/build_trial_order.py
"""
import json
import logging
import os

import numpy as np
import scipy.io

from htrn import setup_logging
from htrn.config import CATEGORIES, DATASET_DIR, SUBJECT_IDS

log = logging.getLogger(__name__)

# Column of image_order.mat per condition: six columns per category in the order
# [0 mask, 0 no-mask, 60 mask, 60 no-mask, 80 mask, 80 no-mask].
IMAGE_ORDER_COLUMNS = {
    f'{mask}-{level}-{cat}': 6 * c + 2 * l + m
    for c, cat in enumerate(CATEGORIES)
    for l, level in enumerate(['0', '60', '80'])
    for m, mask in enumerate(['mask', 'nomask'])
}


def main():
    """Write the aligned image order of all subjects."""
    setup_logging()
    mat_file_path = os.path.join(DATASET_DIR, 'image_sequence_order.mat')
    drop_db_path = os.path.join(DATASET_DIR, 'rejected_indices_database.json')
    output_json_path = os.path.join(DATASET_DIR, 'preprocessed_image_sequence_order.json')

    image_order_matrix = scipy.io.loadmat(mat_file_path)['image_order']
    with open(drop_db_path) as f:
        drop_db = json.load(f)

    final_sequences = {}
    for sub in SUBJECT_IDS:
        sub_key = f'sub-{sub:02d}'
        row_idx = sub - 1  # subject rows in the .mat file are 0-based

        if sub_key not in drop_db:
            log.warning('%s: not in the drop database, skipping', sub_key)
            continue
        if row_idx >= image_order_matrix.shape[0]:
            log.warning('%s: row %d out of bounds in the .mat file, skipping', sub_key, row_idx)
            continue

        final_sequences[sub_key] = {}
        for condition, col_idx in IMAGE_ORDER_COLUMNS.items():
            if condition not in drop_db[sub_key]:
                continue

            try:
                sequence = np.squeeze(image_order_matrix[row_idx, col_idx])
            except IndexError:
                log.error('%s: cannot access column %d', sub_key, col_idx)
                continue

            stats = drop_db[sub_key][condition]

            # The recording may have stopped early: truncate the planned sequence to the raw count
            if stats['original_count'] < len(sequence):
                sequence = sequence[:stats['original_count']]

            # Remove the trials rejected during preprocessing
            valid_drops = [i for i in stats['dropped_indices'] if i < len(sequence)]
            clean_seq = np.delete(sequence, valid_drops)

            if len(clean_seq) > stats['retained_count']:
                log.info('%s %s: truncating %d -> %d', sub_key, condition,
                         len(clean_seq), stats['retained_count'])
                clean_seq = clean_seq[:stats['retained_count']]
            if len(clean_seq) != stats['retained_count']:
                log.warning('%s %s: image/MEG trial mismatch (%d vs %d)', sub_key, condition,
                            len(clean_seq), stats['retained_count'])

            final_sequences[sub_key][condition] = [int(x) for x in clean_seq]

    with open(output_json_path, 'w') as f:
        json.dump(final_sequences, f, indent=4)


if __name__ == '__main__':
    main()
