"""Build the fsaverage oct6 MEG forward model shared by all subjects.

Uses the fsaverage template head (source space, BEM and coordinate transform) with the sensor
layout of ``SAMPLE_RAW``. Output: ``Megocclusion/fwd/fsaverage-meg-oct6-fwd.fif``

Usage:
    python scripts/01_meg_preprocessing/build_forward_model.py
"""
import logging
import os

import mne

from htrn import setup_logging
from htrn.config import DATASET_DIR
from htrn.meg import load_sample_info

log = logging.getLogger(__name__)


def main():
    """Build and save the forward solution."""
    setup_logging()
    output_dir = os.path.join(DATASET_DIR, 'fwd')
    os.makedirs(output_dir, exist_ok=True)
    fwd_fname = os.path.join(output_dir, 'fsaverage-meg-oct6-fwd.fif')

    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    src = mne.setup_source_space(subject='fsaverage', spacing='oct6',
                                 subjects_dir=subjects_dir, add_dist=False)
    mne.add_source_space_distances(src, dist_limit=0.04, n_jobs=-1)

    bem = mne.make_bem_solution(mne.make_bem_model(subject='fsaverage', subjects_dir=subjects_dir))
    trans = os.path.join(str(subjects_dir), 'fsaverage', 'bem', 'fsaverage-trans.fif')

    fwd = mne.make_forward_solution(load_sample_info(), trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, n_jobs=-1)
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
    log.info('Saved %s', fwd_fname)


if __name__ == '__main__':
    main()
