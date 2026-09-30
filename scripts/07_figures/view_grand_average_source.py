"""Interactive 3D view of the grand-average source estimate (0% occlusion, no mask).

Opens the estimate written by ``scripts/02_meg_features/estimate_grand_average_source.py`` on the
fsaverage cortex (PyVista backend).

Usage:
    python scripts/07_figures/view_grand_average_source.py
"""
import os

import mne

from htrn.config import DERIVATIVES_DIR


def main():
    """Open the interactive viewer."""
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'
    stc = mne.read_source_estimate(
        os.path.join(DERIVATIVES_DIR, 'grand_average_stc', 'grand-average_occlusion-0-nomask_meg'),
        subject='fsaverage')

    mne.viz.set_3d_backend('pyvistaqt')
    stc.plot(
        subject='fsaverage',
        subjects_dir=subjects_dir,
        hemi='both',
        smoothing_steps=10,
        size=(1000, 500),
        clim='auto',
        time_viewer=True,
    )
    input('Press Enter to close the viewer...')


if __name__ == '__main__':
    main()
