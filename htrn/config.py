"""Paths and constants shared across the pipeline.

Set ``HTRN_DATA_ROOT`` to the folder that holds ``Megocclusion/`` (MEG data and derivatives),
``ImageFiles/`` (stimuli) and the ``*.pt`` checkpoints. It defaults to the repository root.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('HTRN_DATA_ROOT', REPO_ROOT))

DATASET_DIR = DATA_ROOT / 'Megocclusion'
DERIVATIVES_DIR = DATASET_DIR / 'derivatives'
STIMULI_DIR = DATA_ROOT / 'ImageFiles'
BLT_VS_DIR = REPO_ROOT / 'BLT-VS-main'

# Human accuracy from Rajaei et al. (2019); small enough to ship with the code
HUMAN_ACCURACY_CSV = REPO_ROOT / 'data' / 'Human_behavioral_performance.csv'

# The fine-tuned HTRN used throughout the paper (HTRN-FF / -LR / -TD are readout stages of it).
HTRN_CHECKPOINT = DATA_ROOT / 'finetune_clean_checkpoint.pt'

# Raw file used only to read the sensor layout (any subject's recording works)
SAMPLE_RAW = DATA_ROOT / 'subj04NN_sess01-0_tsss.fif'

# Subject 11 is excluded (noisy recording).
SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
CATEGORIES = ['camel', 'deer', 'car', 'motor']

# Minimum-norm source estimation
INVERSE_LAMBDA2 = 1.0 / 3.0 ** 2
INVERSE_METHOD = 'MNE'

# ROI groups merged from HCP-MMP1 parcels (coarse ventral-stream ROIs used in the paper)
ROIS_TO_MERGE = {
    'V1-3': ['V1', 'V2', 'V3'],
    'LOC': ['LO1', 'LO2', 'LO3', 'V4t'],
    'IT-PHC': ['PHA1', 'PHA2', 'PHA3', 'VMV2', 'VMV3', 'VVC', 'FFC', 'TE1p', 'TE2p'],
}
