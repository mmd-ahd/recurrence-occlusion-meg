"""Model-brain RSA time courses (Fig. 5, Supp. Figs. 6-11).

Compares each model condition's ROI patterns with MEG ROI patterns over time, per subject, mask
condition and occlusion level (sub-averaged RDMs, Spearman correlation; see ``htrn.rsa``).

Inputs (from steps 02 and 03):
    Megocclusion/derivatives/Patterns_MEG/
    Megocclusion/derivatives/Patterns_{DFM,BLT,CORNet}_ROIs/
Output: Megocclusion/derivatives/RSA_Subaveraged_Results_{ResNet,BLT,CORNet}/

Existing outputs are recomputed if any input is newer; use --overwrite to force it.

Usage:
    python scripts/06_meg_analysis/compute_rsa.py --model htrn
"""
import argparse
import json
import logging
import os
from multiprocessing import Pool

from htrn import setup_logging
from htrn.config import DERIVATIVES_DIR, SUBJECT_IDS
from htrn.roi_patterns import MASK_CONDITIONS, OCCLUSION_LEVELS, SEQUENCE_FILE
from htrn.rsa import process_subject

log = logging.getLogger(__name__)

# model -> (pattern folder, output folder, [(pattern subdir, model name in output files)])
# Folder and model names match the files already produced by earlier runs.
MODELS = {
    'htrn': ('Patterns_DFM_ROIs', 'RSA_Subaveraged_Results_ResNet', [
        ('Feedforward', 'DFM-Feedforward'),
        ('Recurrent', 'DFM-Recurrent'),
        ('TopDownFeedback', 'DFM-TopDownFeedback'),
    ]),
    'blt': ('Patterns_BLT_ROIs', 'RSA_Subaveraged_Results_BLT', [
        ('Feedforward', 'BLT-Feedforward'),
        ('Recurrent', 'BLT-Recurrent'),
        ('TopDownFeedback', 'BLT-TopDownFeedback'),
    ]),
    'cornet': ('Patterns_CORNet_ROIs', 'RSA_Subaveraged_Results_CORNet', [
        ('CORNet-Z/Feedforward', 'CORNet-Z-Feedforward'),
        ('CORNet-RT/Feedforward', 'CORNet-RT-Feedforward'),
        ('CORNet-RT/Recurrent', 'CORNet-RT-Recurrent'),
        ('CORNet-S/Feedforward', 'CORNet-S-Feedforward'),
        ('CORNet-S/Recurrent', 'CORNet-S-Recurrent'),
    ]),
}


def main():
    """Compute the RSA time courses of the chosen model for all subjects."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--subjects', type=int, nargs='+', default=SUBJECT_IDS)
    parser.add_argument('--overwrite', action='store_true', help='recompute even if outputs are up to date')
    args = parser.parse_args()

    setup_logging()
    pattern_folder, output_folder, conditions = MODELS[args.model]
    source_dir = str(DERIVATIVES_DIR / pattern_folder)
    cfg = {
        'meg_dir': str(DERIVATIVES_DIR / 'Patterns_MEG'),
        'output_dir': str(DERIVATIVES_DIR / output_folder),
        'sequence_file': SEQUENCE_FILE,
        'mask_conditions': MASK_CONDITIONS,
        'occlusion_levels': OCCLUSION_LEVELS,
        'conditions': [{'subdir': subdir, 'source_dir': source_dir, 'model_name': name}
                       for subdir, name in conditions],
        'overwrite': args.overwrite,
    }
    os.makedirs(cfg['output_dir'], exist_ok=True)

    with open(SEQUENCE_FILE) as f:
        sequence_data = json.load(f)

    n_workers = min(len(args.subjects), os.cpu_count())
    with Pool(n_workers) as pool:
        pool.map(process_subject, [(sub, sequence_data, cfg) for sub in args.subjects])


if __name__ == '__main__':
    main()
