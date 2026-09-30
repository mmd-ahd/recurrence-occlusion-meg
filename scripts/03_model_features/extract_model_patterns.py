"""Extract PCA-reduced model ROI patterns for RSA against the MEG data.

Each stimulus image goes through the model once per condition; layer activations are grouped into
the ROIs V1_3 / LOC / IT, reduced with PCA, and written per subject in trial order to
``Megocclusion/derivatives/Patterns_<name>_ROIs/<condition>/``.

Conditions (paper names in brackets):
    htrn    Feedforward [HTRN-FF], Recurrent [HTRN-LR], TopDownFeedback [HTRN-TD]
    blt     Feedforward [BLT-FF],  Recurrent [BLT-LR],  TopDownFeedback [BLT-TD]
    cornet  CORNet-Z: Feedforward; CORNet-RT and CORNet-S: Feedforward, Recurrent

Usage:
    python scripts/03_model_features/extract_model_patterns.py --model htrn
"""
import argparse
import logging
import os

import torch

from htrn import setup_logging
from htrn.config import DERIVATIVES_DIR, HTRN_CHECKPOINT
from htrn.data import imagenet_transform
from htrn.gpu import find_max_batch_size
from htrn.roi_patterns import (RANDOM_SEED, build_stimulus_cache, load_sequence_data,
                               save_subject_patterns)

log = logging.getLogger(__name__)

# HTRN readout stage per condition (see htrn.models); stage 6 is the first with all blocks active.
HTRN_CONDITIONS = [
    {'name': 'Feedforward',     'target_stage': 0, 'use_topdown': False},
    {'name': 'Recurrent',       'target_stage': 6, 'use_topdown': False},
    {'name': 'TopDownFeedback', 'target_stage': 7, 'use_topdown': True},
]
HTRN_ROI_AREAS = {'V1_3': ['layer1', 'layer2'], 'LOC': ['layer3'], 'IT': ['layer4']}
BLT_ROI_AREAS = {'V1_3': ['V1', 'V2', 'V3'], 'LOC': ['V4'], 'IT': ['LOC']}
CORNET_ROI_AREAS = {'V1_3': ['V1', 'V2'], 'LOC': ['V4'], 'IT': ['IT']}


def htrn_jobs(device, checkpoint):
    """One extraction job per HTRN condition (see ``main`` for the job fields)."""
    from htrn.models import generate_stage_schedule, layer_blocks, load_htrn, stage_outputs

    model = load_htrn(checkpoint, device=device)
    layers = layer_blocks(model)
    schedule = generate_stage_schedule()

    def make_fn(cond):
        def fn(batch):
            outs = stage_outputs(model, layers, batch, schedule, cond['target_stage'],
                                 cond['use_topdown'])
            return dict(zip(['layer1', 'layer2', 'layer3', 'layer4'], outs))
        return fn

    probe = make_fn(HTRN_CONDITIONS[-1])
    for cond in HTRN_CONDITIONS:
        yield {'subdir': cond['name'], 'fn': make_fn(cond), 'roi_areas': HTRN_ROI_AREAS,
               'transform': imagenet_transform(), 'probe': probe, 'probe_key': 'htrn',
               'candidates': None, 'seed': RANDOM_SEED}


def blt_jobs(device, checkpoint):
    """Extraction jobs for the three BLT-VS conditions."""
    from htrn.blt import (ALL_BLT_AREAS, BLT_CONDITIONS, build_all_blt_models, extract_blt_areas,
                          get_blt_vs_transform)

    models = build_all_blt_models(device)

    def make_fn(cond):
        model = models[cond['name']]
        return lambda batch: extract_blt_areas(model, batch, cond['timestep'], ALL_BLT_AREAS)

    probe = make_fn(BLT_CONDITIONS[-1])
    for cond in BLT_CONDITIONS:
        yield {'subdir': cond['name'], 'fn': make_fn(cond), 'roi_areas': BLT_ROI_AREAS,
               'transform': get_blt_vs_transform(), 'probe': probe, 'probe_key': 'blt',
               'candidates': (64, 48, 32, 16, 8, 4, 2, 1), 'seed': None}


def cornet_jobs(device, checkpoint):
    """Extraction jobs for every CORnet variant and condition."""
    from htrn.cornet_variants import CORNET_CONDITIONS, extract_areas, load_cornet_variant

    for variant, conditions in CORNET_CONDITIONS.items():
        model = load_cornet_variant(variant, device)

        def make_fn(cond, variant=variant, model=model):
            return lambda batch: extract_areas(variant, model, batch, cond)

        probe = make_fn(conditions[-1])  # most expensive condition of this variant
        for cond in conditions:
            yield {'subdir': os.path.join(variant, cond['name']), 'fn': make_fn(cond),
                   'roi_areas': CORNET_ROI_AREAS, 'transform': imagenet_transform(),
                   'probe': probe, 'probe_key': variant, 'candidates': None, 'seed': None}


MODELS = {
    'htrn': (htrn_jobs, 'Patterns_DFM_ROIs'),  # folder name kept from earlier result files
    'blt': (blt_jobs, 'Patterns_BLT_ROIs'),
    'cornet': (cornet_jobs, 'Patterns_CORNet_ROIs'),
}


def main():
    """Extract and save the ROI patterns of the chosen model."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--checkpoint', default=str(HTRN_CHECKPOINT), help='HTRN checkpoint (htrn only)')
    args = parser.parse_args()

    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    make_jobs, folder = MODELS[args.model]
    output_dir = os.path.join(DERIVATIVES_DIR, folder)
    sequence_data = load_sequence_data()

    batch_sizes = {}  # one memory probe per model (or CORNet variant)
    for job in make_jobs(device, args.checkpoint):
        log.info('Condition: %s', job['subdir'])
        key = job['probe_key']
        if key not in batch_sizes:
            kwargs = {'candidates': job['candidates']} if job['candidates'] else {}
            batch_sizes[key] = find_max_batch_size(job['probe'], device, **kwargs)
        batch_size = batch_sizes[key]

        if job['seed'] is not None:
            # The feedback state starts from unseeded noise; reseed so each condition is reproducible.
            torch.manual_seed(job['seed'])

        cache = build_stimulus_cache(job['fn'], job['roi_areas'], job['transform'], device, batch_size)
        save_subject_patterns(cache, sequence_data, output_dir, job['subdir'])


if __name__ == '__main__':
    main()
