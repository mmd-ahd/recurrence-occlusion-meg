"""BLT-VS (Bottom-up, Lateral, Top-down) model conditions.

The ``bu=``/``td=`` forward arguments of ``BLT_VS`` only select which activations are returned;
they do not switch pathways off. Mechanisms are therefore isolated by building three ``BLT_VS``
instances with different constructor flags, all loaded from the same ecoset checkpoint:

* BLT-FF: no lateral, no top-down, read out at t=4 (when LOC first receives the input).
* BLT-LR: lateral connections, no top-down, read out at t=11.
* BLT-TD: lateral and top-down connections, read out at t=11.

Note that BLT-VS's own area called ``LOC`` is its IT-equivalent top area, not the lateral
occipital ROI of the MEG analysis.
"""
import logging
import sys

import torch
from huggingface_hub import hf_hub_download

from htrn.config import BLT_VS_DIR

sys.path.insert(0, str(BLT_VS_DIR))
from blt_vs_model.blt_vs import BLT_VS, NoOpModule  # noqa: E402
from blt_vs_model.transforms import get_blt_vs_transform  # noqa: E402,F401

log = logging.getLogger(__name__)

ECOSET_CHECKPOINT_FILENAME = 'blt_vs_slt_111_biounroll_1_t_12_readout_multi_dataset_ecoset_num_1.pth'
TIMESTEPS = 12
BLT_IT_EQUIVALENT_AREA = 'LOC'

ALL_BLT_AREAS = ['V1', 'V2', 'V3', 'V4', BLT_IT_EQUIVALENT_AREA]

BLT_CONDITIONS = [
    {'name': 'Feedforward',     'lateral': False, 'topdown': False, 'timestep': 4},
    {'name': 'Recurrent',       'lateral': True,  'topdown': False, 'timestep': 11},
    {'name': 'TopDownFeedback', 'lateral': True,  'topdown': True,  'timestep': 11},
]


def build_blt_model(device, lateral, topdown, state_dict, timesteps=TIMESTEPS):
    """Construct one BLT-VS variant and load ``state_dict`` non-strictly (missing keys expected)."""
    model = BLT_VS(timesteps=timesteps, num_classes=565, add_feats=100,
                   lateral_connections=lateral, topdown_connections=topdown,
                   skip_connections=True, bio_unroll=True, image_size=224,
                   hook_type='None', readout_type='multi')
    if not topdown:
        # The V4->V1 skip shares a flag with the V1->V4 skip; disable only its top-down half.
        model.connections['V1'].skip_td_depthwise = NoOpModule()
        model.connections['V1'].skip_td_pointwise = NoOpModule()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.debug('lateral=%s topdown=%s: %d missing, %d unexpected keys',
              lateral, topdown, len(missing), len(unexpected))
    return model.to(device).eval()


def load_ecoset_state_dict(device):
    """Download (once) and load the ecoset-trained BLT-VS weights from the Hugging Face hub."""
    weight_path = hf_hub_download(repo_id='novelmartis/blt_vs_model',
                                  filename=ECOSET_CHECKPOINT_FILENAME)
    return torch.load(weight_path, map_location=device)


def build_all_blt_models(device):
    """Return ``{condition_name: model}`` for the three BLT-VS conditions."""
    state_dict = load_ecoset_state_dict(device)
    return {
        cond['name']: build_blt_model(device, cond['lateral'], cond['topdown'], state_dict)
        for cond in BLT_CONDITIONS
    }


def extract_blt_areas(model, images, timestep, areas):
    """Unpooled bottom-up activation maps ``{area: tensor}`` at ``timestep``."""
    with torch.no_grad():
        _, activations = model(images, extract_actvs=True, areas=areas,
                               timesteps=[timestep], bu=True, td=False, concat=False)
    return {area: activations[area][timestep]['bu'] for area in areas}
