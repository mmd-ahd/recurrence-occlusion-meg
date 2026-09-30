# recurrence-occlusion-meg

Code for **"Local recurrence accounts for extended processing during occluded-object recognition"**
(Ahadzadeh, Rajaei & Soltanian-Zadeh; [bioRxiv 2026.09.10.750662](https://www.biorxiv.org/content/10.64898/2026.09.10.750662v1)).

The study combines source-localized MEG, time-resolved decoding, backward masking, representational
Granger causality and computational modelling to ask whether the extra processing needed to recognize
occluded objects relies on **local recurrence** or on **long-range top-down feedback**. This repository
contains the full analysis pipeline: MEG preprocessing and source estimation, decoding and
representational Granger causality, the **Hierarchical Top-down Recurrent Network (HTRN)** and its
training, model-brain RSA against HTRN, BLT-VS and CORnet, and the scripts that draw every figure.

## The HTRN model

HTRN is a ResNet-50 with an added 256-D top-down feedback loop (built on the
[Deep Feedback Model](THIRD_PARTY_NOTICES.md)). The three variants compared in the paper are
**readout stages of one fine-tuned network**, not separate models:

| Variant | Readout | What runs |
|---------|---------|-----------|
| HTRN-FF | stage 0 | shortcut projections only: a feedforward sweep, no residual blocks, no feedback |
| HTRN-LR | stage 6 | all residual blocks active (local recurrence), no feedback |
| HTRN-TD | stage 7 | as HTRN-LR, plus accumulated top-down feedback from layer 4 to the network entry |

Stage `s` of `htrn.models.generate_stage_schedule` activates one more block per layer until all
`[3, 4, 6, 3]` blocks are active. `scripts/07_figures/plot_htrn_unrolling.py` draws the construction (Fig. 4A-B).

## Repository layout

```
htrn/                    shared package
  config.py              paths (HTRN_DATA_ROOT), subjects, categories, ROI definitions
  models.py              HTRN, checkpoint loading, readout stages
  data.py, svm.py        stimulus dataset and the four-way multi-occlusion SVM readout
  blt.py, cornet_variants.py   BLT-VS and CORnet conditions
  roi_patterns.py, rsa.py      model ROI patterns and sub-averaged RSA
  meg.py, decoding.py    MEG helpers and decoding utilities
  imagenet.py, gpu.py    ImageNet pipeline, GPU helpers
  figstyle.py            shared figure style, statistics helpers and drawing functions
dfbmodels/  cornet/  BLT-VS-main/    third-party models (see THIRD_PARTY_NOTICES.md)
scripts/
  01_meg_preprocessing/  ICA + cleaning, drop log, trial order, forward model, inverse operators
  02_meg_features/       ROI evoked responses, MEG patterns, MEG RDMs, representational Granger causality
  03_model_features/     PCA-reduced ROI patterns of HTRN / BLT-VS / CORnet
  04_training/           HTRN training on ImageNet (stage 1: feedback only, stage 2: fine-tuning)
  05_model_evaluation/   readout-stage sweeps, HTRN vs. ResNet-50 on ImageNet
  06_meg_analysis/       category decoding, cross-mask decoding, model-brain RSA
  07_figures/            one script per figure
data/                    human accuracy (Rajaei et al., 2019) used as a reference in the figures
docs/naming.md           old script and folder names -> current names
```

## Installation

```bash
git clone https://github.com/mmd-ahd/recurrence-occlusion-meg.git
cd recurrence-occlusion-meg
conda env create -f environment.yml   # or: pip install -e .
conda activate recurrence-occlusion-meg
```

Python 3.10+ with PyTorch (a GPU is only needed for model feature extraction and training) and
MNE-Python. The analyses were run with Python 3.12, PyTorch 2.14, MNE 1.13 and scikit-learn 1.9.
The 3D viewer (`view_grand_average_source.py`) additionally needs `pip install pyvista pyvistaqt`.

## Data and checkpoints

The MEG-Occlusion dataset (Rajaei & Khaligh-Razavi, 2019, *Megocclusion-vr3*, RepOD, [doi:10.18150/repod.2004402](https://doi.org/10.18150/repod.2004402)) and the trained checkpoints are not part of this repository (see `.gitignore`). Set `HTRN_DATA_ROOT` to a folder containing:

| Path | Content |
|------|---------|
| `Megocclusion/subject<N>.mat` | Epoched MEG of the MEG-Occlusion dataset (Rajaei et al., 2019), subject 11 excluded |
| `Megocclusion/image_sequence_order.mat` | Presented image order (Rajaei et al., 2019) |
| `ImageFiles/<category>/level_<00,60,80>/*.png` | The 768 stimuli (camel, deer, car, motor x 3 occlusion levels x 64) |
| `subj04NN_sess01-0_tsss.fif` | Any subject's raw file; only the sensor layout is read |
| `finetune_clean_checkpoint.pt` | The fine-tuned HTRN used in the paper |
| `DFM_only_checkpoint.pt` | Stage-1 (feedback-only) checkpoint |

`HTRN_DATA_ROOT` defaults to the repository root. BLT-VS and CORnet weights are downloaded on first use
(Hugging Face hub and the CORnet model zoo). ImageNet-1k is needed only for training and for
`compare_htrn_vs_resnet50.py`.

## Running the pipeline

Scripts default to the paper settings; subject, condition and path options are listed by `--help`, and the
scripts that cover several models take a required `--model`. Commands are run from the repository root.

| Step | Command | Output |
|------|---------|--------|
| 1 | `python scripts/01_meg_preprocessing/preprocess_meg.py` | cleaned epochs `Megocclusion/sub-XX/meg/` |
| 2 | `python scripts/01_meg_preprocessing/log_dropped_epochs.py` then `build_trial_order.py` | trial-order JSON |
| 3 | `python scripts/01_meg_preprocessing/build_forward_model.py` then `compute_inverse_operators.py` | inverse operators |
| 4 | `python scripts/02_meg_features/extract_roi_evoked.py`, `extract_meg_patterns.py`, `extract_meg_rdms.py`, `compute_rdm_granger.py` | ROI evoked responses, patterns, RDMs, Granger causality |
| 5 | `python scripts/06_meg_analysis/decode_categories.py`, `decode_cross_mask.py` | decoding scores |
| 6 | `python scripts/04_training/train_htrn_stage1.py` then `train_htrn_finetune.py` | HTRN checkpoints (optional if you use the released ones) |
| 7 | `python scripts/03_model_features/extract_model_patterns.py --model {htrn,blt,cornet}` | model ROI patterns |
| 8 | `python scripts/06_meg_analysis/compute_rsa.py --model {htrn,blt,cornet}` | model-brain RSA time courses |
| 9 | `python scripts/05_model_evaluation/run_stage_sweep.py --model {htrn,blt,cornet}` | four-way accuracy per readout stage |
| 10 | `python scripts/05_model_evaluation/compare_htrn_vs_resnet50.py` | ImageNet comparison (McNemar, bootstrap) |
| 11 | `python scripts/07_figures/plot_*.py` | figures |

## Figures

| Figure | Script |
|--------|--------|
| Fig. 1B, Fig. 2 (decoding, masking generalization) | `plot_decoding.py` |
| Fig. 3B (representational Granger causality), Supp. Figs. 2-4 | `plot_granger_causality.py` |
| Fig. 4A-B (HTRN readout stages) | `plot_htrn_unrolling.py` |
| Fig. 4C (accuracy per readout stage) | `plot_htrn_stage_sweep.py` |
| Fig. 5 (model-brain RSA, HTRN) | `plot_rsa_variants.py --model htrn` |
| Supp. Fig. 1 (ROI evoked responses) | `plot_roi_evoked.py` |
| Supp. Fig. 5 (accuracy of HTRN, BLT-VS, CORnet) | `plot_model_performance.py` |
| Supp. Figs. 6-7 (masking effect on RSA) | `plot_rsa_mask_effect.py` |
| Supp. Figs. 8-9 (BLT-VS RSA) | `plot_rsa_variants.py --model blt` |
| Supp. Figs. 10-11 (CORnet RSA) | `plot_rsa_variants.py --model cornet` |

All figure scripts are in `scripts/07_figures/`. The ImageNet comparison reported in the text
(top-1 77.03% to 78.86%) comes from `compare_htrn_vs_resnet50.py`.

## Reproducibility notes

- Subject 11 is excluded (noisy recording); the ICA components removed per subject are listed in
  `preprocess_meg.py` and `Megocclusion/Excluded Components.txt`.
- Model readouts use fixed seeds (`random_state=715`). The feedback state starts from random noise, so
  `extract_model_patterns.py` reseeds before each condition.
- Decoding pseudo-trials (`decode_categories.py`, `decode_cross_mask.py`) use NumPy's global RNG without
  a seed, so decoding scores vary slightly between runs.
- `compute_rsa.py` recomputes a result whenever its inputs are newer and skips it otherwise; use `--overwrite`
  to force a recomputation. Cluster-permutation p-values in the figure scripts can differ in the fourth
  decimal between runs.
- Result folders and file names produced by earlier runs keep their original names (for example
  `Patterns_DFM_ROIs`, `RSA_Subaveraged_Results_ResNet`, `results_resnet_dfm_stages_multi_occ_train` and the
  `DFM-*` model labels) so existing outputs stay usable; `docs/naming.md` maps them to the HTRN terms.
- Model features, SVM readout results, RSA time courses, Granger-causality traces and all figure scripts
  were checked against the pre-release code and give identical outputs on test data.

## Citation

```bibtex
@article{ahadzadeh2026local,
  title   = {Local recurrence accounts for extended processing during occluded-object recognition},
  author  = {Ahadzadeh, Mohammad and Rajaei, Karim and Soltanian-Zadeh, Hamid},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.10.750662}
}
```

See also `CITATION.cff`. The MEG-Occlusion dataset and paradigm are described in Rajaei et al. (2019),
*Beyond core object recognition: Recurrent processes account for object recognition under occlusion*,
PLoS Computational Biology 15(5): e1007001.

## License

MIT for the code in this repository (see `LICENSE`). Third-party components keep their own licenses;
see `THIRD_PARTY_NOTICES.md`.

## Contact

Karim Rajaei (rajaei.k@ipm.ir)
