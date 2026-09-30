# Naming reference

## Script names

| Earlier name | Current script |
|--------------|----------------|
| `Megocclusion_preprocessing.py` | `scripts/01_meg_preprocessing/preprocess_meg.py` |
| `Megocclusion_epochs_drop_log.py` | `scripts/01_meg_preprocessing/log_dropped_epochs.py` |
| `Megocclusion_preprocessed_stim_order.py` | `scripts/01_meg_preprocessing/build_trial_order.py` |
| `Megocclusion_build_forward_model.py` | `scripts/01_meg_preprocessing/build_forward_model.py` |
| `Megocclusion_inverse_operators.py` | `scripts/01_meg_preprocessing/compute_inverse_operators.py` |
| `Megocclusion_source_estimation_0nomask.py` | `scripts/02_meg_features/estimate_grand_average_source.py` |
| `Megocclusion_ROIs_extraction.py` | `scripts/02_meg_features/extract_roi_evoked.py` |
| `Megocclusion_MEG_pattern_extraction.py` | `scripts/02_meg_features/extract_meg_patterns.py` |
| `Megocclusion_RDM_extraction_8sub.py` ("8sub" = 8 trials per pseudo-trial) | `scripts/02_meg_features/extract_meg_rdms.py` |
| `Megocclusion_RDM_GC_8sub.py` | `scripts/02_meg_features/compute_rdm_granger.py` |
| `Megocclusion_{DFM,BLT,CORNet}_pattern_Extraction_ROIs.py` | `scripts/03_model_features/extract_model_patterns.py --model {htrn,blt,cornet}` |
| `Megocclusion_training_ResNet_DFM_imagenet.py` | `scripts/04_training/train_htrn_stage1.py` |
| `Megocclusion_training_ResNet_DFM_imagenet_fine_tune.py` | `scripts/04_training/train_htrn_finetune.py` |
| `Megocclusion_{DFM,BLT,CORNet}_stages.py` | `scripts/05_model_evaluation/run_stage_sweep.py --model {htrn,blt,cornet}` |
| `Megocclusion_HTRN_vs_Stock_eval.py` | `scripts/05_model_evaluation/compare_htrn_vs_resnet50.py` |
| `Megocclusion_decoding.py` | `scripts/06_meg_analysis/decode_categories.py` |
| `Megocclusion_decoding_cross_condition.py` | `scripts/06_meg_analysis/decode_cross_mask.py` |
| `Megocclusion_RSA_Subaveraged_{ResNet_FF_RT,BLT,CORNet}.py` | `scripts/06_meg_analysis/compute_rsa.py --model {htrn,blt,cornet}` |
| `Megocclusion_Plot_Decoding_v2.py` | `scripts/07_figures/plot_decoding.py` |
| `Megocclusion_Plot_RDM_GC_Direction_v2.py` | `scripts/07_figures/plot_granger_causality.py` |
| `Megocclusion_Plot_HTRN_Unrolling_v2.py` | `scripts/07_figures/plot_htrn_unrolling.py` |
| `Megocclusion_Plot_HTRN_StageSweep_v2.py` | `scripts/07_figures/plot_htrn_stage_sweep.py` |
| `Megocclusion_Plot_RSA_Mechanism_v2.py` (and `_BLT_v2`, `_CORNet_v2`) | `scripts/07_figures/plot_rsa_variants.py --model {htrn,blt,cornet}` |
| `Megocclusion_Plot_RSA_MaskEffect_v2.py` | `scripts/07_figures/plot_rsa_mask_effect.py` |
| `Megocclusion_Plot_Performance_Comparison_v2.py` | `scripts/07_figures/plot_model_performance.py` |
| `Megocclusion_plot_rois_activity_mask.py` | `scripts/07_figures/plot_roi_evoked.py` |
| `Megocclusion_plot_grand_average_stc_0nomask.py` | `scripts/07_figures/view_grand_average_source.py` |
| `megocclusion_figstyle.py` | `htrn/figstyle.py` |

## Model terms

The code base grew before the paper's terminology settled, so some identifiers on disk still use the
earlier names. They are kept so existing outputs load without renaming.

| Paper | In code and output files |
|-------|--------------------------|
| HTRN | "DFM" (Deep Feedback Model), class `ImageNetDFMResNet50` earlier, now `htrn.models.HTRN` |
| HTRN-FF / -LR / -TD | condition folders `Feedforward` / `Recurrent` / `TopDownFeedback`; RSA model names `DFM-Feedforward` / `DFM-Recurrent` / `DFM-TopDownFeedback`; stage-sweep rows `LR_0`, `LR_6`, `TD_7` (`mechanism`, `stage_idx` columns) |
| HTRN (fine-tuned) | `finetune_clean_checkpoint.pt`; stage-sweep model label `DFM-ResNet` |
| HTRN stage-1 checkpoint | `DFM_only_checkpoint.pt` |
| BLT-FF / -LR / -TD (BLT-VS B / BL / BLT) | `Feedforward` / `Recurrent` / `TopDownFeedback` in `Patterns_BLT_ROIs`, model label `BLT-VS` |
| CORnet-Z / -RT / -S | `CORNet-Z`, `CORNet-RT`, `CORNet-S` |

| Folder | Content |
|--------|---------|
| `Megocclusion/derivatives/Patterns_DFM_ROIs` | HTRN ROI patterns |
| `Megocclusion/derivatives/RSA_Subaveraged_Results_ResNet` | HTRN model-brain RSA |
| `results_resnet_dfm_stages_multi_occ_train` | HTRN stage sweep |
