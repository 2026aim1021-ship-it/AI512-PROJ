# CMRF: Cross-Modal Residual Filtering for robust RGB-thermal segmentation

Team 41 course research project. We build on **CMX** (MiT-B2) for RGB-thermal semantic segmentation on **PST900**, and test whether a lightweight **cross-modal residual filtering (CMRF)** module improves robustness when one camera is degraded, without losing clean accuracy.

**Research question.** Can cross-modal feature prediction and selective residual filtering improve RGB-thermal segmentation robustness under modality-specific degradation, while preserving clean-input accuracy and limiting computational overhead?

CMRF is treated as a **research hypothesis**. Its effect must be shown through ablations, and its novelty checked against the literature.

## Experimental plan

| Exp | Description | Status |
|---|---|---|
| E1 | Original CMX baseline | ✅ done (100 epochs) |
| E2 | CMX + cross-modal feature prediction + residual gating (CMRF, stage 3) | ✅ first run done, see findings |
| E2-v2 | E2 with the bug fixes (resumable notebook, held-out validation split) | ✅ done, see findings |
| E3 | E2 + modality-specific corruption during training | queued |
| E4 | E3 + clean-to-corrupted prediction consistency | queued |
| Control | E1 + the same corruption training, without CMRF | queued |
| E1 / E2-v2 re-runs | same as above on the corrected pipeline, for like-for-like comparison | queued |

## Setup

- **Dataset:** PST900, official split, 597 train / 288 test RGB-thermal pairs at 1280×720, 5 classes (Background, Fire-Extinguisher, Backpack, Hand-Drill, Survivor). Not included here; download it from the [PST900 repository](https://github.com/ShreyasSkandanS/pst900_thermal_rgb).
- **Model:** the official [CMX code](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) (MiT-B2, ImageNet-pretrained), cloned and patched by the notebooks.
- **Training:** 100 epochs, batch size 4, AdamW, learning rate 6e-5, 480×640 crops, T4 GPU (about 4.8 h per run).
- **Evaluation:** sliding-window inference on the original 720×1280 labels. We report IoU, Dice, Sensitivity and Specificity per class, plus a robustness benchmark of 7 degradations × 3 severities applied to one camera at a time (RGB low-light, blur, noise, occlusion; thermal blur, noise, occlusion), and parameters, GFLOPs and latency.

## Results so far

| | E1 (CMX) | E2 (CMX + CMRF) | E2-v2 (fixed CMRF) |
|---|---|---|---|
| Clean mIoU, 5 classes | **86.12** | 82.92 | 85.34 |
| Clean mIoU, 4 objects | **82.76** | 78.77 | 81.78 |
| Survivor IoU | **78.78** | 72.43 | 72.46 |
| Average under RGB degradation | **67.42** | 63.90 | 60.40 |
| Average under thermal degradation | **85.53** | 82.49 | 83.39 |
| RGB low-light, severity 3 (object mIoU) | 19.57 (0.17) | 19.44 (0.00) | 19.44 (0.00) |
| Parameters / GFLOPs | 66.56 M / 78.27 | 67.03 M / 78.83 (+0.7%) | 67.03 M / 78.83 |

E1 and E2 used the official `train.py` on all 597 training images. E2-v2 used our resumable engine with a held-out validation split (491 train / 106 val, by recording) and best-validation-loss model selection, so it is not yet a like-for-like comparison (see below).

**E1 finding.** CMX is effectively RGB-dominated on PST900. Thermal damage costs at most 1.3 mIoU, while RGB damage costs up to 66.6. Under severe RGB low-light the model detects no objects, even though the thermal image is intact.

**E2 finding (first run).** CMRF lowered accuracy by about 3 points in every condition and did not change the robustness pattern. Diagnostics:

- The gate collapsed to a constant (about 0.5).
- The cross-modal residual was near zero even under RGB darkness.
- The prediction loss also back-propagated into the encoders.
- Implementation bug: CMRF's two strength parameters were not passed to the optimizer, because the repo's `group_weight` only collects standard layers.
- Clean-only training gives the model no incentive to use thermal.

**E2-v2 finding.** The fixes worked as intended, but CMRF still adds no robustness:

- Clean accuracy recovered to within 0.8 mIoU of E1 (from −3.2), and the training loss matches E1 (0.0086 vs 0.009).
- The CMRF scale parameters are now trained (0.100 → 0.0925), and the thermal residual maps highlight the survivor under RGB darkness.
- The gate is still constant at 0.50: with clean-only training nothing rewards using thermal. Thermal-only input still detects no objects.
- New failure under RGB occlusion (51.4 vs 72.9 for E1): black patches are predicted as Backpack/Hand-Drill. Likely cause: our training loader padded crops with black pixels (ignored in the loss), while the official loader pads with the normalised mean (grey). Fixed in the queue notebook; to be confirmed by the E1 re-run on the same pipeline.

**Next (one notebook, run in sequence):** E1 re-run, E3, Control, E2-v2 re-run and E4, all with the corrected loader, the same validation split and the same training engine.

## Repository layout

```
src/
  cmrf.py          CMRF module + patch that inserts it into the CMX code (stage set in cmrf_config.json)
  rgbt_eval.py     dataset, corruptions, metrics, sliding-window evaluation, robustness benchmark, cost
  cmx_colab.py     fixes for running the official CMX repo on Colab; data layout; config; model loading
  cmrf_engine.py   CMRF v2, resumable training engine (last/best/every-10-epoch checkpoints), E4 consistency loss, run evaluation
notebooks/
  E1_cmx_baseline_colab.ipynb       E1 on Google Colab (resumable, skips finished training, saves model)
  E1_cmx_baseline_lightning.ipynb   E1 on Lightning AI
  E2_cmx_cmrf_colab.ipynb           E2 on Google Colab (E1 + CMRF; E1-vs-E2 tables, modality drop, gate maps)
  E2v2_cmx_cmrf_colab.ipynb         E2-v2 (also runs E1 / E3 / Control): stop and resume any time, best model by validation loss
  CMRF_queue_E1_E3_control_E2v2_E4_colab.ipynb   runs E1, E3, Control, E2-v2 and E4 back to back (corrected loader),
                                    evaluates each and builds the comparison tables; resumable at any point
  extras/                           validation-based selection, minimal local-GPU version, fast MiT-B0 variant
results/
  E1/   benchmark, per-class and robustness CSVs, qualitative figure, training log, executed notebook
  E2/   same, plus E1-vs-E2 tables, modality-drop test, CMRF residual/gate maps
  E2v2/ E2-v2 benchmark, curves, gate maps, training log, validation split, executed notebook
docs/
  E2_CMRF_rationale.pdf   what changed from E1 and why; evidence chain; design choices
  figures/                diagrams used in the rationale document
```

The notebooks are self-contained: each writes its own copy of the helper modules. The files in `src/` are the reference versions.

## How to run (Colab)

1. Open a notebook from `notebooks/` in Google Colab, then choose **Runtime → Change runtime type → T4 GPU**.
2. Put `PST900_RGBT_Dataset.zip` in `MyDrive/AIProj_CMX/datasets/`. The MiT-B2 weights download automatically.
3. Choose **Run all**. Checkpoints and results are saved to Google Drive. After a disconnect, choose Run all again and training resumes.

## Related work reviewed

- CMX: [arXiv 2203.04838](https://arxiv.org/abs/2203.04838)
- Complementary Random Masking (CRM): [arXiv 2303.17386](https://arxiv.org/abs/2303.17386)
- RSFNet: [arXiv 2306.10364](https://arxiv.org/abs/2306.10364)
- UTFNet: [IEEE GRSL 2023](https://doi.org/10.1109/LGRS.2023.3322452) (full text still to be reviewed)
