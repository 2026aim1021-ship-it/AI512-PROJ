# CMRF: Cross-Modal Residual Filtering for robust RGB-thermal segmentation

Team 41 course research project. We build on **CMX** (MiT-B2) for RGB-thermal semantic segmentation on **PST900**, and test whether a lightweight **cross-modal residual filtering (CMRF)** module improves robustness when one camera is degraded, without losing clean accuracy.

**Research question.** Can cross-modal feature prediction and selective residual filtering improve RGB-thermal segmentation robustness under modality-specific degradation, while preserving clean-input accuracy and limiting computational overhead?

CMRF is treated as a **research hypothesis**. Its effect must be shown through ablations, and its novelty checked against the literature.

## Experimental plan

| Exp | Description | Status |
|---|---|---|
| E1 | Original CMX baseline | ✅ done (100 epochs) |
| E2 | CMX + cross-modal feature prediction + residual gating (CMRF, stage 3) | ✅ first run done, see findings |
| E3 | E2 + modality-specific corruption during training | planned |
| E4 | E3 + clean-to-corrupted prediction consistency | planned |
| Control | E1 + the same corruption training, without CMRF | planned |

## Setup

- **Dataset:** PST900, official split, 597 train / 288 test RGB-thermal pairs at 1280×720, 5 classes (Background, Fire-Extinguisher, Backpack, Hand-Drill, Survivor). Not included here; download it from the [PST900 repository](https://github.com/ShreyasSkandanS/pst900_thermal_rgb).
- **Model:** the official [CMX code](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) (MiT-B2, ImageNet-pretrained), cloned and patched by the notebooks.
- **Training:** 100 epochs, batch size 4, AdamW, learning rate 6e-5, 480×640 crops, T4 GPU (about 4.8 h per run).
- **Evaluation:** sliding-window inference on the original 720×1280 labels. We report IoU, Dice, Sensitivity and Specificity per class, plus a robustness benchmark of 7 degradations × 3 severities applied to one camera at a time (RGB low-light, blur, noise, occlusion; thermal blur, noise, occlusion), and parameters, GFLOPs and latency.

## Results so far

| | E1 (CMX) | E2 (CMX + CMRF) |
|---|---|---|
| Clean mIoU, 5 classes | **86.12** | 82.92 |
| Clean mIoU, 4 objects | **82.76** | 78.77 |
| Survivor IoU | **78.78** | 72.43 |
| Average under RGB degradation | **67.42** | 63.90 |
| Average under thermal degradation | **85.53** | 82.49 |
| RGB low-light, severity 3 (object mIoU) | 19.57 (0.17) | 19.44 (0.00) |
| Parameters / GFLOPs | 66.56 M / 78.27 | 67.03 M / 78.83 (+0.7%) |

**E1 finding.** CMX is effectively RGB-dominated on PST900. Thermal damage costs at most 1.3 mIoU, while RGB damage costs up to 66.6. Under severe RGB low-light the model detects no objects, even though the thermal image is intact.

**E2 finding (first run).** CMRF lowered accuracy by about 3 points in every condition and did not change the robustness pattern. Diagnostics:

- The gate collapsed to a constant (about 0.5).
- The cross-modal residual was near zero even under RGB darkness.
- The prediction loss also back-propagated into the encoders.
- Implementation bug: CMRF's two strength parameters were not passed to the optimizer, because the repo's `group_weight` only collects standard layers.
- Clean-only training gives the model no incentive to use thermal.

**Next:** E2-v2 fixes these issues, followed by E3 with the control run.

## Repository layout

```
src/
  cmrf.py          CMRF module + patch that inserts it into the CMX code (stage set in cmrf_config.json)
  rgbt_eval.py     dataset, corruptions, metrics, sliding-window evaluation, robustness benchmark, cost
  cmx_colab.py     fixes for running the official CMX repo on Colab; data layout; config; model loading
notebooks/
  E1_cmx_baseline_colab.ipynb       E1 on Google Colab (resumable, skips finished training, saves model)
  E1_cmx_baseline_lightning.ipynb   E1 on Lightning AI
  E2_cmx_cmrf_colab.ipynb           E2 on Google Colab (E1 + CMRF; E1-vs-E2 tables, modality drop, gate maps)
  extras/                           validation-based selection, minimal local-GPU version, fast MiT-B0 variant
results/
  E1/   benchmark, per-class and robustness CSVs, qualitative figure, training log, executed notebook
  E2/   same, plus E1-vs-E2 tables, modality-drop test, CMRF residual/gate maps
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
