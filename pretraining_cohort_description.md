# ViT Pretraining Cohort Description (7,509 CT Volumes)

This document describes the unlabeled CT cohort used for masked-autoencoder
(MAE) pretraining of the 3D Vision Transformer (ViT) in our work
*"Pretrained Anatomical Prior-Guided Hierarchical Regularization for
Label-Efficient Multicenter 3D CT Body Composition Segmentation."*

## Overview

- **Total volumes:** 7,509 unlabeled CT scans
- **Sources:** 20 datasets / collections spanning public repositories and
  hospital cohorts
- **Purpose:** self-supervised ViT pretraining only (no segmentation labels)
- **Pretraining configuration:** MAE, 2,000 epochs, mask ratio 0.75,
  patch size 16, batch size 4, AdamW optimizer
- **Input resolution:** patches of 96 × 96 × 96 voxels

## Cohort Composition

| Dataset | Scans | Size (GB) | Voxel spacing (mm) | Slice thickness (mm) |
| :--- | ---: | ---: | :--- | :--- |
| GDPH | 1,287 | 112.4 | 0.64–0.95 | 1.25 |
| Multi | 294 | 30.0 | 0.61–0.98 | 0.3–0.5 |
| Lung1 | 421 | 13.9 | 0.97 | 3.0 |
| Argos | 664 | 18.6 | 0.96–1.36 | 2.0–5.0 |
| Intertobs | 21 | 1.1 | 0.97 | 5.0 |
| MSD | 63 | 6.1 | 0.60–0.98 | 0.63–2.5 |
| Rider | 31 | 3.0 | 0.51–0.90 | 1.25 |
| CHSUMC | 211 | 5.5 | 0.83–1.37 | 3.0–5.0 |
| Sheng | 791 | 17.2 | 0.59–0.98 | 2.0–6.4 |
| AN | 159 | 3.1 | 0.63–0.97 | 5.0–6.9 |
| BO YI | 131 | 2.4 | 0.63–0.94 | 0.63–8.0 |
| Radio | 144 | 17.7 | 0.58–0.97 | 0.65–2.5 |
| Shengyi | 296 | 45.1 | 0.47–0.92 | 1.0 |
| Xiehe | 388 | 37.7 | 0.55–0.97 | 0.69–2.0 |
| Yunnan | 198 | 17.5 | 0.58–0.99 | 0.75–2.0 |
| Yichang | 16 | 1.3 | 0.58–0.76 | 0.5–1.3 |
| Zhejiang | 338 | 53.5 | 0.31–0.97 | 0.7–1.25 |
| Zhengzhou | 386 | 47.6 | 0.64–0.98 | 0.4–6.54 |
| CT-Covid19 (public) | 650 | 12.7 | 0.30–1.10 | 0.3–5.0 |
| TCIA-LIDC (public) | 1,018 | 75.2 | 0.46–0.97 | 0.45–5.0 |
| **Total** | **7,509** | | | |


## Data Hygiene and Leakage Prevention

- The cohort was deduplicated at the **patient and examination level**, so no
  patient contributes more than one series per examination.
- **No patient, examination, or reconstructed series from any of the three
  development centers (Center1–Center3) or the three external evaluation
  centers (Center4–Center6) was included in pretraining.**
- Consequently, the pretrained representations were never exposed to the
  data used for segmentation training, validation, or testing.

## Acquisition Characteristics

- **Voxel spacing:** 0.31–1.36 mm (in-plane)
- **Slice thickness:** 0.3–8.0 mm
- **Scanners and protocols:** mixed vendors, reconstruction kernels, and
  acquisition protocols across sources
- **Contrast:** both contrast-enhanced and non-enhanced examinations
- **Anatomical coverage:** chest, abdomen, and pelvis, with per-dataset
  variation

## Preprocessing for Pretraining

1. Load DICOM volumes and sort slices into 3D stacks.
2. Resample / crop patches to 96 × 96 × 96 voxels.
3. Intensity normalization consistent with the downstream nnU-Net pipeline.
4. Patchify into 6 × 6 × 6 tokens (patch size 16) for the ViT.
5. Randomly mask 75% of patches for MAE reconstruction.

## Licensing and Attribution

- **CT-Covid19** and **TCIA-LIDC** are publicly available collections; use
  of these data follows their respective licenses and requires citation of
  the original publications in derivative work.
- Hospital cohorts were used under institutional approval for research
  purposes only and are not redistributed.

## Citation

If you use this pretraining data description, please cite the main paper and
the public datasets (CT-Covid19, LIDC-IDRI) referenced above.
