# MRIxFields Project: Provenance Ledger & Research Insights

This document serves as the absolute, immutable ledger for all research, empirical findings, logical questions, and extracted insights for the MRIxFields project.

## 1. The Provenance Ledger (Empirical Findings)

| The Empirical Finding / Metric | The exact Script/Notebook name that birthed it | The analytical deduction: What this specifically rules out or forces us to do next |
| :--- | :--- | :--- |
| **Highly Unpaired Distribution** (1971 singular scans vs 9 overlapping subjects) | `notebooks/N00/n00-path-exploration.ipynb` | Eliminates paired models (Pix2Pix); forces the architectural requirement to Unpaired Image-to-Image translation (CycleGAN/CUT). |
| **Standardized Z-Axis Spacing** (Flat 0.50mm across all field strengths) | `notebooks/N01/n01-extensive-eda.ipynb` | Eliminates the need for 3D spatial resampling/interpolation blocks in the preprocessing pipeline. |
| **Wild Intensity Variations** (T1W drops significantly in 5T/7T) | `notebooks/N01/n01-extensive-eda.ipynb` | Eliminates global Min-Max normalization; forces the implementation of volume-wise Z-score normalization or 99th-percentile clipping. |

## 2. Research & Extracted Insights
- **Medical Prior (CycleGAN vs CUT):** Literature on 1.5T to 7T translation confirms that while CycleGAN is foundational, Contrastive Unpaired Translation (CUT) yields higher structural fidelity. Cycle-consistency often hallucinates artifacts when mapping ultra-high-resolution (7T) back to low-resolution (1.5T). CUT utilizes patch-based contrastive loss to strictly preserve structural content without requiring the inverse mapping.
- **Lateral Prior (Extreme Dynamic Ranges):** Research in non-medical domains (e.g., HDR tone mapping, ultrasound beamforming) establishes that Adaptive Instance Normalization (AdaIN) or strict volume-wise Z-score normalization successfully mitigates discriminator collapse caused by extreme variance in raw pixel intensities.

## 3. Logical Questions & Next Steps
- **Falsifiable Hypothesis:** Implementing a 2D patch-based dataloader that applies strict volume-wise Z-score normalization will produce standardized tensor distributions (mean 0, std 1) across all field strengths, neutralizing the intensity mismatch observed in N01 before it reaches the generator.
- **Define Next Unknown (N02):** Can we build an empirical sandbox (minimal PyTorch DataLoader) that loads unpaired 2D slices from 1.5T and 7T `.nii` volumes, normalizes them dynamically, and mathematically verifies (via `tensor.mean()` and `tensor.std()`) that the field-strength intensity gap is closed without exhausting Kaggle RAM?
