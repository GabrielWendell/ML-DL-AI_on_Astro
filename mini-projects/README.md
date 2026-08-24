# Mini-Projects

This directory contains the seven final mini-project briefs for **Machine Learning and Artificial Intelligence for Modern Astrophysics**.

Each project is provided in two formats:

- a **Jupyter Notebook (`.ipynb`)** intended to be copied and completed by students;
- a **Markdown (`.md`)** version for direct browsing on GitHub.

The notebooks are intentionally **unsolved scaffolds**. They define the scientific question, required
workflow, minimum models, validation requirements, figures/tables, deliverables, reproducibility
standards, and optional advanced extensions.

## Project menu

| # | Mini-project | Main ML setting | Difficulty |
|---:|---|---|---|
| 1 | [Variable-Star Classification from Astronomical Light Curves](01_variable_star_classification/01_variable_star_classification.md) | Classification / time-series learning | Intermediate |
| 2 | [Galaxy Morphology Classification with Convolutional Neural Networks](02_galaxy_morphology/02_galaxy_morphology.md) | Image classification / CNNs | Intermediate |
| 3 | [Photometric Redshift Estimation with Machine Learning](03_photometric_redshift/03_photometric_redshift.md) | Regression / tabular ML | Intermediate |
| 4 | [Stellar Parameter Inference from Spectra](04_stellar_spectra_parameters/04_stellar_spectra_parameters.md) | Regression / spectroscopy / 1D CNNs | Intermediate–Advanced |
| 5 | [Exoplanet Transit Candidate Classification](05_exoplanet_transit_classification/05_exoplanet_transit_classification.md) | Rare-event classification / time series | Intermediate–Advanced |
| 6 | [Anomaly Detection in Astronomical Data](06_astrophysical_anomaly_detection/06_astrophysical_anomaly_detection.md) | Unsupervised / semi-supervised learning | Advanced |
| 7 | [Inferring Circumstellar-Disk Geometry from Stellar Polarimetry](07_polarimetry_disk_geometry/07_polarimetry_disk_geometry.md) | Regression/classification / polarimetry / multimodal ML | Intermediate–Advanced |
| 8 | [Physics-Informed Neural Networks for Stellar Structure](08_pinn_stellar_structure.md) | Physics-Informed Neural Networks (PINNs) / Scientific Machine Learning | Intermediate–Advanced |

## Common scientific workflow

**Astrophysical question → dataset audit → validation design → baseline → main model → quantitative evaluation → failure analysis → astrophysical interpretation.**

The projects are designed so that students are rewarded for methodological rigor and scientific
interpretation rather than for using the most complex architecture.

## Suggested layout

```text
student_project/
├── README.md
├── analysis.ipynb
├── requirements.txt
├── data/
├── src/
└── results/
    ├── figures/
    └── tables/
```

Large datasets and trained checkpoints should normally be excluded from Git.
