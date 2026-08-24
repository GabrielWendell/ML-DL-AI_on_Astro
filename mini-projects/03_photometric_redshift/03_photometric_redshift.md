# Mini-Project — Photometric Redshift Estimation with Machine Learning

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Instructor:** Gabriel Wendell Celestino Rocha    
**Project type:** Regression / tabular ML  
**Difficulty:** Intermediate  

> **Scientific question:** How accurately can redshift be inferred from broadband photometry, where do catastrophic failures occur, and how should uncertainty and selection effects be reported?

---

## 1. Scientific context

Spectroscopic redshifts are expensive, while photometric surveys provide enormous samples. Machine
learning can approximate the mapping from colors and magnitudes to redshift, but the mapping is
degenerate and strongly dependent on the training-set selection function. Catastrophic outliers are
scientifically important and should not be hidden by a single global RMSE.

## 2. Project objective

Develop a complete, reproducible machine-learning analysis that answers the scientific question above.
The project should be treated as a small research study: the goal is **not** merely to obtain a good
score, but to justify the representation, validation protocol, model family, metrics, and astrophysical
interpretation.

## 3. Dataset requirements

Use a catalog containing broadband magnitudes or fluxes and reliable spectroscopic redshifts. A subset
from SDSS, DES, HSC, KiDS, COSMOS, or a course-provided matched catalog is suitable. Preserve
photometric errors and any survey/quality flags when possible.

Before modeling, document:

- source and provenance of the data;
- number of physical objects;
- available observables and labels/targets;
- missing-data and quality flags;
- important selection effects;
- whether multiple surveys, fields, instruments, or observing conditions are mixed.

## 4. Learning objectives

By completing this project, you should be able to:

- translate an astrophysical question into a well-defined ML task;
- construct a leakage-safe validation strategy;
- establish a simple baseline before using a more complex model;
- evaluate performance with metrics appropriate to the scientific use case;
- diagnose failure modes and observational shortcuts;
- communicate what the model does **and does not** establish scientifically.

## 5. Required workflow

1. Audit redshift coverage, missing photometry, magnitude limits, photometric uncertainties, and selection effects.
2. Construct colors and/or normalized flux features while documenting transformations.
3. Create train/validation/test partitions and visualize whether their redshift and magnitude distributions are comparable.
4. Train a simple linear or nearest-neighbor baseline.
5. Train at least one nonlinear tree/boosting model.
6. Train an MLP regressor or another neural model.
7. Evaluate residuals as a function of true redshift and apparent magnitude.
8. Quantify catastrophic outliers rather than reporting only mean error.
9. Estimate predictive uncertainty or construct empirical prediction intervals.
10. Explain where the color–redshift mapping becomes degenerate.

## 6. Minimum modeling requirements

- One simple regression baseline
- One nonlinear classical model
- One neural regressor

The final comparison must use the **same frozen test set** for all models.

## 7. Evaluation

Recommended metrics/diagnostics:

- MAE
- RMSE
- Bias in $Δz/(1+z)$
- NMAD
- Catastrophic outlier fraction
- Residual trends versus redshift/magnitude

Do not report a single headline metric without a corresponding error analysis.

## 8. Required figures and tables

Your final notebook/report should contain at least:

- Photometric color distributions
- Predicted versus spectroscopic redshift
- Residual versus true redshift
- Residual distribution
- Outlier fraction versus magnitude/redshift
- Optional uncertainty calibration plot

Also provide one compact table containing the principal models, representations, hyperparameters, and
test-set metrics.

## 9. Reproducibility checklist

- Record the random seed and exact train/validation/test object IDs.
- Do not tune on the test set.
- Keep preprocessing fitted on the training set only.
- Save model/configuration choices and package versions.
- Generate all final plots and tables from code rather than manual editing.
- Document negative results and failed experiments.

## 10. Advanced extensions

These are optional. Choose one only after the required analysis is complete.

- Compare magnitudes, colors, and flux-ratio representations.
- Use quantile regression or probabilistic neural networks.
- Train on a bright sample and test on a fainter sample.
- Compare survey-specific and pooled models.
- Investigate domain adaptation or importance weighting.

## 11. Questions that must be answered in the conclusion

- What did the best model actually learn?
- Which examples are hardest, and why?
- What is the dominant limitation: data quantity, label quality, representation, domain shift, noise, or model capacity?
- Would you trust the model for the intended astrophysical use case?
- What additional observation or dataset would most improve the scientific conclusion?

---

## Starter workspace

The cells below are deliberately incomplete. Replace the placeholders with your own data and analysis.
