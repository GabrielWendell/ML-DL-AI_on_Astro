# Mini-Project — Stellar Parameter Inference from Spectra

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Instructor:** Gabriel Wendell Celestino Rocha   
**Project type:** Regression / spectroscopy / 1D CNNs  
**Difficulty:** Intermediate–Advanced  

> **Scientific question:** How much information about stellar atmospheric parameters can be recovered from spectra, and which spectral regions drive the inference?

---

## 1. Scientific context

Stellar spectra encode effective temperature, surface gravity, chemical abundances, rotation, and
other physical information through continuum shape and absorption/emission lines. ML provides a useful
laboratory for comparing engineered spectral summaries with representation learning, but preprocessing,
resolution, normalization, and label quality can dominate performance.

## 2. Project objective

Develop a complete, reproducible machine-learning analysis that answers the scientific question above.
The project should be treated as a small research study: the goal is **not** merely to obtain a good
score, but to justify the representation, validation protocol, model family, metrics, and astrophysical
interpretation.

## 3. Dataset requirements

Use observed or synthetic spectra with labels such as effective temperature Teff, surface gravity log g,
and metallicity [Fe/H]. APOGEE, LAMOST, GALAH, SDSS/SEGUE, synthetic atmosphere grids, or a
course-provided subset are suitable.

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

1. Inspect wavelength coverage, spectral resolution, missing pixels, signal-to-noise, and target-parameter distributions.
2. Define and justify continuum normalization or flux scaling.
3. Place spectra on a common wavelength grid if required and propagate masks.
4. Train a compact baseline using PCA or selected line/continuum features plus a regressor.
5. Train a 1D CNN that operates directly on the spectrum.
6. Evaluate Teff, log g, and/or [Fe/H] separately rather than collapsing all errors into one number.
7. Analyze residuals versus signal-to-noise and stellar parameter regime.
8. Identify spectral regions that influence the prediction using ablation or attribution.
9. Discuss extrapolation: what happens near or outside the label-space boundaries?

## 6. Minimum modeling requirements

- PCA/features + classical regressor
- 1D convolutional neural network

The final comparison must use the **same frozen test set** for all models.

## 7. Evaluation

Recommended metrics/diagnostics:

- MAE per stellar parameter
- RMSE per parameter
- Bias
- Residual trends versus S/N and true parameter
- Coverage of uncertainty intervals if estimated

Do not report a single headline metric without a corresponding error analysis.

## 8. Required figures and tables

Your final notebook/report should contain at least:

- Representative normalized spectra
- Target-parameter distributions
- Predicted versus true parameters
- Residual versus S/N
- Residual versus true parameter
- Spectral-region importance / wavelength ablation

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

- Multi-task learning for Teff, log g, and [Fe/H].
- Transfer between synthetic and observed spectra.
- Include per-pixel uncertainties and masks.
- Estimate aleatoric uncertainty.
- Compare full-spectrum and line-window models.

## 11. Questions that must be answered in the conclusion

- What did the best model actually learn?
- Which examples are hardest, and why?
- What is the dominant limitation: data quantity, label quality, representation, domain shift, noise, or model capacity?
- Would you trust the model for the intended astrophysical use case?
- What additional observation or dataset would most improve the scientific conclusion?

---

## Starter workspace

The cells below are deliberately incomplete. Replace the placeholders with your own data and analysis.
