# Week 2 Wednesday — Representations: Spectra and Time Series

This folder contains the Wednesday hands-on class for the mini-course
**Machine Learning and Artificial Intelligence for Modern Astrophysics**.

## Scientific theme

Monday changed the model while keeping a tabular representation fixed. Wednesday asks a different
question:

> How much performance comes from the representation itself?

## Slot 1 — Stellar spectra

- synthetic normalized spectra with controlled line physics and nuisance variables;
- hand-designed line indices + logistic regression;
- PCA coefficients + logistic regression;
- raw spectrum + 1D CNN;
- wavelength masking as a direct representation ablation.

## Slot 2 — Irregular light curves

- irregular quasi-periodic and aperiodic/red-noise-like time series;
- time-domain summary features + random forest;
- Lomb–Scargle frequency representation + logistic regression;
- interpolated raw sequence + 1D CNN;
- student perturbation challenge.

## Files

- `wednesday_representations_instructor.ipynb`
- `wednesday_representations_student.ipynb`
- `wednesday_representations.md`
- `requirements.txt`

The datasets are generated locally so the class runs without internet access.
