# Mini-Project — Variable-Star Classification from Astronomical Light Curves

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Instructor:** Gabriel Wendell Celestino Rocha      
**Project type:** Classification / time-series learning  
**Difficulty:** Intermediate  

> **Scientific question:** How reliably can stellar variability classes be identified from irregularly sampled light curves, and which representation captures the relevant astrophysical structure best?

---

## 1. Scientific context

Variable-star light curves encode pulsation, rotation, eclipses, outbursts, disk evolution, and other
physical processes. In survey data these signals are complicated by irregular cadence, gaps,
heteroscedastic uncertainties, class imbalance, and survey-specific systematics. This project asks
students to build a scientifically defensible classifier rather than simply maximize accuracy.

## 2. Project objective

Develop a complete, reproducible machine-learning analysis that answers the scientific question above.
The project should be treated as a small research study: the goal is **not** merely to obtain a good
score, but to justify the representation, validation protocol, model family, metrics, and astrophysical
interpretation.

## 3. Dataset requirements

Use a labeled light-curve sample from a survey such as OGLE, TESS, Kepler/K2, ZTF, or a course-provided
subset. Each object should ideally contain time, flux or magnitude, measurement uncertainty, and a
class label. If multiple surveys or fields are present, preserve a source/domain identifier.

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

1. Audit the class distribution, cadence, number of observations, time baseline, missingness, and uncertainty distribution.
2. Create leakage-safe train/validation/test partitions at the physical-object level.
3. Construct at least one engineered-feature representation (e.g. robust moments, amplitudes, Lomb–Scargle descriptors, autocorrelation/structure-function summaries).
4. Train a classical baseline such as logistic regression, random forest, or gradient boosting.
5. Construct one sequence-based representation using the raw or resampled light curve.
6. Train one neural sequence model (1D CNN, TCN, GRU/LSTM, or compact Transformer).
7. Compare performance under identical test partitions.
8. Inspect false positives and false negatives and relate them to light-curve morphology.
9. Discuss whether the model learned stellar variability or survey/cadence shortcuts.

## 6. Minimum modeling requirements

- One interpretable classical baseline
- One deep sequence model

The final comparison must use the **same frozen test set** for all models.

## 7. Evaluation

Recommended metrics/diagnostics:

- Balanced accuracy
- Precision
- Recall
- F1
- Matthews correlation coefficient
- Precision–recall curve / average precision
- Per-class confusion matrix

Do not report a single headline metric without a corresponding error analysis.

## 8. Required figures and tables

Your final notebook/report should contain at least:

- Representative light curves by class
- Class and cadence distributions
- Precision–recall curves
- Confusion matrix
- Example false positives / false negatives
- Optional embedding visualization

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

- Compare feature, raw-sequence, and periodogram representations.
- Introduce class-weighted or focal-loss training.
- Train on one survey/field and test on another.
- Use transfer learning or a pretrained astronomical time-series encoder.
- Calibrate probabilities and define a candidate-selection threshold.

## 11. Questions that must be answered in the conclusion

- What did the best model actually learn?
- Which examples are hardest, and why?
- What is the dominant limitation: data quantity, label quality, representation, domain shift, noise, or model capacity?
- Would you trust the model for the intended astrophysical use case?
- What additional observation or dataset would most improve the scientific conclusion?

---

## Starter workspace

The cells below are deliberately incomplete. Replace the placeholders with your own data and analysis.
