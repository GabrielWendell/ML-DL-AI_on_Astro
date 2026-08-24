# Mini-Project — Exoplanet Transit Candidate Classification

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Instructor:** Gabriel Wendell Celestino Rocha    
**Project type:** Rare-event classification / time series  
**Difficulty:** Intermediate–Advanced  

> **Scientific question:** Can a classifier identify credible exoplanet transit candidates while controlling the false-positive rate in a strongly imbalanced setting?

---

## 1. Scientific context

Transit searches are classic rare-event problems. Periodic decreases in brightness may be caused by
planets, eclipsing binaries, stellar variability, systematics, or processing artifacts. The scientific
utility of a model therefore depends strongly on recall, precision, threshold choice, and error analysis,
not on raw accuracy.

## 2. Project objective

Develop a complete, reproducible machine-learning analysis that answers the scientific question above.
The project should be treated as a small research study: the goal is **not** merely to obtain a good
score, but to justify the representation, validation protocol, model family, metrics, and astrophysical
interpretation.

## 3. Dataset requirements

Use labeled Kepler, K2, or TESS threshold-crossing events/candidates, or a course-prepared collection of
phase-folded light curves. Where possible include auxiliary quantities such as period, depth, duration,
odd/even depth differences, centroid diagnostics, and disposition labels.

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

1. Quantify class imbalance and inspect representative planet-like and false-positive events.
2. Create leakage-safe splits at the target-star level.
3. Build a feature-based baseline using transit-shape and period-related descriptors.
4. Train a classifier appropriate for imbalance (e.g. weighted random forest/boosting).
5. Train a 1D CNN on phase-folded or local/global transit views.
6. Compare threshold choices for high-recall candidate mining versus precision-oriented follow-up.
7. Report PR-AUC/average precision and do not rely on accuracy alone.
8. Inspect false positives and false negatives for plausible astrophysical causes.
9. Assess calibration or confidence ranking for follow-up prioritization.

## 6. Minimum modeling requirements

- Feature-based imbalance-aware classifier
- 1D CNN on transit light curves

The final comparison must use the **same frozen test set** for all models.

## 7. Evaluation

Recommended metrics/diagnostics:

- Precision
- Recall
- F1
- Average precision / PR-AUC
- MCC
- Recall at fixed precision or precision at fixed recall
- Calibration/Brier score (optional)

Do not report a single headline metric without a corresponding error analysis.

## 8. Required figures and tables

Your final notebook/report should contain at least:

- Representative phase-folded events
- Class-balance plot
- Precision–recall curve
- Threshold trade-off plot
- Confusion matrix
- False-positive / false-negative gallery

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

- Use local and global light-curve views in a two-branch network.
- Inject synthetic transits and perform a small recovery experiment.
- Compare candidate ranking before and after calibration.
- Test robustness to missing cadences or increased noise.
- Combine light curves with stellar metadata.

## 11. Questions that must be answered in the conclusion

- What did the best model actually learn?
- Which examples are hardest, and why?
- What is the dominant limitation: data quantity, label quality, representation, domain shift, noise, or model capacity?
- Would you trust the model for the intended astrophysical use case?
- What additional observation or dataset would most improve the scientific conclusion?

---

## Starter workspace

The cells below are deliberately incomplete. Replace the placeholders with your own data and analysis.
