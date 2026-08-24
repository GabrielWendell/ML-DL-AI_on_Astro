# Mini-Project — Inferring Circumstellar-Disk Geometry from Stellar Polarimetry

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Instructor:** Gabriel Wendell Celestino Rocha     
**Project type:** Regression/classification / polarimetry / multimodal ML  
**Difficulty:** Intermediate–Advanced  

> **Scientific question:** How much information about circumstellar-disk inclination or viewing geometry can be recovered from linear polarimetric observables, and which coordinate representation is most suitable for ML?

---

## 1. Scientific context

Linear polarization probes asymmetric scattering geometries and can therefore carry information about
circumstellar disks and inclination. This project is designed around a subtle representation question:
the Stokes-space coordinates (q,u) and the derived variables (p,χ) encode related information but present
very different geometry to a learning algorithm. Measurement uncertainties and the periodicity of χ
make the problem scientifically and computationally rich.

## 2. Project objective

Develop a complete, reproducible machine-learning analysis that answers the scientific question above.
The project should be treated as a small research study: the goal is **not** merely to obtain a good
score, but to justify the representation, validation protocol, model family, metrics, and astrophysical
interpretation.

## 3. Dataset requirements

Use observed or simulated stellar polarimetry with normalized Stokes parameters q=Q/I and u=U/I,
preferably across multiple bands or epochs, together with a target such as disk inclination or a
coarse orientation label. A course-provided synthetic dataset is acceptable if realistic noise and
uncertainties are included.

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

1. Inspect q, u, polarization degree p, polarization angle χ, measurement uncertainties, and target distribution.
2. Visualize the sample in the q–u plane and identify geometric structure.
3. Construct at least three feature sets: (q,u), (p,χ), and an angle-safe representation such as (p, sin 2χ, cos 2χ).
4. Choose either inclination regression or pole-on/intermediate/edge-on classification as the primary task.
5. Train a simple linear/logistic baseline.
6. Train a nonlinear classical model such as random forest or gradient boosting.
7. Train an MLP and compare all representations under identical splits.
8. Propagate or otherwise account for measurement uncertainties in q and u.
9. Investigate performance as a function of inclination and polarization strength.
10. Explain why directly using χ may create artificial discontinuities for an ML model.

## 6. Minimum modeling requirements

- Linear/logistic baseline
- Nonlinear tree/boosting model
- MLP

The final comparison must use the **same frozen test set** for all models.

## 7. Evaluation

Recommended metrics/diagnostics:

- For regression: MAE, RMSE, bias, residual versus inclination
- For classification: balanced accuracy, F1, MCC, PR-AUC
- Performance versus polarization S/N
- Optional calibration / uncertainty coverage

Do not report a single headline metric without a corresponding error analysis.

## 8. Required figures and tables

Your final notebook/report should contain at least:

- q–u diagram colored by inclination/class
- Polarization degree and angle distributions
- Representation comparison table/plot
- Predicted versus true inclination or confusion matrix
- Residual/performance versus polarization S/N
- Representative difficult cases

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

- Use multi-band polarimetry.
- Use multi-epoch q–u trajectories with sequence models.
- Combine photometry and polarimetry in a multimodal model.
- Add survey/domain offsets and test robustness.
- Model heteroscedastic uncertainty explicitly.
- Apply the framework to Be-star disk geometry or candidate characterization.

## 11. Questions that must be answered in the conclusion

- What did the best model actually learn?
- Which examples are hardest, and why?
- What is the dominant limitation: data quantity, label quality, representation, domain shift, noise, or model capacity?
- Would you trust the model for the intended astrophysical use case?
- What additional observation or dataset would most improve the scientific conclusion?

---

## Starter workspace

The cells below are deliberately incomplete. Replace the placeholders with your own data and analysis.
