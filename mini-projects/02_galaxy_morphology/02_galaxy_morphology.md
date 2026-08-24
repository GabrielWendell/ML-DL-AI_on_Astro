# Mini-Project — Galaxy Morphology Classification with Convolutional Neural Networks

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Instructor:** Gabriel Wendell Celestino Rocha     
**Project type:** Image classification / CNNs  
**Difficulty:** Intermediate  

> **Scientific question:** How well can galaxy morphology be inferred from images, and how robust are the predictions to image quality, preprocessing, and observational biases?

---

## 1. Scientific context

Galaxy morphology encodes information about dynamical history, star formation, mergers, and environment.
Image-based classification is therefore a natural application of CNNs, but apparent morphology also
depends on signal-to-noise ratio, angular size, redshift, point-spread function, and survey characteristics.
A useful classifier must be tested against these confounders.

## 2. Project objective

Develop a complete, reproducible machine-learning analysis that answers the scientific question above.
The project should be treated as a small research study: the goal is **not** merely to obtain a good
score, but to justify the representation, validation protocol, model family, metrics, and astrophysical
interpretation.

## 3. Dataset requirements

Use a labeled galaxy image sample such as Galaxy Zoo-derived labels, SDSS cutouts, DECaLS/Legacy Survey
images, HST subsets, or a course-provided dataset. Labels can be binary (e.g. spiral/elliptical) or
multiclass if the sample size permits.

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

1. Inspect image dimensions, dynamic range, missing/corrupt samples, label balance, and representative examples.
2. Create object-level train/validation/test splits and avoid duplicate observations across partitions.
3. Define a transparent image preprocessing pipeline (normalization, clipping, resizing/cropping).
4. Train a simple non-deep baseline using compact image descriptors or flattened/downsampled pixels.
5. Train a CNN from scratch.
6. Evaluate the effect of physically reasonable data augmentation.
7. Analyze errors as a function of signal-to-noise, apparent size, brightness, or redshift if available.
8. Use a saliency/attribution visualization cautiously to inspect model sensitivity.
9. Discuss whether the classifier is learning morphology or image-quality/survey artifacts.

## 6. Minimum modeling requirements

- One simple baseline
- One convolutional neural network

The final comparison must use the **same frozen test set** for all models.

## 7. Evaluation

Recommended metrics/diagnostics:

- Balanced accuracy
- Precision
- Recall
- F1
- ROC-AUC or PR-AUC
- Confusion matrix
- Performance versus an observational quality variable

Do not report a single headline metric without a corresponding error analysis.

## 8. Required figures and tables

Your final notebook/report should contain at least:

- Image montage by class
- Training/validation curves
- Confusion matrix
- Misclassified image gallery
- Performance versus brightness/redshift/size
- Attribution or saliency examples

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

- Fine-tune a pretrained vision backbone.
- Compare single-band with multiband images.
- Investigate domain transfer between surveys.
- Use probabilistic labels or Galaxy Zoo vote fractions.
- Perform out-of-distribution testing on unusual morphologies.

## 11. Questions that must be answered in the conclusion

- What did the best model actually learn?
- Which examples are hardest, and why?
- What is the dominant limitation: data quantity, label quality, representation, domain shift, noise, or model capacity?
- Would you trust the model for the intended astrophysical use case?
- What additional observation or dataset would most improve the scientific conclusion?

---

## Starter workspace

The cells below are deliberately incomplete. Replace the placeholders with your own data and analysis.
