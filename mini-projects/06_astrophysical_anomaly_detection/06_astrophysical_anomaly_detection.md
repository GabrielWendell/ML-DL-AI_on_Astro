# Mini-Project — Anomaly Detection in Astronomical Data

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Instructor:** Gabriel Wendell Celestino Rocha     
**Project type:** Unsupervised / semi-supervised learning  
**Difficulty:** Advanced  

> **Scientific question:** How can we identify scientifically interesting rare objects when the anomalous classes are not known in advance?

---

## 1. Scientific context

Astronomy routinely contains rare, poorly labeled, or genuinely novel phenomena. This makes anomaly
detection attractive but also dangerous: an algorithm often flags bad data, low signal-to-noise,
edge effects, or survey failures rather than new astrophysics. A successful project must distinguish
data-quality anomalies from scientifically meaningful outliers.

## 2. Project objective

Develop a complete, reproducible machine-learning analysis that answers the scientific question above.
The project should be treated as a small research study: the goal is **not** merely to obtain a good
score, but to justify the representation, validation protocol, model family, metrics, and astrophysical
interpretation.

## 3. Dataset requirements

Use a homogeneous collection of catalog vectors, spectra, images, or light-curve features containing a
large ordinary population and a smaller number of known or suspected unusual objects. Labels are not
required for training, but a small reference set is useful for evaluation.

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

1. Define what 'anomaly' means scientifically and distinguish it from bad data.
2. Audit missing values, measurement quality, coverage, and obvious instrumental failures.
3. Construct a baseline low-dimensional representation using PCA or another transparent method.
4. Apply at least one classical anomaly detector such as Isolation Forest or Local Outlier Factor.
5. Train an autoencoder or another representation-learning model and derive an anomaly score.
6. Compare the overlap and disagreement between anomaly rankings.
7. Inspect the highest-ranked objects manually or with diagnostic plots.
8. Determine whether anomaly score correlates with data quality variables.
9. Create a taxonomy of discovered anomalies: astrophysical, instrumental, processing-related, or uncertain.
10. Discuss how one should validate an anomaly detector without complete ground-truth labels.

## 6. Minimum modeling requirements

- One classical anomaly detector
- One learned representation / autoencoder

The final comparison must use the **same frozen test set** for all models.

## 7. Evaluation

Recommended metrics/diagnostics:

- Ranking stability
- Overlap among top-k lists
- Precision@k if reference labels exist
- Recovery of injected/known rare objects
- Correlation with quality metrics

Do not report a single headline metric without a corresponding error analysis.

## 8. Required figures and tables

Your final notebook/report should contain at least:

- Low-dimensional embedding
- Anomaly-score distribution
- Top-ranked anomaly gallery
- Comparison of anomaly rankings
- Anomaly score versus data quality
- Taxonomy of inspected anomalies

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

- Inject controlled synthetic anomalies and measure recovery.
- Use self-supervised embeddings before anomaly detection.
- Compare global and local anomaly methods.
- Cluster the highest-ranked anomalies into phenomenological groups.
- Perform active-learning style human-in-the-loop refinement.

## 11. Questions that must be answered in the conclusion

- What did the best model actually learn?
- Which examples are hardest, and why?
- What is the dominant limitation: data quantity, label quality, representation, domain shift, noise, or model capacity?
- Would you trust the model for the intended astrophysical use case?
- What additional observation or dataset would most improve the scientific conclusion?

---

## Starter workspace

The cells below are deliberately incomplete. Replace the placeholders with your own data and analysis.
