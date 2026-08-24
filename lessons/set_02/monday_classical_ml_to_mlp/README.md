# Week 2 Monday — Classical ML → Neural Network

This folder contains the Monday hands-on class for the mini-course
**Machine Learning and Artificial Intelligence for Modern Astrophysics**.

## Files

- `monday_classical_ml_to_mlp_instructor.ipynb` — fully runnable projected/live-coding notebook.
- `monday_classical_ml_to_mlp_student.ipynb` — student copy.
- `monday_classical_ml_to_mlp.md` — GitHub-readable Markdown version.
- `requirements.txt` — minimal environment.

## Pedagogical structure

The notebook is one continuous scientific story rather than a sequence of unrelated exercises.

**Slot 1:** catalog audit → frozen split → majority baseline → logistic regression → threshold policy → random forest.

**Slot 2:** PyTorch tensors/loaders → MLP → training/validation curves → fair comparison → student investigation.

The central question is:

> Does the neural network improve the scientifically important result enough to justify its added complexity?

The notebook uses a reproducible synthetic stellar catalog so it can run without external data.
