# Exercise Set 2 — Notebook 1
## Catalog machine learning and a first neural network

**Course:** Machine Learning, Deep Learning, and AI for Astrophysics  
**Week 2 — Monday (Slots 7–8)**

This notebook is deliberately self-contained: it generates a synthetic but astrophysically motivated catalog so that the complete workflow can be run without downloading external data. In the actual mini-course, the same workflow can be applied to a real survey table.

### Learning goals
By the end of this notebook you should be able to:

- audit a tabular astrophysical dataset before modeling;
- create leakage-aware train/validation/test partitions;
- establish classical ML baselines;
- evaluate rare-class performance with metrics beyond accuracy;
- tune an operating threshold for a scientific objective;
- diagnose survey/source shortcut learning;
- train a small PyTorch MLP and compare it fairly with simpler baselines.

> **Rule for the exercises:** do not report a score without also stating what split, metric, and scientific use case produced it.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve
)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

```

## 1. Build an astrophysical toy catalog

We simulate a population in which a rare positive class is associated with stronger variability and infrared excess. Two surveys have different cadence/noise properties. Survey identity is **not** the scientific target, but it is intentionally easy to infer from some observational descriptors.

This creates a realistic teaching problem: a model can obtain an apparently good score while partly learning data provenance.


```python
n = 4500
survey = rng.choice(["A", "B"], size=n, p=[0.68, 0.32])

# Latent astrophysical quantities
teff = rng.normal(12000, 2800, n).clip(6500, 25000)
variability = rng.lognormal(mean=-0.4, sigma=0.65, size=n)
ir_excess = rng.normal(0.0, 0.35, n)

# Scientific probability: depends on physical quantities, not directly on survey.
logit = -3.0 + 1.65 * variability + 1.45 * ir_excess
prob = 1.0 / (1.0 + np.exp(-logit))
y = rng.binomial(1, prob)

# Survey-dependent observing characteristics.
n_obs = np.where(survey == "A", rng.poisson(220, n), rng.poisson(85, n)).clip(20)
phot_err = np.where(survey == "A", rng.lognormal(-3.6, 0.25, n), rng.lognormal(-3.0, 0.35, n))
cadence_days = np.where(survey == "A", rng.normal(2.2, 0.35, n), rng.normal(5.5, 0.9, n)).clip(0.3)

# Measured astrophysical features.
g_r = 0.9 - (teff - 6500) / 25000 + rng.normal(0, 0.05, n)
r_i = 0.45 * g_r + 0.32 * ir_excess + rng.normal(0, 0.04, n)
amplitude = 0.04 + 0.22 * variability + 0.12 * y + rng.normal(0, 0.04, n)
periodic_power = np.clip(0.15 + 0.35 * variability + 0.14 * y + rng.normal(0, 0.12, n), 0, None)
skewness = rng.normal(0.15 * y, 0.8, n)

catalog = pd.DataFrame({
    "survey": survey,
    "teff_proxy": teff,
    "g_r": g_r,
    "r_i": r_i,
    "amplitude": amplitude,
    "periodic_power": periodic_power,
    "skewness": skewness,
    "n_obs": n_obs,
    "phot_err": phot_err,
    "cadence_days": cadence_days,
    "target": y,
})

catalog.head()

```


```python
print(catalog.shape)
print("Positive fraction:", catalog["target"].mean())
print("\nClass fraction by survey:")
print(catalog.groupby("survey")["target"].agg(["count", "mean"]))

```

### Exercise 1 — Audit before modeling

1. Compute descriptive statistics by class and by survey.
2. Identify which columns are primarily astrophysical and which are observational/provenance descriptors.
3. Make at least two plots that help you answer: *could a classifier infer survey identity even without an explicit survey column?*
4. Explain why a source-prediction model is a useful diagnostic but not, by itself, proof that the science classifier is invalid.


```python
# Suggested starting point for Exercise 1
print(catalog.groupby("target")[["amplitude", "periodic_power", "r_i", "phot_err", "cadence_days"]].median())

plt.figure(figsize=(6, 4))
for name, group in catalog.groupby("survey"):
    plt.hist(group["cadence_days"], bins=30, alpha=0.6, label=f"Survey {name}")
plt.xlabel("Cadence [days]")
plt.ylabel("Count")
plt.legend()
plt.show()

```

## 2. Define a leakage-aware split

For a real project, all observations of the same physical object must remain in one partition. Here each row is already one object. We stratify on the combination of class and survey so that both sources are represented in every partition.


```python
strata = catalog["survey"].astype(str) + "_" + catalog["target"].astype(str)
train_df, test_df = train_test_split(
    catalog, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
train_strata = train_df["survey"].astype(str) + "_" + train_df["target"].astype(str)
train_df, val_df = train_test_split(
    train_df, test_size=0.20, random_state=RANDOM_STATE, stratify=train_strata
)

for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    print(name, len(df), "positive fraction=", round(df["target"].mean(), 3),
          "survey A fraction=", round((df["survey"] == "A").mean(), 3))

```

### Exercise 2 — Split critique

- Why should the **test** set remain untouched until model and threshold choices have been frozen?
- What additional grouping would be mandatory if several rows corresponded to the same star?
- Design a second, harder protocol that trains on Survey A and tests on Survey B. What scientific question would that protocol answer?

## 3. Classical baselines

We deliberately exclude the explicit `survey` column. We will first compare logistic regression and random forest using the same split.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

feature_cols = [
    "teff_proxy", "g_r", "r_i", "amplitude", "periodic_power", "skewness",
    "n_obs", "phot_err", "cadence_days"
]

X_train, y_train = train_df[feature_cols], train_df["target"]
X_val, y_val = val_df[feature_cols], val_df["target"]
X_test, y_test = test_df[feature_cols], test_df["target"]

logreg = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))
])
rf = RandomForestClassifier(
    n_estimators=140, class_weight="balanced_subsample", min_samples_leaf=3,
    random_state=RANDOM_STATE, n_jobs=-1
)

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

```


```python
def summarize_binary(name, y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return pd.Series({
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "average_precision": average_precision_score(y_true, prob),
    })

val_results = pd.DataFrame([
    summarize_binary("logistic", y_val, logreg.predict_proba(X_val)[:, 1]),
    summarize_binary("random_forest", y_val, rf.predict_proba(X_val)[:, 1]),
])
val_results

```

### Exercise 3 — Baseline interpretation

1. Which metric should be primary if the goal is **high-recall candidate mining**?
2. Which metric or curve becomes especially useful if follow-up resources are expensive and false positives matter?
3. Compare the models. Does the more flexible model clearly dominate?
4. Remove `n_obs`, `phot_err`, and `cadence_days`, retrain both models, and compare the change in validation performance. What does this tell you about observational shortcuts?

## 4. Threshold selection is a scientific decision

A probability threshold of 0.5 is conventional, not sacred. We choose a threshold **only on the validation set**.


```python
val_prob = rf.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_prob)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Validation precision–recall curve")
plt.show()

# Example policy: choose the highest threshold that still yields recall >= 0.80.
valid = np.where(recall[:-1] >= 0.80)[0]
chosen_threshold = thresholds[valid[-1]] if len(valid) else 0.5
print("Chosen threshold:", chosen_threshold)
print(summarize_binary("RF tuned", y_val, val_prob, chosen_threshold))

```

### Exercise 4 — Build two operating policies

Using the validation predictions, define:

- **Policy A:** maximize recall subject to precision ≥ 0.40;
- **Policy B:** maximize precision subject to recall ≥ 0.70.

For each policy, record the threshold and explain the corresponding scientific use case. Then freeze one policy for final testing.

## 5. Source-prediction diagnostic

Now we explicitly ask whether the same feature table reveals survey identity.


```python
from sklearn.metrics import roc_auc_score

source_target_train = (train_df["survey"] == "B").astype(int)
source_target_val = (val_df["survey"] == "B").astype(int)

source_model = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
])
source_model.fit(X_train, source_target_train)
source_prob = source_model.predict_proba(X_val)[:, 1]
print("Survey prediction ROC-AUC:", roc_auc_score(source_target_val, source_prob))

```

### Exercise 5 — Shortcut-learning diagnosis

1. Which features make survey identity easy to predict?
2. Train the science classifier using only the more obviously astrophysical columns.
3. Compare pooled performance and cross-survey performance.
4. Write a short conclusion using careful language: what evidence suggests shortcut risk, and what evidence would be required to claim domain-robust learning?

## 6. A small PyTorch MLP

The objective is **not** to beat the classical models at all costs. The objective is to learn the mechanics of a deep-learning workflow and compare it under exactly the same evaluation protocol.


```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(RANDOM_STATE)

scaler = StandardScaler().fit(X_train)
Xtr = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
Xva = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
Xte = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
ytr = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
yva = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)

model = nn.Sequential(
    nn.Linear(Xtr.shape[1], 32),
    nn.ReLU(),
    nn.Dropout(0.15),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

# Positive-class weighting estimated from the training data.
pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

history = []
for epoch in range(10):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(xb)

    model.eval()
    with torch.no_grad():
        val_logits = model(Xva)
        val_loss = criterion(val_logits, yva).item()
    history.append((running / len(Xtr), val_loss))

history = np.asarray(history)
plt.figure(figsize=(6, 4))
plt.plot(history[:, 0], label="train")
plt.plot(history[:, 1], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

```


```python
model.eval()
with torch.no_grad():
    mlp_val_prob = torch.sigmoid(model(Xva)).numpy().ravel()

pd.DataFrame([
    summarize_binary("random_forest", y_val, val_prob),
    summarize_binary("MLP", y_val, mlp_val_prob),
])

```

### Exercise 6 — Deep-learning diagnostics

1. Change the hidden-layer widths and observe train/validation behavior.
2. Remove dropout. Does the validation curve change?
3. Remove `pos_weight`. Which metrics are affected most?
4. Compare the MLP with the random forest. If the MLP is not better, is that a failure of the exercise? Explain.
5. Freeze the best model and threshold using only training/validation data, then evaluate **once** on the test set.

## 7. Final scientific reflection

Write a short research-note paragraph answering all of the following:

- Which model would you deploy for candidate ranking and why?
- What is the dominant failure mode?
- Is source identity encoded in the features?
- Which result is likely to change most under a new survey?
- What additional evidence would you demand before making an astrophysical claim?

### Optional extension
Repeat the entire experiment with a **cross-survey split**: train on Survey A and test on Survey B. Compare this with the pooled random split and discuss the difference between interpolation and domain transfer.
