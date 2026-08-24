# Week 2 — Monday Hands-on
## From Classical Machine Learning to a Neural Network

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Format:** live Jupyter laboratory  
**Total duration:** approximately 100–120 minutes (two 50–60 minute slots)

### Central question

> **Does a neural network actually improve the scientific result, or is a simpler model already sufficient?**

Today we will work on **one astrophysical classification problem from beginning to end**. The same
frozen train/validation/test split will be used for all models so that the comparison is scientifically
fair.

---

## Learning objectives

By the end of the class, students should be able to:

1. turn an astrophysical question into a supervised classification problem;
2. audit an imbalanced catalog before modeling;
3. construct leakage-safe train/validation/test partitions;
4. establish a simple baseline before using a more complex model;
5. train and evaluate logistic regression and random forest models;
6. understand why the probability threshold is a scientific decision;
7. implement a small multilayer perceptron (MLP) in PyTorch;
8. diagnose overfitting using training and validation curves;
9. compare classical ML and a neural network under the **same evaluation protocol**;
10. decide whether the added complexity of deep learning is justified.

---

## Class rhythm

| Approx. time | Activity |
|---:|---|
| 0–10 min | Scientific problem + data audit |
| 10–25 min | Split, preprocessing, majority baseline |
| 25–45 min | Logistic regression + metrics |
| 45–55 min | Random forest + first comparison |
| **break / transition** | |
| 55–70 min | PyTorch tensors, loaders, MLP |
| 70–90 min | Training + validation |
| 90–105 min | Thresholds and fair comparison |
| 105–115 min | Student challenge |
| 115–120 min | Scientific synthesis |

The recurring teaching pattern is:

\[
\boxed{\text{Predict} \rightarrow \text{Run} \rightarrow \text{Explain}}
\]

Before many cells, pause and ask students what they expect to happen.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score,
    confusion_matrix, precision_recall_curve
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("PyTorch:", torch.__version__)
```

# Slot 1 — Classical ML pipeline

## 1. Scientific problem

Imagine a time-domain survey has produced a catalog of stellar sources. Expensive spectroscopy can
confirm a rare target population, but only for a small number of objects.

We want to build a **candidate-ranking/classification model** from inexpensive photometric and
variability descriptors.

For this teaching example the positive class is a synthetic **rare active/emission-line stellar
candidate**. The dataset is simulated locally so the notebook is fully reproducible and requires no
external download.

Each object has features resembling quantities that may occur in real survey pipelines:

- `color_bp_rp`: broad-band color;
- `abs_mag_proxy`: luminosity-like proxy;
- `amplitude`: robust photometric variability amplitude;
- `skewness`: light-curve asymmetry;
- `periodic_power`: strength of a dominant periodic component;
- `long_term_slope`: secular brightening/fading descriptor;
- `n_obs`: number of observations;
- `phot_noise`: typical photometric uncertainty;
- `survey`: observing domain/source.

The target is

\[
y =
\begin{cases}
1,&\text{rare candidate},\\
0,&\text{ordinary object}.
\end{cases}
\]

### Scientific warning

The class is intentionally imbalanced. A model that predicts **every object as ordinary** can therefore
obtain a deceptively high accuracy.

### Predict → Run → Explain

Before generating the data:

> If only about 8–10% of the catalog belongs to the positive class, what accuracy would the
> trivial majority classifier obtain?

## 2. Generate the teaching catalog

The generator below creates a deliberately imperfect problem:

- the positive class is rare;
- classes overlap;
- measurement quality varies;
- one survey has a slightly different selection function;
- no single feature perfectly determines the label.

This is much closer to a scientific ML problem than a perfectly separable classroom dataset.

```python
def make_stellar_catalog(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    survey_code = rng.integers(0, 2, size=n)

    color_bp_rp = rng.normal(0.15 + 0.08 * survey_code, 0.28, size=n)
    abs_mag_proxy = rng.normal(-0.6 + 0.15 * survey_code, 0.9, size=n)
    amplitude = rng.gamma(shape=1.8, scale=0.08, size=n)
    skewness = rng.normal(0.0, 0.75, size=n)
    periodic_power = rng.beta(1.5, 4.0, size=n)
    long_term_slope = rng.normal(0.0, 0.045, size=n)
    n_obs = rng.poisson(110 - 22 * survey_code, size=n) + 25
    phot_noise = np.exp(rng.normal(-3.5 + 0.28 * survey_code, 0.35, size=n))

    # Nonlinear latent score. The survey term is intentionally small but nonzero.
    score = (
        -4.4
        - 1.1 * color_bp_rp
        - 0.25 * abs_mag_proxy
        + 5.2 * amplitude
        + 0.65 * np.maximum(skewness, 0)
        + 2.1 * periodic_power
        + 8.0 * np.abs(long_term_slope)
        - 6.0 * phot_noise
        + 0.30 * survey_code
        + 1.6 * amplitude * periodic_power
    )
    prob = 1.0 / (1.0 + np.exp(-score))
    y = rng.binomial(1, prob)

    # Add measurement scatter to observed features after labels are generated.
    amplitude_obs = np.clip(amplitude + rng.normal(0, phot_noise * 0.5), 0, None)
    slope_obs = long_term_slope + rng.normal(0, phot_noise * 0.18)

    df = pd.DataFrame({
        "object_id": [f"STAR_{i:05d}" for i in range(n)],
        "color_bp_rp": color_bp_rp,
        "abs_mag_proxy": abs_mag_proxy,
        "amplitude": amplitude_obs,
        "skewness": skewness,
        "periodic_power": periodic_power,
        "long_term_slope": slope_obs,
        "n_obs": n_obs,
        "phot_noise": phot_noise,
        "survey_code": survey_code,
        "target": y,
    })
    return df

df = make_stellar_catalog()
df.head()
```

## 3. Audit before modeling

Never begin with `model.fit(...)`.

First ask:

- How many objects are there?
- How imbalanced is the target?
- Are there missing values?
- Are numerical ranges very different?
- Is the positive fraction the same in both surveys?
- Are some variables suspiciously related to survey provenance?

### Predict → Run → Explain

> Which metric will be most misleading under strong imbalance: accuracy, recall, or average precision?

```python
print("Objects:", len(df))
print("\nTarget counts:")
print(df["target"].value_counts())
print("\nPositive fraction:", df["target"].mean().round(4))
print("\nMissing values:")
print(df.isna().sum())

print("\nPositive fraction by survey:")
print(df.groupby("survey_code")["target"].agg(["count", "mean"]))
```

```python
fig, ax = plt.subplots(figsize=(6.5, 4))
counts = df["target"].value_counts().sort_index()
ax.bar(["ordinary", "rare candidate"], counts.values)
ax.set_ylabel("Number of objects")
ax.set_title("Class imbalance")
plt.show()

fig, ax = plt.subplots(figsize=(6.5, 4))
for label, group in df.groupby("target"):
    ax.hist(group["amplitude"], bins=40, alpha=0.55, density=True,
            label="candidate" if label == 1 else "ordinary")
ax.set_xlabel("Variability amplitude")
ax.set_ylabel("Density")
ax.legend()
ax.set_title("Overlapping class-conditional distributions")
plt.show()
```

## 4. Freeze train / validation / test partitions

We use three partitions:

\[
\mathcal D
=
\mathcal D_{\rm train}
\cup
\mathcal D_{\rm val}
\cup
\mathcal D_{\rm test}.
\]

- **Training:** fit model parameters.
- **Validation:** choose hyperparameters and thresholds.
- **Test:** final estimate only.

We stratify by the target so that the rare-class fraction is approximately preserved.

> **Scientific rule:** once we inspect test-set performance, we should not use it to redesign the model.

The exact row IDs are saved so every later model uses the same objects.

```python
feature_cols = [
    "color_bp_rp",
    "abs_mag_proxy",
    "amplitude",
    "skewness",
    "periodic_power",
    "long_term_slope",
    "n_obs",
    "phot_noise",
    "survey_code",
]

train_df, temp_df = train_test_split(
    df,
    test_size=0.40,
    stratify=df["target"],
    random_state=SEED,
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["target"],
    random_state=SEED,
)

split_table = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "N": [len(train_df), len(val_df), len(test_df)],
    "positive_fraction": [
        train_df["target"].mean(),
        val_df["target"].mean(),
        test_df["target"].mean(),
    ],
})
split_table
```

```python
split_ids = pd.concat([
    train_df[["object_id"]].assign(split="train"),
    val_df[["object_id"]].assign(split="validation"),
    test_df[["object_id"]].assign(split="test"),
], ignore_index=True)

split_ids.to_csv("monday_frozen_splits.csv", index=False)
print("Saved:", "monday_frozen_splits.csv")
```

## 5. The first model should be embarrassingly simple

Before logistic regression, random forests, or neural networks, establish the **majority-class baseline**.

If a sophisticated model cannot beat a trivial reference under appropriate metrics, the sophistication
is not scientifically useful.

### Predict

> Will the majority classifier have high accuracy?  
> What will its recall for the rare class be?

```python
def metrics_dict(y_true, y_pred, y_score=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None:
        out["average_precision"] = average_precision_score(y_true, y_score)
    return out

y_test = test_df["target"].to_numpy()
majority_pred = np.zeros_like(y_test)

pd.Series(metrics_dict(y_test, majority_pred), name="majority baseline")
```

## 6. Logistic regression: the first serious baseline

For binary classification,

\[
P(y=1\mid \mathbf x)
=
\sigma(\mathbf w^\top\mathbf x+b),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

Logistic regression is valuable because it is:

- fast;
- interpretable;
- easy to regularize;
- a strong sanity check.

We standardize the continuous variables **using training data only**.

We also use `class_weight="balanced"` because the positive class is rare.

### Predict → Run → Explain

> Do you expect class weighting to increase or decrease positive-class recall?  
> Must it improve overall accuracy?

```python
X_train = train_df[feature_cols].to_numpy()
X_val = val_df[feature_cols].to_numpy()
X_test = test_df[feature_cols].to_numpy()

y_train = train_df["target"].to_numpy()
y_val = val_df["target"].to_numpy()
y_test = test_df["target"].to_numpy()

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

logreg = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=SEED,
)
logreg.fit(X_train_s, y_train)

val_prob_lr = logreg.predict_proba(X_val_s)[:, 1]
test_prob_lr = logreg.predict_proba(X_test_s)[:, 1]

test_pred_lr = (test_prob_lr >= 0.5).astype(int)
pd.Series(metrics_dict(y_test, test_pred_lr, test_prob_lr), name="logistic regression")
```

```python
coef = pd.Series(logreg.coef_[0], index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.barh(coef.index, coef.values)
ax.set_xlabel("Standardized logistic coefficient")
ax.set_title("Logistic-regression coefficients")
plt.show()
```

```python
cm = confusion_matrix(y_test, test_pred_lr)
fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(cm)
for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, str(value), ha="center", va="center")
ax.set_xticks([0, 1], ["ordinary", "candidate"])
ax.set_yticks([0, 1], ["ordinary", "candidate"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Logistic regression — test confusion matrix")
plt.show()
```

## 7. A probability is not yet a decision

Most classifiers return a score or estimated probability

\[
\hat p_i=P(y_i=1\mid\mathbf x_i).
\]

A label requires a threshold

\[
\hat y_i(\tau)=
\begin{cases}
1,&\hat p_i\ge\tau,\\
0,&\hat p_i<\tau.
\end{cases}
\]

The conventional \(\tau=0.5\) is **not a law of nature**.

In astronomy, different use cases imply different operating points:

- candidate mining → prioritize recall;
- expensive follow-up → prioritize precision;
- catalog construction → seek a compromise.

### Student challenge

Use the **validation set only** to find the smallest threshold that achieves at least 85% recall.
Then report the corresponding precision.

```python
precision, recall, thresholds = precision_recall_curve(y_val, val_prob_lr)

# precision and recall have one extra element relative to thresholds.
valid = np.where(recall[:-1] >= 0.85)[0]
if len(valid):
    # Among thresholds satisfying recall target, choose the one with highest precision.
    best_idx = valid[np.argmax(precision[:-1][valid])]
    tau_lr = thresholds[best_idx]
else:
    tau_lr = 0.5

val_pred_tuned = (val_prob_lr >= tau_lr).astype(int)

print(f"Chosen validation threshold: {tau_lr:.3f}")
print(f"Validation recall:    {recall_score(y_val, val_pred_tuned):.3f}")
print(f"Validation precision: {precision_score(y_val, val_pred_tuned):.3f}")
```

```python
test_pred_lr_tuned = (test_prob_lr >= tau_lr).astype(int)
pd.Series(
    metrics_dict(y_test, test_pred_lr_tuned, test_prob_lr),
    name=f"logistic regression @ tau={tau_lr:.3f}",
)
```

## 8. Random forest: a nonlinear classical model

A random forest combines many decision trees trained on perturbed versions of the data.

Why include it?

- nonlinear decision boundaries;
- feature interactions;
- little preprocessing;
- strong tabular baseline;
- easy feature-importance diagnostics.

This is the model the neural network must beat or meaningfully complement.

```python
rf = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced_subsample",
    min_samples_leaf=3,
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

val_prob_rf = rf.predict_proba(X_val)[:, 1]
test_prob_rf = rf.predict_proba(X_test)[:, 1]
test_pred_rf = (test_prob_rf >= 0.5).astype(int)

pd.Series(metrics_dict(y_test, test_pred_rf, test_prob_rf), name="random forest")
```

```python
importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.barh(importance.index, importance.values)
ax.set_xlabel("Random-forest feature importance")
ax.set_title("Which variables does the forest use?")
plt.show()
```

```python
p_lr, r_lr, _ = precision_recall_curve(y_test, test_prob_lr)
p_rf, r_rf, _ = precision_recall_curve(y_test, test_prob_rf)

fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.plot(r_lr, p_lr, label=f"Logistic (AP={average_precision_score(y_test, test_prob_lr):.3f})")
ax.plot(r_rf, p_rf, label=f"Random forest (AP={average_precision_score(y_test, test_prob_rf):.3f})")
ax.axhline(y_test.mean(), linestyle="--", label="class prevalence")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–recall comparison")
ax.legend()
plt.show()
```

# End of Slot 1 — What have we established?

We now have:

\[
\text{data audit}
\rightarrow
\text{frozen split}
\rightarrow
\text{trivial baseline}
\rightarrow
\text{logistic regression}
\rightarrow
\text{threshold analysis}
\rightarrow
\text{random forest}.
\]

Before moving to deep learning, ask the class:

> **What scientific problem remains unsolved by simply adding a neural network?**

A neural network will not automatically fix:

- weak labels;
- selection effects;
- domain shift;
- poor validation;
- inappropriate metrics.

It only changes the function class we optimize.

# Slot 2 — Replace the classical model with a neural network

## 9. Why try an MLP?

The same tabular feature vector can be passed to a multilayer perceptron:

\[
\mathbf x
\rightarrow
\mathbf h_1
\rightarrow
\mathbf h_2
\rightarrow
\hat p.
\]

For hidden layer \(\ell\),

\[
\mathbf h^{(\ell+1)}
=
\sigma\left(
W^{(\ell)}\mathbf h^{(\ell)}
+\mathbf b^{(\ell)}
\right).
\]

The important question is not

> “Can we train a neural network?”

but

> **“Does learned nonlinear representation improve generalization enough to justify the added
> complexity?”**

We will keep:

- the same features;
- the same train/validation/test objects;
- the same target;
- the same metric suite.

## 10. The minimal PyTorch workflow

A PyTorch training pipeline has a small number of moving parts:

\[
\boxed{
\text{TensorDataset}
\rightarrow
\text{DataLoader}
\rightarrow
\text{Model}
\rightarrow
\text{Loss}
\rightarrow
\text{Optimizer}
\rightarrow
\text{Training loop}
}
\]

For binary classification we train on **logits** using binary cross-entropy with logits.

The sigmoid is applied only when we want probabilities.

```python
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

y_train_t = torch.tensor(y_train[:, None], dtype=torch.float32)
y_val_t = torch.tensor(y_val[:, None], dtype=torch.float32)
y_test_t = torch.tensor(y_test[:, None], dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=128,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=256,
    shuffle=False,
)

print("Training tensor:", X_train_t.shape)
print("First batch:", next(iter(train_loader))[0].shape)
```

```python
class StellarMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)

model = StellarMLP(len(feature_cols))
print(model)
```

```python
positive = y_train.sum()
negative = len(y_train) - positive
pos_weight = torch.tensor([negative / positive], dtype=torch.float32)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

print("positive-class weight:", float(pos_weight))
```

## 11. Train while watching validation performance

Training loss alone is not enough.

We record

\[
\mathcal L_{\rm train}(t),
\qquad
\mathcal L_{\rm val}(t)
\]

at every epoch.

A common overfitting pattern is:

- training loss continues decreasing;
- validation loss reaches a minimum;
- validation loss then starts increasing.

We therefore save the parameters corresponding to the best validation loss.

### Predict → Run → Explain

> If the MLP has substantially more flexibility than logistic regression, should we expect lower
> training loss?  
> Does that imply lower test error?

```python
def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n

history = {"train_loss": [], "val_loss": []}
best_state = None
best_val = np.inf
patience = 25
epochs_without_improvement = 0
max_epochs = 250

for epoch in range(max_epochs):
    model.train()
    total = 0.0
    n = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        total += loss.item() * len(xb)
        n += len(xb)

    train_loss = total / n
    val_loss = evaluate_loss(model, val_loader, loss_fn)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    if val_loss < best_val - 1e-5:
        best_val = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        break

model.load_state_dict(best_state)

print(f"Stopped after {len(history['train_loss'])} epochs")
print(f"Best validation loss: {best_val:.4f}")
```

```python
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(history["train_loss"], label="training")
ax.plot(history["val_loss"], label="validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Weighted BCE loss")
ax.set_title("MLP learning curves")
ax.legend()
plt.show()
```

```python
def predict_proba_torch(model, X_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze(1)
        return torch.sigmoid(logits).cpu().numpy()

val_prob_mlp = predict_proba_torch(model, X_val_t)
test_prob_mlp = predict_proba_torch(model, X_test_t)
test_pred_mlp = (test_prob_mlp >= 0.5).astype(int)

pd.Series(metrics_dict(y_test, test_pred_mlp, test_prob_mlp), name="MLP")
```

## 12. Fair comparison

All models now face exactly the same test objects.

We compare:

- majority baseline;
- balanced logistic regression;
- random forest;
- MLP.

The final question is not simply:

> Which row contains the largest number?

Instead ask:

1. Is the difference large enough to matter?
2. Which errors changed?
3. Is the more complex model more stable?
4. Does it improve the scientifically important metric?
5. What additional computational/interpretive cost did we pay?

```python
rows = []

rows.append({"model": "Majority baseline", **metrics_dict(y_test, majority_pred)})
rows.append({"model": "Logistic regression", **metrics_dict(y_test, test_pred_lr, test_prob_lr)})
rows.append({"model": "Random forest", **metrics_dict(y_test, test_pred_rf, test_prob_rf)})
rows.append({"model": "MLP", **metrics_dict(y_test, test_pred_mlp, test_prob_mlp)})

comparison = pd.DataFrame(rows).set_index("model")
comparison.round(3)
```

```python
curves = [
    ("Logistic regression", test_prob_lr),
    ("Random forest", test_prob_rf),
    ("MLP", test_prob_mlp),
]

fig, ax = plt.subplots(figsize=(6.8, 5))
for name, prob in curves:
    p, r, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax.plot(r, p, label=f"{name} (AP={ap:.3f})")

ax.axhline(y_test.mean(), linestyle="--", label="class prevalence")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("All learned models on the same test set")
ax.legend()
plt.show()
```

# 13. Student investigation — 15 minutes

Work in pairs. Choose **one** investigation.

### A. Threshold policy

Find a validation threshold for the MLP that achieves:

\[
\mathrm{Recall}\ge 0.90.
\]

Report the test-set precision and compare with the random forest.

### B. Remove a suspicious feature

Remove `survey_code` from the model inputs.

Retrain **one** model.

Does performance improve, degrade, or remain unchanged?

What does that say about possible provenance information?

### C. Change model complexity

Modify the MLP:

- fewer hidden units, or
- one additional hidden layer.

Does validation performance change meaningfully?

### D. Remove class weighting

Train the MLP without positive-class weighting.

Which metric changes the most?

---

## Report back

Each pair should answer in **three sentences**:

1. What did you change?
2. What happened?
3. Why might that matter scientifically?

```python
# Student workspace
#
# Choose A, B, C, or D from the challenge above.
#
# Write your code here.
```

# 14. Scientific synthesis

The lesson is not “neural networks are better” or “classical ML is better.”

The lesson is:

\[
\boxed{
\text{model complexity must earn its place through validation}
}
\]

For structured/tabular astrophysical data, a strong tree ensemble may rival or outperform a small
neural network. That is not a failure of deep learning—it is information about the geometry and sample
size of the problem.

### Final discussion

> **Was deep learning necessary for this dataset?**

A defensible answer should mention:

- predictive performance;
- rare-class recall/precision;
- stability;
- interpretability;
- computational complexity;
- scientific use case.

### Bridge to Wednesday

Today all models saw a hand-designed feature vector.

On Wednesday we ask a deeper question:

> **What if the representation itself is the main limitation?**

We will compare engineered representations with models that learn directly from spectra and time
series.
