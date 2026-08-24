# Week 2 — Wednesday Hands-on
## Representations: Stellar Spectra and Astronomical Time Series

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Format:** live Jupyter laboratory  
**Total duration:** approximately 100–120 minutes (two 50–60 minute slots)

### Central question

> **How much of model performance is actually a consequence of how we represent the astrophysical data?**

Monday kept the representation fixed and changed the model. Today we do almost the opposite:
we keep the scientific target and validation protocol fixed while changing what the algorithm is allowed
to see.

\[
\boxed{
\text{physical signal}
\rightarrow
\text{representation}
\rightarrow
\text{model}
\rightarrow
\text{scientific inference}
}
\]

---

## Learning objectives

By the end of the class, students should be able to:

1. distinguish **physical data** from a **computational representation** of those data;
2. explain why two mathematically related representations can lead to different ML behavior;
3. construct compact engineered representations of spectra and light curves;
4. use PCA as a linear learned representation;
5. train a 1D CNN directly on a stellar spectrum;
6. identify which wavelength regions matter through a controlled ablation;
7. handle irregular time sampling explicitly rather than hiding it;
8. compare time-domain features, frequency-domain information, and resampled raw sequences;
9. make fair comparisons using identical train/validation/test objects;
10. decide whether an apparent model improvement is really an **architecture effect** or a
    **representation effect**.

---

## Class rhythm

| Approx. time | Activity |
|---:|---|
| 0–10 min | Representation as a scientific choice |
| 10–25 min | Synthetic spectra + physical audit |
| 25–40 min | PCA/features + classical baseline |
| 40–55 min | 1D CNN + wavelength ablation |
| **break / transition** | |
| 55–70 min | Irregular light curves + visualization |
| 70–85 min | Engineered time-domain features |
| 85–100 min | Frequency-domain representation |
| 100–110 min | Resampled raw-sequence model |
| 110–118 min | Student representation challenge |
| 118–120 min | Scientific synthesis |

As on Monday, use:

\[
\boxed{\text{Predict} \rightarrow \text{Run} \rightarrow \text{Explain}}
\]

but today students should increasingly make the experimental choices themselves.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score,
    confusion_matrix, precision_recall_curve
)

from scipy.signal import lombscargle
from scipy.stats import skew

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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

print("PyTorch:", torch.__version__)
```

# Slot 1 — Stellar spectra: what should the model see?

A stellar spectrum is physically observed as something like

\[
\{(\lambda_j, f_j, \sigma_j)\}_{j=1}^{L}.
\]

But an ML model could receive:

- equivalent widths or hand-designed line indices;
- PCA coefficients;
- the complete normalized flux vector;
- selected wavelength windows;
- a learned embedding from a 1D CNN.

These are **not equivalent computational problems**, even when they originate from the same photons.

### Teaching problem

We will classify a synthetic stellar sample into

\[
y=
\begin{cases}
0,&\text{metal-poor-like spectrum},\\
1,&\text{metal-rich-like spectrum}.
\end{cases}
\]

The spectra also vary in effective-temperature proxy, radial-velocity shift, line broadening, and
signal-to-noise ratio. These are nuisance variables for the classification target.

The key question is:

> Can a compact representation preserve the abundance-sensitive line information while suppressing
> nuisance variation?

## 1. Generate a controlled spectral dataset

The generator contains several absorption features. Their depths depend on a continuous metallicity-like
latent variable, while temperature proxy, broadening, radial velocity, and noise create overlap.

The binary label is derived from the sign of the latent metallicity-like variable.

This is not intended as a realistic stellar atmosphere model. It is a **controlled teaching model**
whose physical signal and nuisance variables are known.

```python
def gaussian_line(wave, center, sigma):
    return np.exp(-0.5 * ((wave - center) / sigma) ** 2)

def make_spectral_dataset(n=1600, n_pix=512, seed=42):
    rng = np.random.default_rng(seed)
    wave = np.linspace(4100.0, 6800.0, n_pix)

    # Abundance-sensitive and nuisance-sensitive line centers.
    metal_lines = np.array([4383.0, 5175.0, 5270.0, 5892.0, 6495.0])
    temp_lines = np.array([4340.0, 4861.0, 6563.0])

    X = np.empty((n, n_pix), dtype=np.float32)
    meta = []

    for i in range(n):
        metallicity = rng.normal(0.0, 0.75)
        label = int(metallicity > 0.0)

        temp_proxy = rng.uniform(-1.0, 1.0)
        rv_kms = rng.normal(0.0, 55.0)
        broadening = rng.uniform(1.0, 2.4)
        snr = rng.uniform(25.0, 100.0)

        # Small continuum curvature that will largely be removed by normalization.
        x = (wave - wave.mean()) / np.ptp(wave)
        continuum = 1.0 + 0.035 * temp_proxy * x + 0.015 * x**2
        flux = continuum.copy()

        shift_factor = 1.0 + rv_kms / 299792.458

        for center in metal_lines:
            depth = np.clip(0.045 + 0.030 * metallicity + rng.normal(0, 0.006), 0.008, 0.14)
            flux -= depth * gaussian_line(wave, center * shift_factor, 5.0 * broadening)

        for center in temp_lines:
            depth = np.clip(0.055 + 0.025 * temp_proxy + rng.normal(0, 0.006), 0.012, 0.13)
            flux -= depth * gaussian_line(wave, center * shift_factor, 6.0 * broadening)

        flux += rng.normal(0.0, 1.0 / snr, size=n_pix)

        # Simple per-spectrum pseudo-continuum normalization.
        norm = np.median(flux)
        flux = flux / norm

        X[i] = flux.astype(np.float32)
        meta.append({
            "object_id": f"SPEC_{i:05d}",
            "target": label,
            "metallicity_latent": metallicity,
            "temp_proxy": temp_proxy,
            "rv_kms": rv_kms,
            "broadening": broadening,
            "snr": snr,
        })

    return wave, X, pd.DataFrame(meta), metal_lines

wave, X_spec, spec_meta, metal_lines = make_spectral_dataset()
print(X_spec.shape)
spec_meta.head()
```

## 2. Inspect before learning

Ask the class:

> If the metal-sensitive lines become deeper in one class, should a CNN automatically outperform
> equivalent-width features?

Not necessarily. If the hand-designed features already isolate the relevant physics, a simple classifier
may be hard to beat.

We first visualize spectra at different labels and S/N.

```python
print(spec_meta["target"].value_counts())
print(spec_meta.groupby("target")[["metallicity_latent", "temp_proxy", "rv_kms", "snr"]].mean())

fig, ax = plt.subplots(figsize=(10, 4.2))
for label, name in [(0, "metal-poor-like"), (1, "metal-rich-like")]:
    idx = spec_meta.index[spec_meta["target"] == label][:4]
    ax.plot(wave, X_spec[idx].mean(axis=0), label=name)
ax.set_xlabel("Wavelength")
ax.set_ylabel("Normalized flux")
ax.set_title("Mean synthetic spectra by class")
ax.legend()
plt.show()
```

```python
train_idx, temp_idx = train_test_split(
    np.arange(len(spec_meta)),
    test_size=0.40,
    stratify=spec_meta["target"],
    random_state=SEED,
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=spec_meta.loc[temp_idx, "target"],
    random_state=SEED,
)

y_spec = spec_meta["target"].to_numpy()
print("split sizes:", len(train_idx), len(val_idx), len(test_idx))
```

## 3. Representation A — Hand-designed line indices

For a line centered at \(\lambda_0\), construct a simple absorption-strength proxy

\[
I(\lambda_0)
=
1-\langle f_\lambda\rangle_{\lambda_0-\Delta\lambda}^{\lambda_0+\Delta\lambda}.
\]

This is intentionally simple: the representation encodes prior knowledge about where informative lines
should be.

Then train logistic regression.

### Predict

> What happens if the chosen line list misses a physically informative wavelength region?

```python
def line_index_matrix(X, wave, centers, half_width=12.0):
    feats = []
    for center in centers:
        m = np.abs(wave - center) <= half_width
        feats.append(1.0 - X[:, m].mean(axis=1))
    return np.column_stack(feats)

line_features = line_index_matrix(X_spec, wave, metal_lines)

scaler_line = StandardScaler()
Xtr_line = scaler_line.fit_transform(line_features[train_idx])
Xva_line = scaler_line.transform(line_features[val_idx])
Xte_line = scaler_line.transform(line_features[test_idx])

lr_lines = LogisticRegression(max_iter=2000, random_state=SEED)
lr_lines.fit(Xtr_line, y_spec[train_idx])

prob_lines = lr_lines.predict_proba(Xte_line)[:, 1]
pred_lines = (prob_lines >= 0.5).astype(int)

pd.Series(metrics_dict(y_spec[test_idx], pred_lines, prob_lines), name="line indices + logistic")
```

## 4. Representation B — PCA coefficients

PCA learns linear directions of large variance:

\[
X \approx ZW^\top,
\qquad
Z=XW.
\]

Unlike hand-designed line indices, PCA does not know which variance is astrophysically relevant to the
target.

A high-variance direction could describe:

- temperature;
- radial velocity;
- broadening;
- noise;
- metallicity.

### Predict → Run → Explain

> Must the first principal component be the most useful component for classification?

```python
# Standardize each wavelength using training spectra only.
spec_mean = X_spec[train_idx].mean(axis=0)
spec_std = X_spec[train_idx].std(axis=0) + 1e-6

Xtr_std = (X_spec[train_idx] - spec_mean) / spec_std
Xva_std = (X_spec[val_idx] - spec_mean) / spec_std
Xte_std = (X_spec[test_idx] - spec_mean) / spec_std

pca = PCA(n_components=20, random_state=SEED)
Ztr = pca.fit_transform(Xtr_std)
Zva = pca.transform(Xva_std)
Zte = pca.transform(Xte_std)

lr_pca = LogisticRegression(max_iter=2000, random_state=SEED)
lr_pca.fit(Ztr, y_spec[train_idx])

prob_pca = lr_pca.predict_proba(Zte)[:, 1]
pred_pca = (prob_pca >= 0.5).astype(int)

print("Explained variance (20 PCs):", pca.explained_variance_ratio_.sum().round(3))
pd.Series(metrics_dict(y_spec[test_idx], pred_pca, prob_pca), name="PCA + logistic")
```

```python
fig, ax = plt.subplots(figsize=(6.5, 5))
for label, name in [(0, "metal-poor-like"), (1, "metal-rich-like")]:
    m = y_spec[train_idx] == label
    ax.scatter(Ztr[m, 0], Ztr[m, 1], s=12, alpha=0.45, label=name)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("First two PCA coordinates")
ax.legend()
plt.show()
```

## 5. Representation C — Raw normalized spectrum + 1D CNN

A 1D CNN can learn local spectral motifs directly:

\[
f_\lambda
\rightarrow
\text{convolution}
\rightarrow
\text{local feature maps}
\rightarrow
\text{global representation}
\rightarrow
\hat y.
\]

The CNN has less explicit physical guidance than the hand-crafted line representation, but more
flexibility.

We train on exactly the same objects used by the classical baselines.

```python
class SpectralCNN(nn.Module):
    def __init__(self, n_pix):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(12, 24, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 16, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

Xtr_cnn = torch.tensor(Xtr_std[:, None, :], dtype=torch.float32)
Xva_cnn = torch.tensor(Xva_std[:, None, :], dtype=torch.float32)
Xte_cnn = torch.tensor(Xte_std[:, None, :], dtype=torch.float32)

ytr_cnn = torch.tensor(y_spec[train_idx, None], dtype=torch.float32)
yva_cnn = torch.tensor(y_spec[val_idx, None], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xtr_cnn, ytr_cnn), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(Xva_cnn, yva_cnn), batch_size=128)

spec_cnn = SpectralCNN(X_spec.shape[1])
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(spec_cnn.parameters(), lr=2e-3, weight_decay=1e-4)

history_spec = {"train": [], "val": []}
best_state = None
best_val = np.inf
patience = 12
bad = 0

for epoch in range(80):
    spec_cnn.train()
    total, nobs = 0.0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(spec_cnn(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
        nobs += len(xb)
    train_loss = total / nobs

    spec_cnn.eval()
    total, nobs = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            loss = loss_fn(spec_cnn(xb), yb)
            total += loss.item() * len(xb)
            nobs += len(xb)
    val_loss = total / nobs

    history_spec["train"].append(train_loss)
    history_spec["val"].append(val_loss)

    if val_loss < best_val - 1e-5:
        best_val = val_loss
        best_state = {k: v.detach().clone() for k, v in spec_cnn.state_dict().items()}
        bad = 0
    else:
        bad += 1
    if bad >= patience:
        break

spec_cnn.load_state_dict(best_state)
print("epochs:", len(history_spec["train"]), "best val:", round(best_val, 4))
```

```python
spec_cnn.eval()
with torch.no_grad():
    prob_cnn = torch.sigmoid(spec_cnn(Xte_cnn).squeeze(1)).numpy()
pred_cnn = (prob_cnn >= 0.5).astype(int)

pd.Series(metrics_dict(y_spec[test_idx], pred_cnn, prob_cnn), name="raw spectrum + CNN")
```

```python
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(history_spec["train"], label="training")
ax.plot(history_spec["val"], label="validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE loss")
ax.set_title("Spectral CNN learning curves")
ax.legend()
plt.show()
```

## 6. Wavelength ablation: turn interpretation into an experiment

Attribution maps can be useful, but a very direct scientific test is:

> **Remove a physically meaningful wavelength interval and measure what changes.**

We will mask one of the strongest abundance-sensitive regions and re-evaluate the already-trained CNN.

This is an **intervention on the input representation**.

If performance drops strongly, the result supports the claim that the model uses that spectral region.
It does not, by itself, establish causal stellar physics—but it is much more informative than merely
looking at the final accuracy.

```python
# Mask a strong abundance-sensitive region around 5175 Å only at test time.
Xte_masked = Xte_std.copy()
mask_region = np.abs(wave - 5175.0) <= 22.0
Xte_masked[:, mask_region] = 0.0  # zero = training-standardized mean flux

with torch.no_grad():
    prob_cnn_masked = torch.sigmoid(
        spec_cnn(torch.tensor(Xte_masked[:, None, :], dtype=torch.float32)).squeeze(1)
    ).numpy()

pred_cnn_masked = (prob_cnn_masked >= 0.5).astype(int)

ablation_table = pd.DataFrame([
    {"condition": "original spectrum", **metrics_dict(y_spec[test_idx], pred_cnn, prob_cnn)},
    {"condition": "5175 Å region masked", **metrics_dict(y_spec[test_idx], pred_cnn_masked, prob_cnn_masked)},
]).set_index("condition")

ablation_table.round(3)
```

```python
spectral_rows = [
    {"representation": "Line indices", "model": "Logistic regression",
     **metrics_dict(y_spec[test_idx], pred_lines, prob_lines)},
    {"representation": "20 PCA coefficients", "model": "Logistic regression",
     **metrics_dict(y_spec[test_idx], pred_pca, prob_pca)},
    {"representation": "Raw normalized spectrum", "model": "1D CNN",
     **metrics_dict(y_spec[test_idx], pred_cnn, prob_cnn)},
]
pd.DataFrame(spectral_rows).set_index(["representation", "model"]).round(3)
```

# Slot 1 synthesis

We compared:

\[
\boxed{
\text{line indices}
\quad\text{vs.}\quad
\text{PCA}
\quad\text{vs.}\quad
\text{raw spectrum + CNN}
}
\]

The important question is not which representation is “modern.”

Ask instead:

1. Which representation aligns with the relevant physics?
2. Which one suppresses nuisance information?
3. Which one requires the most data?
4. Which one is easiest to audit?
5. What scientific statement is supported by the wavelength ablation?

### Transition

Spectra live on an approximately ordered wavelength grid.

Astronomical light curves are often more difficult because the sampling itself is irregular.

So now representation becomes even more consequential.

# Slot 2 — Astronomical time series: irregular sampling is part of the data

For a light curve,

\[
x_i=\{(t_{ij}, f_{ij}, \sigma_{ij})\}_{j=1}^{n_i}.
\]

Different objects may have:

- different numbers of observations;
- different cadences;
- seasonal gaps;
- different uncertainty levels.

A model cannot ingest “an irregular light curve” without some representational decision.

Possible choices include:

\[
\boxed{
\text{summary features}
\quad
\text{frequency representation}
\quad
\text{resampled raw sequence}
}
\]

### Teaching problem

We classify:

- **quasi-periodic pulsator-like objects**;
- **aperiodic/red-noise-like variables**.

Amplitude distributions are intentionally overlapping, so trivial variance alone is not enough.

## 7. Generate irregular synthetic light curves

Each physical object receives its own irregular observing times.

For the periodic class, the signal contains a dominant oscillation plus a weaker harmonic.

For the aperiodic class, we construct a correlated stochastic process with comparable amplitude.

Noise and number of observations vary from object to object.

### Predict

> If we sort only by standard deviation, should the two classes separate cleanly?

```python
def ou_process(times, tau, sigma, rng):
    x = np.zeros(len(times))
    x[0] = rng.normal(0, sigma)
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        phi = np.exp(-dt / tau)
        var = sigma**2 * (1 - phi**2)
        x[i] = phi * x[i-1] + rng.normal(0, np.sqrt(max(var, 1e-12)))
    return x

def make_light_curves(n=1200, seed=123):
    rng = np.random.default_rng(seed)
    curves = []
    meta = []

    for i in range(n):
        label = int(rng.random() < 0.5)
        n_obs = int(rng.integers(70, 150))
        # Irregular cadence with two observing seasons.
        t1 = rng.uniform(0, 45, size=n_obs // 2)
        t2 = rng.uniform(65, 120, size=n_obs - n_obs // 2)
        t = np.sort(np.concatenate([t1, t2]))

        amp = rng.uniform(0.06, 0.16)
        noise = rng.uniform(0.015, 0.045)

        if label == 1:
            period = rng.uniform(3.0, 12.0)
            phase = rng.uniform(0, 2*np.pi)
            flux = (
                amp * np.sin(2*np.pi*t/period + phase)
                + 0.35*amp*np.sin(4*np.pi*t/period + 0.5*phase)
            )
        else:
            period = np.nan
            tau = rng.uniform(4.0, 18.0)
            flux = ou_process(t, tau=tau, sigma=amp*0.75, rng=rng)

        flux += rng.normal(0, noise, size=n_obs)
        flux -= np.median(flux)

        curves.append({"time": t, "flux": flux, "error": np.full(n_obs, noise)})
        meta.append({
            "object_id": f"LC_{i:05d}",
            "target": label,
            "n_obs": n_obs,
            "noise": noise,
            "true_period": period,
        })

    return curves, pd.DataFrame(meta)

curves, lc_meta = make_light_curves()
lc_meta["target"].value_counts()
```

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
for row, label in enumerate([0, 1]):
    ids = lc_meta.index[lc_meta["target"] == label][:2]
    for col, idx in enumerate(ids):
        c = curves[idx]
        axes[row, col].errorbar(c["time"], c["flux"], yerr=c["error"],
                                fmt=".", ms=3, alpha=0.8)
        axes[row, col].set_title(
            ("aperiodic/red-noise-like" if label == 0 else "quasi-periodic")
            + f" — {lc_meta.loc[idx, 'object_id']}"
        )
        axes[row, col].set_xlabel("Time")
        axes[row, col].set_ylabel("Relative flux")
plt.tight_layout()
plt.show()
```

```python
lc_train, lc_temp = train_test_split(
    np.arange(len(lc_meta)),
    test_size=0.40,
    stratify=lc_meta["target"],
    random_state=SEED,
)
lc_val, lc_test = train_test_split(
    lc_temp,
    test_size=0.50,
    stratify=lc_meta.loc[lc_temp, "target"],
    random_state=SEED,
)
y_lc = lc_meta["target"].to_numpy()
```

## 8. Representation A — Time-domain summary features

We begin with a compact vector:

- standard deviation;
- robust amplitude;
- skewness;
- lag-ordered correlation proxy;
- number of observations;
- cadence statistics.

This representation is cheap and interpretable.

But it compresses the entire temporal ordering into a few numbers.

### Question

> What physical information is definitely lost by this compression?

```python
def light_curve_features(curve):
    t = curve["time"]
    f = curve["flux"]
    dt = np.diff(t)

    # Ordered-neighbor correlation proxy.
    if len(f) > 2 and np.std(f[:-1]) > 0 and np.std(f[1:]) > 0:
        lag_corr = np.corrcoef(f[:-1], f[1:])[0, 1]
    else:
        lag_corr = 0.0

    return np.array([
        np.std(f),
        np.percentile(f, 95) - np.percentile(f, 5),
        skew(f),
        lag_corr,
        len(f),
        np.median(dt),
        np.std(dt),
    ], dtype=float)

X_feat = np.vstack([light_curve_features(c) for c in curves])

rf_lc = RandomForestClassifier(
    n_estimators=350,
    min_samples_leaf=3,
    random_state=SEED,
    n_jobs=-1,
)
rf_lc.fit(X_feat[lc_train], y_lc[lc_train])

prob_feat = rf_lc.predict_proba(X_feat[lc_test])[:, 1]
pred_feat = (prob_feat >= 0.5).astype(int)

pd.Series(metrics_dict(y_lc[lc_test], pred_feat, prob_feat),
          name="time-domain features + RF")
```

## 9. Representation B — Frequency-domain power

For irregularly sampled data, we can evaluate a Lomb–Scargle-like periodogram.

The model sees power as a function of frequency:

\[
\mathbf p_i =
\left(
P_i(\omega_1),\ldots,P_i(\omega_K)
\right).
\]

This explicitly emphasizes periodic structure.

We then use a simple logistic classifier.

### Predict

> If the scientific distinction is “periodic vs. aperiodic,” could a simple model on a good
> representation outperform a more flexible model on poor features?

```python
freq_grid = np.linspace(1/15.0, 1/2.0, 80)  # cycles per time unit
angular_freq = 2 * np.pi * freq_grid

def periodogram_representation(curve):
    t = curve["time"]
    f = curve["flux"] - np.mean(curve["flux"])
    # scipy.signal.lombscargle expects angular frequencies.
    p = lombscargle(t, f, angular_freq, normalize=True)
    return p

X_freq = np.vstack([periodogram_representation(c) for c in curves])

scaler_freq = StandardScaler()
Xtr_freq = scaler_freq.fit_transform(X_freq[lc_train])
Xva_freq = scaler_freq.transform(X_freq[lc_val])
Xte_freq = scaler_freq.transform(X_freq[lc_test])

lr_freq = LogisticRegression(max_iter=2000, random_state=SEED)
lr_freq.fit(Xtr_freq, y_lc[lc_train])

prob_freq = lr_freq.predict_proba(Xte_freq)[:, 1]
pred_freq = (prob_freq >= 0.5).astype(int)

pd.Series(metrics_dict(y_lc[lc_test], pred_freq, prob_freq),
          name="periodogram + logistic")
```

```python
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for label, name in [(0, "aperiodic"), (1, "quasi-periodic")]:
    idx = lc_train[y_lc[lc_train] == label]
    ax.plot(freq_grid, X_freq[idx].mean(axis=0), label=name)
ax.set_xlabel("Frequency")
ax.set_ylabel("Mean normalized Lomb–Scargle power")
ax.set_title("Average frequency representation")
ax.legend()
plt.show()
```

## 10. Representation C — Interpolated raw sequence

A standard neural network wants a fixed-size tensor.

One pragmatic approach is:

1. choose a common normalized time grid;
2. interpolate each irregular light curve;
3. add a mask/coverage channel if desired;
4. train a sequence model.

Here we use a compact 1D CNN on the interpolated flux sequence.

### Scientific caution

Interpolation is **not neutral**.

It imposes assumptions about the behavior between observations and may create artificial smoothness.
The model can also learn cadence artifacts unless coverage information is handled carefully.

```python
common_grid = np.linspace(0, 120, 160)

def interpolate_curve(curve):
    t = curve["time"]
    f = curve["flux"]
    # np.interp uses edge values outside the observed range; our simulated coverage spans both ends
    # reasonably well, but this is still a representational choice.
    y = np.interp(common_grid, t, f)
    y = (y - np.mean(y)) / (np.std(y) + 1e-6)
    return y.astype(np.float32)

X_seq = np.vstack([interpolate_curve(c) for c in curves])

class LightCurveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(12, 20, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(12),
            nn.Flatten(),
            nn.Linear(20 * 12, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x):
        return self.net(x)

Xtr_seq = torch.tensor(X_seq[lc_train, None, :], dtype=torch.float32)
Xva_seq = torch.tensor(X_seq[lc_val, None, :], dtype=torch.float32)
Xte_seq = torch.tensor(X_seq[lc_test, None, :], dtype=torch.float32)
ytr_seq = torch.tensor(y_lc[lc_train, None], dtype=torch.float32)
yva_seq = torch.tensor(y_lc[lc_val, None], dtype=torch.float32)

seq_train_loader = DataLoader(TensorDataset(Xtr_seq, ytr_seq), batch_size=64, shuffle=True)
seq_val_loader = DataLoader(TensorDataset(Xva_seq, yva_seq), batch_size=128)

lc_cnn = LightCurveCNN()
loss_fn_lc = nn.BCEWithLogitsLoss()
opt_lc = torch.optim.AdamW(lc_cnn.parameters(), lr=2e-3, weight_decay=1e-4)

history_lc = {"train": [], "val": []}
best_state_lc = None
best_val_lc = np.inf
bad = 0
patience = 10

for epoch in range(70):
    lc_cnn.train()
    total, nobs = 0.0, 0
    for xb, yb in seq_train_loader:
        opt_lc.zero_grad()
        loss = loss_fn_lc(lc_cnn(xb), yb)
        loss.backward()
        opt_lc.step()
        total += loss.item() * len(xb)
        nobs += len(xb)
    tr_loss = total / nobs

    lc_cnn.eval()
    total, nobs = 0.0, 0
    with torch.no_grad():
        for xb, yb in seq_val_loader:
            loss = loss_fn_lc(lc_cnn(xb), yb)
            total += loss.item() * len(xb)
            nobs += len(xb)
    va_loss = total / nobs

    history_lc["train"].append(tr_loss)
    history_lc["val"].append(va_loss)

    if va_loss < best_val_lc - 1e-5:
        best_val_lc = va_loss
        best_state_lc = {k: v.detach().clone() for k, v in lc_cnn.state_dict().items()}
        bad = 0
    else:
        bad += 1
    if bad >= patience:
        break

lc_cnn.load_state_dict(best_state_lc)

with torch.no_grad():
    prob_seq = torch.sigmoid(lc_cnn(Xte_seq).squeeze(1)).numpy()
pred_seq = (prob_seq >= 0.5).astype(int)

print("epochs:", len(history_lc["train"]))
pd.Series(metrics_dict(y_lc[lc_test], pred_seq, prob_seq),
          name="interpolated sequence + CNN")
```

```python
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(history_lc["train"], label="training")
ax.plot(history_lc["val"], label="validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE loss")
ax.set_title("Light-curve CNN learning curves")
ax.legend()
plt.show()
```

## 11. Fair representation comparison

Now compare three pipelines on the same test objects:

1. engineered time-domain features + random forest;
2. frequency representation + logistic regression;
3. interpolated raw sequence + 1D CNN.

The models differ, but the main scientific comparison is really between what information each
representation exposes.

Ask:

> If the frequency representation wins, is that evidence that logistic regression is a better model
> than a CNN?

No. It may simply mean that the frequency transform expresses the relevant physical distinction more
directly.

```python
ts_rows = [
    {"representation": "Time-domain summary features", "model": "Random forest",
     **metrics_dict(y_lc[lc_test], pred_feat, prob_feat)},
    {"representation": "Lomb–Scargle power vector", "model": "Logistic regression",
     **metrics_dict(y_lc[lc_test], pred_freq, prob_freq)},
    {"representation": "Interpolated raw sequence", "model": "1D CNN",
     **metrics_dict(y_lc[lc_test], pred_seq, prob_seq)},
]
ts_comparison = pd.DataFrame(ts_rows).set_index(["representation", "model"])
ts_comparison.round(3)
```

```python
curves_for_pr = [
    ("features + RF", prob_feat),
    ("periodogram + logistic", prob_freq),
    ("interpolated sequence + CNN", prob_seq),
]

fig, ax = plt.subplots(figsize=(6.8, 5))
for name, prob in curves_for_pr:
    p, r, _ = precision_recall_curve(y_lc[lc_test], prob)
    ap = average_precision_score(y_lc[lc_test], prob)
    ax.plot(r, p, label=f"{name} (AP={ap:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Same test objects, different representations")
ax.legend()
plt.show()
```

# 12. Student representation challenge — 15 minutes

Work in small groups. Choose **one** intervention.

### A. Remove the strongest spectral region

Repeat the spectral ablation with a different line window.

Predict first: small, moderate, or large performance drop?

### B. Reduce spectral resolution

Downsample every spectrum by a factor of 2 or 4.

Which model/representation is most robust?

### C. Destroy time ordering

Randomly permute the flux values within each test light curve while keeping the times fixed.

Which representation should suffer most?

### D. Reduce cadence

Keep only 50% of the observations in each light curve.

Which pipeline degrades most?

### E. Remove periodogram peak localization

Smooth the frequency representation strongly before classification.

Does the model need precise frequency localization or only broad-band power?

---

## Report back

Use four sentences:

1. What representation did you perturb?
2. What information did the intervention remove?
3. What happened to performance?
4. What does that imply about what the pipeline was using?

```python
# Student workspace
#
# Choose one intervention from A–E.
# State your prediction before writing code.
```

# 13. Scientific synthesis

Monday taught:

\[
\boxed{\text{model complexity must earn its place}}
\]

Wednesday adds:

\[
\boxed{\text{representation choice can matter as much as architecture choice}}
\]

A useful astrophysical ML workflow therefore asks, in order:

1. What physical information is present?
2. Which nuisance variables are present?
3. What representation exposes the relevant signal?
4. What information does that representation discard?
5. Only then: which model is appropriate?

### Final discussion

> **Which of today's improvements came from a better model, and which came from a better representation?**

### Bridge to Friday

Friday will make this even more explicit with polarimetry, where

\[
(q,u)
\quad\text{and}\quad
(p,\chi)
\]

encode closely related physical information but have very different geometry for a learning algorithm.

That class will then transition from guided experiments into the final mini-project design.
