# Exercise Set 2 — Notebook 2
## Deep learning for spectra and astronomical time series

**Course:** Machine Learning, Deep Learning, and AI for Astrophysics  
**Week 2 — Wednesday (Slots 9–10)**

This notebook contains two compact laboratories. Both use generated data so that the notebook is reproducible offline, but the workflow is designed to transfer directly to real spectra and light curves.

### Part A — Spectra
Compare engineered spectral descriptors with a 1D CNN operating on the full spectrum.

### Part B — Time series
Compare summary features with a 1D temporal model, while explicitly tracking the role of cadence and temporal representation.


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

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(RANDOM_STATE)

```

# Part A — Stellar spectra

We simulate normalized optical spectra on a common wavelength grid. The positive class has stronger Hα emission on average, while both classes also contain temperature-dependent continuum and absorption-line structure. The point is not astrophysical realism at publication level; it is to create a controlled benchmark where representation choice can be studied.


```python
n_spec = 800
wavelength = np.linspace(4000.0, 7000.0, 320)
y_spec = rng.binomial(1, 0.32, n_spec)

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

spectra = []
for y in y_spec:
    slope = rng.normal(0.0, 0.06)
    continuum = 1.0 + slope * (wavelength - wavelength.mean()) / np.ptp(wavelength)
    flux = continuum.copy()
    # Common absorption features.
    flux -= rng.uniform(0.03, 0.10) * gaussian(wavelength, 4861.0, rng.uniform(5, 10))
    flux -= rng.uniform(0.02, 0.08) * gaussian(wavelength, 4340.0, rng.uniform(5, 10))
    # H-alpha: emission is statistically stronger for the positive class.
    ha_amp = rng.normal(0.15 if y else -0.03, 0.07)
    flux += ha_amp * gaussian(wavelength, 6563.0, rng.uniform(5, 14))
    flux += rng.normal(0.0, 0.018, size=wavelength.size)
    spectra.append(flux)

spectra = np.asarray(spectra, dtype=np.float32)
print(spectra.shape, y_spec.mean())

for idx in [0, 1, 2, 3]:
    plt.figure(figsize=(7, 3))
    plt.plot(wavelength, spectra[idx])
    plt.xlabel("Wavelength")
    plt.ylabel("Normalized flux")
    plt.title(f"Example spectrum — class {y_spec[idx]}")
    plt.show()

```

### Exercise 1 — Engineer a physically motivated baseline

Construct at least three scalar descriptors. Suggested examples:

- an Hα equivalent-width proxy;
- continuum slope;
- flux variance in a line-free region;
- line asymmetry or width.

Train a simple classifier on those descriptors before touching the CNN. Explain what astrophysical information each feature is intended to capture.


```python
def band_mean(spec, lo, hi):
    m = (wavelength >= lo) & (wavelength <= hi)
    return spec[:, m].mean(axis=1)

ha_core = band_mean(spectra, 6545, 6580)
cont_blue = band_mean(spectra, 6200, 6300)
cont_red = band_mean(spectra, 6800, 6900)
continuum_slope = cont_red - cont_blue
line_contrast = ha_core - 0.5 * (cont_blue + cont_red)
blue_variance = spectra[:, (wavelength > 5000) & (wavelength < 5500)].var(axis=1)

spec_features = pd.DataFrame({
    "line_contrast": line_contrast,
    "continuum_slope": continuum_slope,
    "blue_variance": blue_variance,
})

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    spec_features, y_spec, test_size=0.25, stratify=y_spec, random_state=RANDOM_STATE
)

from sklearn.linear_model import LogisticRegression
spec_baseline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))
])
spec_baseline.fit(Xs_train, ys_train)
prob = spec_baseline.predict_proba(Xs_test)[:, 1]
pred = (prob >= 0.5).astype(int)
print(classification_report(ys_test, pred, digits=3))
print("Average precision:", average_precision_score(ys_test, prob))

```

### Exercise 2 — Stress-test the feature baseline

1. Remove the Hα-related descriptor. How much performance remains?
2. Increase the noise level in the generator and rerun the baseline.
3. Explain whether a strong Hα feature makes the task too easy and how you would design a more scientifically interesting benchmark.

## A 1D CNN on the full spectrum

A convolution assumes that local patterns matter and that similar local filters can be reused across wavelength. This is a useful inductive bias for spectra, but it does not automatically encode all spectroscopy-specific physics.


```python
idx_train, idx_test = train_test_split(
    np.arange(n_spec), test_size=0.25, stratify=y_spec, random_state=RANDOM_STATE
)

# Normalize using training spectra only.
mean_spec = spectra[idx_train].mean(axis=0, keepdims=True)
std_spec = spectra[idx_train].std(axis=0, keepdims=True) + 1e-6
spec_norm = (spectra - mean_spec) / std_spec

Xtr = torch.tensor(spec_norm[idx_train, None, :], dtype=torch.float32)
Xte = torch.tensor(spec_norm[idx_test, None, :], dtype=torch.float32)
ytr = torch.tensor(y_spec[idx_train, None], dtype=torch.float32)
yte = y_spec[idx_test]
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

class SpectralCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.head(h)

cnn = SpectralCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(cnn.parameters(), lr=2e-3, weight_decay=1e-4)

for epoch in range(4):
    cnn.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(cnn(xb), yb)
        loss.backward()
        optimizer.step()

cnn.eval()
with torch.no_grad():
    cnn_prob = torch.sigmoid(cnn(Xte)).numpy().ravel()
cnn_pred = (cnn_prob >= 0.5).astype(int)
print(classification_report(yte, cnn_pred, digits=3))
print("Average precision:", average_precision_score(yte, cnn_prob))

```

### Exercise 3 — CNN interpretation

1. Compare the CNN with the feature baseline on the **same split**.
2. Change the kernel width. What spectral scale does the kernel roughly correspond to?
3. Mask the Hα region at test time and quantify the performance drop.
4. Explain why this masking experiment is more informative than merely stating that the CNN has high accuracy.
5. Optional: inspect first-layer filters or compute gradient-based saliency, but state clearly what such diagnostics can and cannot establish.

# Part B — Astronomical time series

We now generate fixed-length light curves for teaching purposes. Class 1 contains stronger periodic variability plus slow secular modulation; class 0 contains weaker stochastic variability. In real survey data the next complications would be irregular sampling, missing epochs, heteroscedastic errors, and source-dependent cadence.


```python
n_lc = 700
n_t = 128
time = np.linspace(0, 60, n_t)
y_lc = rng.binomial(1, 0.28, n_lc)
curves = np.empty((n_lc, n_t), dtype=np.float32)

for i, y in enumerate(y_lc):
    period = rng.uniform(1.5, 8.0)
    phase = rng.uniform(0, 2*np.pi)
    amp = rng.uniform(0.04, 0.10) if y == 0 else rng.uniform(0.10, 0.24)
    signal = amp * np.sin(2*np.pi*time/period + phase)
    if y:
        signal += rng.uniform(0.04, 0.10) * np.sin(2*np.pi*time/rng.uniform(20, 45) + phase/2)
        center = rng.uniform(15, 45)
        signal += rng.uniform(0.05, 0.14) * np.exp(-0.5*((time-center)/rng.uniform(2, 6))**2)
    signal += rng.normal(0, 0.045, n_t)
    curves[i] = signal

for idx in [0, 1, 2, 3]:
    plt.figure(figsize=(7, 3))
    plt.plot(time, curves[idx])
    plt.xlabel("Time")
    plt.ylabel("Relative flux")
    plt.title(f"Example light curve — class {y_lc[idx]}")
    plt.show()

```

### Exercise 4 — Compare representations

Create a feature vector for every light curve using at least:

- standard deviation or robust amplitude;
- skewness;
- lag-1 autocorrelation;
- strongest Fourier/periodogram power;
- a slow-trend descriptor.

Train a random forest and record balanced accuracy, MCC, and average precision.


```python
from scipy.stats import skew
from scipy.signal import periodogram
from sklearn.ensemble import RandomForestClassifier

def lc_features(curve):
    ac = np.corrcoef(curve[:-1], curve[1:])[0, 1]
    _, pxx = periodogram(curve)
    slow = np.mean(curve[-32:]) - np.mean(curve[:32])
    return [np.std(curve), skew(curve), ac, np.max(pxx[1:]), slow]

F = np.asarray([lc_features(c) for c in curves])
idx_train, idx_test = train_test_split(
    np.arange(n_lc), test_size=0.25, stratify=y_lc, random_state=RANDOM_STATE
)
rf_lc = RandomForestClassifier(
    n_estimators=120, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
)
rf_lc.fit(F[idx_train], y_lc[idx_train])
rf_prob = rf_lc.predict_proba(F[idx_test])[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)
print(classification_report(y_lc[idx_test], rf_pred, digits=3))
print("MCC:", matthews_corrcoef(y_lc[idx_test], rf_pred))
print("Average precision:", average_precision_score(y_lc[idx_test], rf_prob))

```

## A compact temporal CNN

Here the input is the full sequence. The model is intentionally small so that you can ask whether end-to-end representation learning adds value over the engineered baseline.


```python
curve_mean = curves[idx_train].mean()
curve_std = curves[idx_train].std() + 1e-6
curves_norm = (curves - curve_mean) / curve_std

Xtr = torch.tensor(curves_norm[idx_train, None, :], dtype=torch.float32)
Xte = torch.tensor(curves_norm[idx_test, None, :], dtype=torch.float32)
ytr = torch.tensor(y_lc[idx_train, None], dtype=torch.float32)
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        return self.head(self.encoder(x).squeeze(-1))

tcnn = TemporalCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(tcnn.parameters(), lr=2e-3, weight_decay=1e-4)
for epoch in range(4):
    tcnn.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(tcnn(xb), yb)
        loss.backward()
        optimizer.step()

tcnn.eval()
with torch.no_grad():
    tcnn_prob = torch.sigmoid(tcnn(Xte)).numpy().ravel()
tcnn_pred = (tcnn_prob >= 0.5).astype(int)
print(classification_report(y_lc[idx_test], tcnn_pred, digits=3))
print("MCC:", matthews_corrcoef(y_lc[idx_test], tcnn_pred))
print("Average precision:", average_precision_score(y_lc[idx_test], tcnn_prob))

```

### Exercise 5 — Time-series ablations

Perform at least three of the following:

1. Randomly permute the time order at test time. What happens?
2. Crop every test curve to half its duration and decide how to handle the changed length.
3. Increase observational noise only in the test set.
4. Remove the slow-trend feature from the classical baseline.
5. Train the temporal CNN after shuffling each training curve independently.
6. Compare a model using only periodogram information with one using the raw sequence.

For each ablation, state which physical or statistical assumption is being tested.

## Final Wednesday synthesis

Write a short comparison table with rows = {feature baseline, deep model} and columns = {data representation, inductive bias, main strength, main weakness, likely failure under domain shift}.

Then answer:

> **When does deep learning add scientific value rather than merely model capacity?**

Your answer should refer to representation, sample size, transfer, uncertainty, and the strength of the classical baseline.
