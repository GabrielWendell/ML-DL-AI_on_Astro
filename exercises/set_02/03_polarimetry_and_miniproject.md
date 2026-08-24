# Exercise Set 2 — Notebook 3
## Polarimetry, advanced AI, and the final mini-project

**Course:** Machine Learning, Deep Learning, and AI for Astrophysics  
**Week 2 — Friday (Slots 11–12)**

This notebook uses stellar polarimetry as a research-style case study. It is intentionally built around a deeper question than “which classifier wins?”:

> **How does the physical representation of polarization affect what a machine-learning model can infer about circumstellar geometry?**

The second half of the notebook is a mini-project planning scaffold for the final course activity.


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

# Part A — Polarimetric inference

Linear polarization can be represented using normalized Stokes parameters

\[
q = Q/I,\qquad u = U/I,
\]

or polarization amplitude and position angle

\[
p = \sqrt{q^2+u^2},\qquad
\chi = \frac{1}{2}\operatorname{atan2}(u,q).
\]

The angle has a 180° degeneracy, which is why the physically natural angular encoding involves \(2\chi\). The notebook will compare multiple input representations rather than assuming that mathematically equivalent coordinates are equally convenient for ML.

## 1. Generate a controlled polarimetric dataset

We simulate a circumstellar disk whose intrinsic polarization increases approximately with `sin(i)^2`, modulated by disk density. The polarization orientation is random on the sky. We also include heteroscedastic measurement noise and a survey-dependent interstellar-polarization offset to create a realistic nuisance variable.

This is a **teaching simulator**, not a publication-grade radiative-transfer model.


```python
n = 1800
survey = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])

# Isotropic orientation: cos(i) uniform on [0,1].
cos_i = rng.uniform(0, 1, n)
inclination = np.degrees(np.arccos(cos_i))
disk_density = rng.lognormal(mean=-0.1, sigma=0.45, size=n)
psi = rng.uniform(0, np.pi, n)  # sky position angle, 180-degree periodicity

# Intrinsic polarization amplitude (toy model).
p0 = 0.010 * disk_density * (np.sin(np.radians(inclination)) ** 2) / (1 + 0.35 * disk_density)

# Three bands with a mild wavelength-dependent response.
band_scale = {"B": 0.90, "V": 1.00, "R": 1.08}

# Survey-dependent interstellar-polarization offsets (nuisance).
isp_q = np.where(survey == "A", 0.0012, -0.0010)
isp_u = np.where(survey == "A", -0.0007, 0.0011)

magnitude = rng.normal(15.2, 1.1, n)
noise_sigma = 0.00045 + 0.00012 * np.clip(magnitude - 14, 0, None)

pol = pd.DataFrame({
    "survey": survey,
    "inclination_deg": inclination,
    "disk_density_proxy": disk_density + rng.normal(0, 0.08, n),
    "photometric_excess": 0.18 * disk_density + rng.normal(0, 0.05, n),
    "magnitude": magnitude,
    "pol_sigma": noise_sigma,
})

for band, scale in band_scale.items():
    p = p0 * scale
    q = p * np.cos(2 * psi) + isp_q + rng.normal(0, noise_sigma)
    u = p * np.sin(2 * psi) + isp_u + rng.normal(0, noise_sigma)
    pol[f"q_{band}"] = q
    pol[f"u_{band}"] = u
    pol[f"p_{band}"] = np.sqrt(q*q + u*u)
    pol[f"chi_{band}"] = 0.5 * np.arctan2(u, q)

pol.head()

```


```python
plt.figure(figsize=(6, 6))
plt.scatter(pol["q_V"], pol["u_V"], s=8, alpha=0.35)
plt.xlabel("q_V")
plt.ylabel("u_V")
plt.title("Synthetic q-u plane")
plt.axis("equal")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(pol["inclination_deg"], pol["p_V"], s=8, alpha=0.3)
plt.xlabel("Inclination [deg]")
plt.ylabel("Observed p_V")
plt.title("Polarization amplitude vs inclination")
plt.show()

```

### Exercise 1 — Physical audit

1. Explain the broad trend in `p_V` versus inclination.
2. Why is the relation not one-to-one?
3. Inspect the q–u plane separately for Survey A and Survey B. What nuisance effect is visible?
4. Which variables should be treated as measurements, which as uncertainty descriptors, and which as provenance?
5. Why would directly feeding `survey` into the model be scientifically risky even if it improves predictive performance?

## 2. Three representations of the same polarimetric information

We compare:

- **Representation A:** raw Stokes coordinates `(q, u)` in three bands;
- **Representation B:** `(p, chi)` with the angle used naively;
- **Representation C:** `(p, sin(2chi), cos(2chi))`, which respects the 180° periodicity.

A fourth multimodal representation adds a photometric disk-density proxy.


```python
for band in ["B", "V", "R"]:
    pol[f"sin2chi_{band}"] = np.sin(2 * pol[f"chi_{band}"])
    pol[f"cos2chi_{band}"] = np.cos(2 * pol[f"chi_{band}"])

features_A = [f"{s}_{b}" for b in ["B", "V", "R"] for s in ["q", "u"]]
features_B = [f"{s}_{b}" for b in ["B", "V", "R"] for s in ["p", "chi"]]
features_C = [f"{s}_{b}" for b in ["B", "V", "R"] for s in ["p", "sin2chi", "cos2chi"]]
features_D = features_C + ["photometric_excess", "disk_density_proxy", "pol_sigma"]

# Turn inclination into a three-class problem for a compact teaching benchmark.
def inclination_class(x):
    return np.where(x < 30, 0, np.where(x < 65, 1, 2))

pol["geometry_class"] = inclination_class(pol["inclination_deg"].to_numpy())
pol["geometry_class"].value_counts(normalize=True).sort_index()

```


```python
from sklearn.ensemble import RandomForestClassifier

train_df, test_df = train_test_split(
    pol, test_size=0.25, stratify=pol["geometry_class"], random_state=RANDOM_STATE
)

def fit_representation(cols):
    model = RandomForestClassifier(
        n_estimators=140, min_samples_leaf=3, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(train_df[cols], train_df["geometry_class"])
    pred = model.predict(test_df[cols])
    return model, {
        "balanced_accuracy": balanced_accuracy_score(test_df["geometry_class"], pred),
        "mcc": matthews_corrcoef(test_df["geometry_class"], pred),
    }

rows = []
models = {}
for name, cols in {
    "A: q,u": features_A,
    "B: p,chi(raw)": features_B,
    "C: p,periodic-angle": features_C,
    "D: polarimetry+photometry": features_D,
}.items():
    model, metrics = fit_representation(cols)
    models[name] = model
    rows.append({"representation": name, **metrics})

pd.DataFrame(rows)

```

### Exercise 2 — Representation matters

1. Compare Representations B and C. Why can a periodic encoding of the angle help even though no information was intentionally added?
2. Compare A and C. Which coordinate system is more natural for the model, and is the answer architecture-dependent?
3. Compare C and D. Does adding photometric information help break the inclination–disk-density degeneracy?
4. Train on Survey A and test on Survey B. Which representation transfers best?
5. Remove the survey-dependent interstellar-polarization offset from the simulator and repeat the comparison. What changed?

## 3. Regression: infer inclination directly

Classification is convenient, but the underlying physical quantity is continuous. We now ask the model to estimate inclination in degrees.


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

reg = RandomForestRegressor(
    n_estimators=160, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
)
reg.fit(train_df[features_D], train_df["inclination_deg"])
pred_i = reg.predict(test_df[features_D])

print("MAE [deg]:", mean_absolute_error(test_df["inclination_deg"], pred_i))
print("RMSE [deg]:", mean_squared_error(test_df["inclination_deg"], pred_i) ** 0.5)

plt.figure(figsize=(6, 5))
plt.scatter(test_df["inclination_deg"], pred_i, s=10, alpha=0.35)
lims = [0, 90]
plt.plot(lims, lims, linestyle="--")
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("True inclination [deg]")
plt.ylabel("Predicted inclination [deg]")
plt.title("Polarimetric inclination regression")
plt.show()

```

### Exercise 3 — Error structure is astrophysics

1. Bin the residuals by true inclination and calculate MAE in each bin.
2. Bin the residuals by `pol_sigma` and by `disk_density_proxy`.
3. Identify the physical regimes where inclination is intrinsically difficult to infer.
4. Explain why reporting one global RMSE would hide scientifically important information.
5. Propose a probabilistic model that predicts both inclination and an object-dependent uncertainty.

## 4. Measurement uncertainty and polarization bias

Because

\[
p=\sqrt{q^2+u^2}\ge 0,
\]

noise in \(q\) and \(u\) does not transform into symmetric Gaussian noise in \(p\), particularly at low signal-to-noise ratio.

### Exercise 4 — Monte Carlo uncertainty propagation

Choose 20 low-polarization objects. For each object:

1. sample many realizations of \(q\) and \(u\) from the reported measurement uncertainty;
2. transform every realization to \(p\);
3. inspect the resulting distribution;
4. compare its mean/median with the naive observed \(p\);
5. explain the consequences for using `p` directly as a feature.

**Advanced:** propagate these Monte Carlo samples through the trained regression model and estimate prediction uncertainty induced by measurement noise.


```python
# Starter example for one object.
row = test_df.sort_values("p_V").iloc[5]
mc_n = 3000
q_mc = rng.normal(row["q_V"], row["pol_sigma"], mc_n)
u_mc = rng.normal(row["u_V"], row["pol_sigma"], mc_n)
p_mc = np.sqrt(q_mc*q_mc + u_mc*u_mc)

print("Observed p_V:", row["p_V"])
print("MC mean p_V:", p_mc.mean())
print("MC median p_V:", np.median(p_mc))

plt.figure(figsize=(6, 4))
plt.hist(p_mc, bins=45)
plt.xlabel("Monte Carlo p_V")
plt.ylabel("Count")
plt.title("Non-Gaussian transformed uncertainty")
plt.show()

```

# Part B — Optional advanced AI extension

## Autoencoder-style anomaly detection in polarimetric feature space

A useful research extension is to train an autoencoder or other self-supervised representation model on a ``normal'' population and rank objects by reconstruction error or latent-space distance.

Possible injected anomalies for this teaching simulator include:

- wavelength-dependent rotations of polarization angle;
- abrupt q–u excursions between epochs;
- unusually strong polarization for the inferred disk-density proxy;
- a source whose q–u pattern is inconsistent with the dominant survey offsets.

### Exercise 5 — Design before coding

Before implementing an autoencoder, answer:

1. What constitutes a ``normal'' training population?
2. How would you avoid defining anomalies merely as low-S/N objects?
3. Which representation would you feed to the autoencoder?
4. Which validation strategy can evaluate anomaly detection when labels are sparse or incomplete?
5. What astrophysical follow-up would turn a high anomaly score into a useful scientific result?

# Part C — Final mini-project menu

Students work individually or in pairs. The expected final presentation is short and scientific: **question → data → representation → model → validation → interpretation → limitations**.

### Project 1 — Variable-star classification
- **Target:** variability class or rare-object candidate status.
- **Baseline:** engineered time-series features + random forest.
- **Advanced model:** 1D CNN, GRU, TCN, or transformer.
- **Key question:** does the sequence representation learn information beyond period/amplitude features?

### Project 2 — Galaxy morphology
- **Target:** morphology class or structural parameter.
- **Baseline:** catalog morphology features.
- **Advanced model:** CNN / transfer learning.
- **Key question:** how robust is the result to PSF, redshift, orientation, and image quality?

### Project 3 — Photometric redshift
- **Target:** continuous redshift.
- **Baseline:** linear/boosted regression.
- **Advanced model:** MLP or probabilistic regressor.
- **Key question:** where does the model extrapolate poorly and how should uncertainty be reported?

### Project 4 — Stellar spectra
- **Target:** class, atmospheric parameter, or emission-line state.
- **Baseline:** engineered line/continuum features.
- **Advanced model:** 1D CNN or transformer.
- **Key question:** which wavelength regions actually drive the inference?

### Project 5 — Exoplanet transit classification
- **Target:** transit candidate vs non-transit.
- **Baseline:** transit-shape descriptors + tree model.
- **Advanced model:** temporal CNN.
- **Key question:** how do class imbalance and physically valid augmentation affect candidate recovery?

### Project 6 — Astrophysical anomaly detection
- **Target:** no fixed supervised label required.
- **Baseline:** robust distances / isolation forest.
- **Advanced model:** autoencoder or self-supervised embedding.
- **Key question:** are the highest-scoring anomalies scientifically unusual or simply data-quality failures?

### Project 7 — **Polarimetry: circumstellar-disk geometry**
- **Target:** inclination regression or pole/intermediate/edge classification.
- **Inputs:** q/u, p/χ, multi-band polarimetry, optional photometry.
- **Baseline:** random forest / gradient boosting.
- **Advanced model:** MLP, sequence model for multi-epoch q–u trajectories, or multimodal fusion.
- **Core experiment:** compare coordinate representations and quantify transfer across observing sources.
- **Key question:** what information about geometry is genuinely present in polarimetry, and where do degeneracies dominate?

## Mini-project planning worksheet

Before coding, complete the following in a Markdown cell or short proposal:

1. **Scientific question:** one sentence.
2. **Prediction target:** what exactly is being inferred?
3. **Unit of independence:** what is one statistically independent object?
4. **Input representation:** raw data, engineered features, or both?
5. **Baseline:** the simplest credible model.
6. **Advanced model:** why is additional complexity scientifically justified?
7. **Split strategy:** how will you prevent leakage?
8. **Primary metric:** why does it match the scientific use case?
9. **Domain-shift test:** what new survey/instrument/quality regime would challenge the model?
10. **Uncertainty/calibration:** what would make the output scientifically actionable?
11. **Interpretation:** which success/failure cases will you inspect?
12. **Minimum publishable negative result:** what could you learn even if the advanced model does not win?

### Final presentation rule
Do **not** organize the talk around the architecture. Organize it around the scientific inference problem.
