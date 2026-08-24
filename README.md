# 🤖🤝🌌 Machine Learning, Deep Learning, and AI for Astrophysics

Repository-ready teaching materials for a two-week postgraduate mini-course aimed at Astrophysics students.

## Exercise materials

```text
exercises/
├── set_01/
│   ├── exercise_set_01.tex
│   └── exercise_set_01.pdf
└── set_02/
    ├── 01_catalog_ml_and_mlp.ipynb
    ├── 01_catalog_ml_and_mlp.md
    ├── 02_spectra_and_time_series.ipynb
    ├── 02_spectra_and_time_series.md
    ├── 03_polarimetry_and_miniproject.ipynb
    └── 03_polarimetry_and_miniproject.md
```

### Exercise Set 1 — Week 1

Theory- and methodology-oriented problems organized by the three class days:

- Monday: statistical learning, inference, metrics, validation, class imbalance, and leakage;
- Wednesday: neural networks, optimization, CNNs, sequence models, attention, and representation learning;
- Friday: domain shift, calibration, uncertainty, interpretability, transfer learning, self-supervision, diffusion models, and integrative astrophysical problems.

The LaTeX source is included so the handout can be modified and recompiled directly from the repository.

### Exercise Set 2 — Week 2

Three self-contained computational notebooks:

1. **Catalog ML and MLP** — classical baselines, rare-class metrics, threshold selection, survey shortcut diagnostics, and a small PyTorch network.
2. **Spectra and time series** — engineered baselines versus 1D CNNs, representation ablations, and scientific interpretation.
3. **Polarimetry and mini-project** — Stokes-space representations, inclination inference, uncertainty propagation, advanced extensions, and the final mini-project menu.

The notebooks generate synthetic astrophysical teaching datasets locally and therefore run without internet access. Their Markdown mirrors are included for readable GitHub previews and static review.

## Suggested environment

Python 3.11+ is recommended. Install the packages listed in `requirements.txt`.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

The computational notebooks have been execution-tested with NumPy, pandas, SciPy, Matplotlib, scikit-learn, PyTorch, and Jupyter.

## Teaching philosophy

The exercises are organized around the full scientific inference chain:

```text
astrophysical question
        ↓
data and provenance
        ↓
representation
        ↓
      model
        ↓
validation and uncertainty
        ↓
astrophysical interpretation
```

The objective is not to reward the most complex architecture, but to build models whose assumptions, validation, limitations, and physical interpretation are scientifically defensible.
