# Mini-Project — Physics-Informed Neural Networks for Stellar Structure

**Course:** Machine Learning and Artificial Intelligence for Modern Astrophysics  
**Instructor:** Gabriel Wendell Celestino Rocha    
**Project type:** Physics-Informed Neural Networks (PINNs) / Scientific Machine Learning  
**Suggested difficulty:** Intermediate–Advanced  

> **Central scientific question:**  
> Can a neural network recover the structure of a self-gravitating polytropic star by being trained
> primarily on the governing differential equation rather than on a large labeled dataset?

---

## 1. Scientific context

Physics-Informed Neural Networks (PINNs) provide a bridge between deep learning and mathematical
physics. Instead of learning only from input–output examples, a PINN is trained so that its prediction
approximately satisfies a differential equation, boundary/initial conditions, and, when available,
observational data.

The **Lane–Emden equation** is a particularly suitable Astrophysics mini-project because it:

- arises directly from hydrostatic stellar structure under a polytropic equation of state;
- is nonlinear for general polytropic index $n$;
- has simple physical boundary conditions;
- has a trusted conventional numerical solution for comparison;
- has an analytic $n=1$ case that can be used as a sanity check;
- contains a coordinate singularity at the stellar center, forcing careful physical implementation.

The goal is **not to prove that PINNs are superior to ODE solvers**. For this one-dimensional problem,
a classical solver will usually be faster and more accurate. The goal is to understand how physics
enters a neural-network loss, how automatic differentiation is used, and how PINN solutions must be
validated scientifically.

---

## 2. Physical background

Assume a polytropic equation of state

$$
P = K\rho^{1+1/n}~,
$$

where $P$ is pressure, $\rho$ is density, $K$ is the polytropic constant, and $n$ is the
polytropic index.

Combining this relation with spherical hydrostatic equilibrium and mass continuity gives, after
nondimensionalization,

$$
\frac{1}{\xi^2}\frac{d}{d\xi}
\left(
\xi^2\frac{d\theta}{d\xi}
\right)
+\theta^n=0~,
\qquad
\text{or}
\qquad
\theta''(\xi)+\frac{2}{\xi}\theta'(\xi)+\theta(\xi)^n=0~.
$$

The central boundary conditions are

$$
\theta(0)=1~,
\qquad
\theta'(0)=0~.
$$

The dimensionless density profile is

$$
\frac{\rho(\xi)}{\rho_c}=\theta(\xi)^n~.
$$

For finite-radius polytropes, the first positive zero

$$
\theta(\xi_1)=0~,
$$

defines the dimensionless stellar surface.

For $n=1$,

$$
\theta(\xi)=\frac{\sin\xi}{\xi}~,
\qquad
\xi_1=\pi~,
$$

with the regular limit $\theta(0)=1$.

### Core benchmark cases

- **$n=1$:** analytic validation case;
- **$n=3$:** nonlinear stellar-structure case.

---

## 3. Learning objectives

By the end of the project, you should be able to:

1. distinguish ordinary supervised learning from physics-informed learning;
2. translate a differential equation into a neural-network optimization objective;
3. use automatic differentiation to compute first and second derivatives;
4. impose boundary conditions softly or exactly by network construction;
5. handle the central coordinate singularity carefully;
6. construct a classical numerical reference solution;
7. train and diagnose a PINN;
8. compare solution error and differential-equation residual;
9. determine whether low PINN loss implies an accurate physical solution;
10. explain when a PINN is scientifically useful and when a classical solver is preferable.

---

## 4. Required workflow

### Stage A — Recover the classical stellar-structure problem

Explain briefly:

1. how hydrostatic equilibrium, mass continuity, and a polytropic equation of state lead to the
   Lane–Emden problem;
2. the physical meanings of $\theta$, $\xi$, $n$, and $\xi_1$;
3. why regularity at the center requires
   $$
   \theta(0)=1~,\qquad \theta'(0)=0~;
   $$
4. why the expanded equation appears singular at $\xi=0$.

**Deliverable:** concise physical derivation/explanation in the notebook.

---

### Stage B — Build a trusted numerical reference

Use a standard numerical method such as `scipy.integrate.solve_ivp`.

Do not start the expanded equation naively at $\xi=0$. A convenient strategy is to begin at
$\epsilon>0$ using

$$
\theta(\xi)=
1-\frac{\xi^2}{6}
+\frac{n\xi^4}{120}
+\mathcal{O}(\xi^6)~,
$$

$$
\theta'(\xi)=
-\frac{\xi}{3}
+\frac{n\xi^3}{30}
+\mathcal{O}(\xi^5)~.
$$

Required reference cases:

- $n=1$;
- $n=3$.

For $n=1$, validate the numerical solution against

$$
\theta_1(\xi)=\frac{\sin\xi}{\xi}~.
$$

**Deliverables:**

- reference profiles;
- numerical error for $n=1$;
- estimate of the first zero $\xi_1$;
- explanation of why this is the scientific reference solution.

---

### Stage C — Design the PINN

Let an MLP $N_\phi(\xi)$ approximate a latent correction.

A recommended hard-constrained trial solution is

$$
\widehat{\theta}_\phi(\xi)=
1+\xi^2 N_\phi(\xi).
$$

This automatically guarantees

$$
\widehat{\theta}_\phi(0)=1~,
\qquad
\widehat{\theta}'_\phi(0)=0~.
$$

Use automatic differentiation to obtain

$$
\widehat{\theta}'_\phi(\xi)~,
\qquad
\widehat{\theta}''_\phi(\xi)~.
$$

---

### Stage D — Define the physics-informed loss

The direct residual is

$$
r_\phi(\xi)=
\widehat{\theta}''_\phi
+
\frac{2}{\xi}\widehat{\theta}'_\phi
+
\widehat{\theta}_\phi^n~.
$$

Near the origin, use a nonsingular equivalent residual such as

$$
\widetilde r_\phi(\xi)=
\xi\,\widehat{\theta}''_\phi
+
2\widehat{\theta}'_\phi
+
\xi\,\widehat{\theta}_\phi^n~.
$$

For the hard-constrained model,

$$
\mathcal{L}_{\mathrm{phys}}=
\frac{1}{N_c}
\sum_{j=1}^{N_c}
\left|
\widetilde r_\phi(\xi_j)
\right|^2~,
$$

where the $\xi_j$ are collocation points.

If you test soft boundary conditions, use a loss of the form

$$
\mathcal{L}=
\lambda_r\mathcal{L}_{\mathrm{phys}}
+
\lambda_0|\widehat{\theta}(0)-1|^2
+
\lambda_1|\widehat{\theta}'(0)|^2~.
$$

**Required discussion:** hard versus soft physical constraints.

---

### Stage E — Train the PINN

Train at least:

- one PINN for $n=1$;
- one PINN for $n=3$.

Recommended starting point:

- activation: `tanh`;
- 3–5 hidden layers;
- 32–64 neurons per layer;
- optimizer: Adam;
- several hundred collocation points.

Optionally refine with L-BFGS after Adam.

Record:

- architecture;
- optimizer and learning rate;
- collocation strategy;
- number of optimization steps;
- random seed;
- loss history.

Do not use the dense reference solution as training labels in the core forward-PINN experiment.

---

### Stage F — Validate in solution space and equation space

Compare

$$
\theta_{\rm PINN}(\xi)
\quad\text{with}\quad
\theta_{\rm reference}(\xi)~.
$$

Compute at least:

$$
\mathrm{MAE}=
\frac{1}{N}\sum_i
|\theta_{\mathrm{PINN},i}-\theta_{\mathrm{ref},i}|~,
$$

$$
\mathrm{RMSE}=
\sqrt{
\frac{1}{N}
\sum_i
(\theta_{\mathrm{PINN},i}-\theta_{\mathrm{ref},i})^2
}~.
$$

Also report:

- maximum absolute error;
- residual as a function of radius;
- error near the center;
- error near the first zero;
- PINN estimate $\widehat{\xi}_1$;
- error in $\widehat{\xi}_1$.

For $n=3$, compare the physical density profile

$$
\rho/\rho_c=\theta^3~,
$$

inside the stellar surface.

---

### Stage G — Perform at least two controlled ablations

Choose at least two:

1. hard vs. soft boundary conditions;
2. number of collocation points;
3. uniform vs. nonuniform collocation sampling;
4. network depth/width;
5. activation function;
6. Adam only vs. Adam + L-BFGS;
7. direct vs. nonsingular residual;
8. training-domain size;
9. multiple random seeds.

State a hypothesis **before** running each ablation.

---

### Stage H — Scientific interpretation

Your conclusion must answer:

1. Does the PINN reproduce the $n=1$ analytic solution?
2. How accurately does it reproduce the $n=3$ numerical solution?
3. Where is the PINN least accurate?
4. Does small physics loss guarantee small solution error?
5. How sensitive is training to collocation and optimization choices?
6. What advantages does the PINN provide?
7. What disadvantages does it have relative to `solve_ivp`?
8. For what more difficult astrophysical problem might a PINN become more attractive?

---

## 5. Minimum modeling requirements

A complete submission must contain:

- a conventional Lane–Emden solver;
- $n=1$ analytic validation;
- $n=3$ numerical reference;
- a PyTorch PINN using automatic differentiation;
- explicit physics residual;
- explicit treatment of the central boundary conditions;
- quantitative PINN/reference comparison;
- at least two ablations;
- astrophysical interpretation;
- reproducibility information.

---

## 6. Required figures

Prepare at least:

1. reference $\theta(\xi)$ curves for $n=1$ and $n=3$;
2. PINN vs. reference for $n=1$;
3. PINN vs. reference for $n=3$;
4. absolute error vs. radius;
5. physics residual vs. radius;
6. training-loss history;
7. $n=3$ density profile $\rho/\rho_c=\theta^3$;
8. ablation comparison.

---

## 7. Required summary table

| Model | $n$ | Constraint | $N_c$ | MAE | RMSE | Max error | $\widehat{\xi}_1$ | Radius error |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Analytic/reference | 1 | — | — | — | — | — | $\pi$ | — |
| PINN-A | 1 | hard | ... | ... | ... | ... | ... | ... |
| Numerical reference | 3 | — | — | — | — | — | ... | — |
| PINN-B | 3 | hard | ... | ... | ... | ... | ... | ... |

---

## 8. Optional advanced extensions

### Extension A — Inverse PINN: infer $n$

Treat the polytropic index as trainable and provide only sparse structural observations.

Optimize

$$
\mathcal{L}=
\lambda_{\rm phys}\mathcal{L}_{\rm phys}
+
\lambda_{\rm data}\mathcal{L}_{\rm data}~.
$$

Ask whether the correct $n$ can be recovered from sparse data.

### Extension B — Parameterized PINN

Train one network

$$
\widehat{\theta}(\xi,n)~,
$$

and test interpolation to an unseen polytropic index.

### Extension C — Sparse-data + physics comparison

Compare:

1. a purely supervised neural network trained on sparse samples;
2. a PINN trained on the same samples plus the differential-equation residual.

Ask whether physical constraints improve reconstruction in the small-data regime.

### Extension D — Generalized stellar-structure equation

Replace the standard Lane–Emden equation with a related stellar-structure problem and evaluate how
well the same PINN design transfers.

---

## 9. Reproducibility checklist

- [ ] Fix random seeds.
- [ ] Record architecture and all hyperparameters.
- [ ] Record the collocation sampling rule.
- [ ] Keep the reference solver separate from core PINN training.
- [ ] Save loss histories.
- [ ] Generate final figures from code.
- [ ] Record failed or unstable runs.
- [ ] Record package versions.
- [ ] Clearly distinguish analytic truth, numerical reference, and PINN prediction.

---

## 10. Suggested reading

- Raissi, Perdikaris & Karniadakis (2019), *Physics-informed neural networks: A deep learning framework
  for solving forward and inverse problems involving nonlinear partial differential equations*,
  Journal of Computational Physics, 378, 686–707.
- Baty (2023), *Modelling Lane–Emden type equations using Physics-Informed Neural Networks*,
  Astronomy and Computing, 44, 100734.

---

# Starter workspace

The cells below are deliberately incomplete. They provide structure but do not solve the project.