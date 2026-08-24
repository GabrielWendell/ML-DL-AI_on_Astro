# Week 2 Friday — Research-style laboratory

This folder contains the final hands-on class for the mini-course
**Machine Learning and Artificial Intelligence for Modern Astrophysics**.

## Slot 1 — Polarimetry and representation geometry

The class studies multi-band linear polarimetry and the 180-degree topology of polarization position
angle. It compares:

1. V-band physical baseline;
2. naive `(p, chi)` representation with scalar-angle regression;
3. raw Stokes `(q, u)` with circular target encoding;
4. `(p, cos 2chi, sin 2chi)` with circular target encoding.

Diagnostics include global circular error, wrap-region error, and low-S/N performance. Students then
choose one controlled intervention.

## Slot 2 — Mini-project research studio

Groups choose a mini-project direction, complete a research canvas, define a minimum viable experiment,
give a 2–3 minute pitch, receive peer critique, and revise one design decision.

The intended progression is from guided coding to independent scientific decision-making.

## Files

- `friday_research_lab_instructor.ipynb`
- `friday_research_lab_student.ipynb`
- `friday_research_lab.md`
- `requirements.txt`

The polarimetry data are synthetic and generated locally, so the notebook runs offline.
