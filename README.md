# Machine Learning Assignment 2: Bagging, Boosting and Bias-Variance on Olivetti Faces

Name: Kalpana Yadav  
Roll number: M24MA2005

Single decision tree, bagging (own implementation), SAMME boosting (own implementation) and a
bias-variance study on the Olivetti faces (40 people, 10 images each, 64x64, raw flattened pixels, no PCA).

## Folder layout

```
common/    split_data.py, samme.py, paths.py   shared code (split, boosting, output locations)
q1/        q1a.py ... q1d.py                   Question 1
q2/        q2a.py ... q2d.py                   Question 2
q3/        q3.py                               Question 3 (a to d in one run)
data/      split.npz                           saved split
configs/   q1_config.json, q1_final_B.json, q2_config.json   frozen settings read by later scripts
models/    *.joblib                            saved ensembles
logs/      *_log.txt                           final logs
results/   *.csv, q3_resamples.npy, q3_raw.npz result tables and raw arrays
figures/   *.png, *.pdf                        figures used in the report
report/    main.tex                            report source
```

## Environment

- Python: `<fill in the output of: python --version>`
- Libraries: see `requirements.txt` (numpy, scikit-learn, matplotlib, joblib)

```
pip install -r requirements.txt
```

The Olivetti data is downloaded automatically by scikit-learn on the first run
(stored in `scikit_learn_data/`, which is not committed).

## How to reproduce

Run from the repository root, in this order. Later scripts read files written by earlier ones.
Outputs go to the folders above automatically, whichever folder you run from.

```
python common/split_data.py   # split.npz (6 train / 2 val / 2 test per person, seed 242005)
python q1/q1a.py              # single tree grid, freezes configs/q1_config.json
python q1/q1b.py              # bagging, saves models/q1_bagging_models.joblib
python q1/q1c.py              # OOB, disagreement, vote margin, final B
python q1/q1d.py              # tree vs bagging, clean and masked
python q2/q2a.py              # one SAMME round on the toy example
python q2/q2b.py              # boosting grid, freezes configs/q2_config.json
python q2/q2c.py              # boosting vs bagging, heavily weighted face
python q2/q2d.py              # clean vs masked for all three models
python q3/q3.py               # bias and variance over 30 refits (long run)
```

## Seeds and settings

- Split: `numpy.random.default_rng(242005)` (numeric characters of M24MA2005).
- Decision trees and boosting weak learners: `random_state=242005`.
- Q1 bagging bootstraps: one generator `default_rng(242005)`, drawn in the order of the B values.
- Q3: one generator `default_rng(242005)`; the 30 resamples are drawn first, then the bagging bootstraps in a fixed order, so `q3/q3.py` must be run as a whole.
- Models are trained on clean images only. The masked test set zeroes rows and columns 24 to 39.

## Notes

- Training times in Q2(c) depend on the machine.
- `q3/q3.py` trains about 700 trees per repetition, so it takes a while.
