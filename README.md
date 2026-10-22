# Machine Learning Assignment 2: Bagging, Boosting and Bias-Variance on Olivetti Faces

Name: Kalpana Yadav  
Roll number: M24MA2005

Single decision tree, bagging (own implementation), SAMME boosting (own implementation) and a
bias-variance study on the Olivetti faces (40 people, 10 images each, 64x64, raw flattened pixels, no PCA).

## Environment

- Python: `<fill in the output of: python --version>`
- Libraries: see `requirements.txt` (numpy, scikit-learn, matplotlib, joblib)

```
pip install -r requirements.txt
```

The Olivetti data is downloaded automatically by scikit-learn the first time a script runs
(it is stored in `scikit_learn_data/`, which is not committed).

## How to reproduce

Run all scripts from the repository folder, in this order. Later scripts read files written by earlier ones.

```
python split_data.py     # creates split.npz (6 train / 2 val / 2 test per person, seed 242005)
python q1a.py            # single tree grid, freezes q1_config.json
python q1b.py            # bagging, saves q1_bagging_models.joblib
python q1c.py            # OOB accuracy, disagreement, vote margin, picks final B (q1_final_B.json)
python q1d.py            # tree vs bagging on clean and masked test sets
python q2a.py            # one SAMME round on the toy example
python q2b.py            # boosting grid, freezes q2_config.json and q2_selected_model.joblib
python q2c.py            # boosting vs bagging, heavily weighted face
python q2d.py            # clean vs masked for all three models
python q3.py             # bias and variance over 30 refits (takes a long time)
```

## Files

| File | Purpose |
|---|---|
| `split_data.py` | Makes the split once and provides `load_split()` (clean train/val/test and the masked test set) |
| `samme.py` | SAMME boosting implementation, used for the toy example and for the faces |
| `q1a.py` ... `q1d.py` | Question 1 (a) to (d) |
| `q2a.py` ... `q2d.py` | Question 2 (a) to (d) |
| `q3.py` | Question 3 (a) to (d) in one run |
| `split.npz` | Saved split (indices of train, validation and test images) |
| `q1_config.json`, `q1_final_B.json`, `q2_config.json` | Frozen settings used by later scripts |
| `*_log.txt`, `*.csv` | Final logs and result tables |
| `*.png`, `*.pdf` | Figures used in the report |
| `q3_resamples.npy` | The 30 resampled training sets used in Q3 |
| `main.tex` | Report source |

## Seeds and settings

- Split: `numpy.random.default_rng(242005)` (numeric characters of the roll number M24MA2005).
- Decision trees and boosting weak learners: `random_state=242005`.
- Bagging bootstrap samples (Q1): one generator `default_rng(242005)`, drawn in the order of the B values.
- Q3: one generator `default_rng(242005)`; the 30 resamples are drawn first, then the bagging bootstraps in a fixed order, so `q3.py` must be run as a whole.
- Models are trained on clean images only. The masked test set sets rows and columns 24 to 39 of the test images to zero.

## Notes

- Training times in Q2(c) depend on the machine.
- `q3.py` trains about 700 trees per repetition, so it takes a while.
- `.joblib` files are produced by `q1b.py` and `q2b.py`.