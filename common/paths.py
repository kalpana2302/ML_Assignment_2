"""
Folder layout of the project and where each file lives.

P(name) returns the full path of a file, chosen by its type, so scripts can be run from any folder:
  split.npz              -> data/
  .json (configs)        -> configs/
  .joblib (models)       -> models/
  .txt (logs)            -> logs/
  .csv, other .npz/.npy  -> results/
  .png, .pdf (figures)   -> figures/
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def P(name):
    name = str(name)
    ext = Path(name).suffix
    if name == "split.npz":
        folder = "data"
    elif ext == ".json":
        folder = "configs"
    elif ext == ".joblib":
        folder = "models"
    elif ext == ".txt":
        folder = "logs"
    elif ext in (".png", ".pdf"):
        folder = "figures"
    else:
        folder = "results"
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
    return str(ROOT / folder / name)
