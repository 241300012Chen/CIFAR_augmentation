# CIFAR_augmentation

CIFAR-10 augmentation experiments (training scripts, plots, and appendix tables).

This repo contains:
- Training/evaluation: run_experiment.py
- Plotting: plot_all.py
- Augmentation visualization: visualize_aug.py
- Appendix table generation: make_appendix_tables.py
- LaTeX per-seed tables: appendix_per_seed_table_10.tex, appendix_per_seed_table_20.tex

Note: data/, checkpoints/, runs/, results/ are excluded from Git tracking by .gitignore (large or generated artifacts).

## Quick Start

### 1) Create env + install deps

Windows (PowerShell)
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- pip install -U pip
- pip install -r requirements.txt

Windows (CMD)
- python -m venv .venv
- .\.venv\Scripts\activate.bat
- pip install -U pip
- pip install -r requirements.txt

macOS / Linux
- python -m venv .venv
- source .venv/bin/activate
- pip install -U pip
- pip install -r requirements.txt

If requirements.txt does not exist yet, generate it after installing packages:
- pip freeze > requirements.txt

## Data

This project uses CIFAR-10. Typical options:
- Use torchvision.datasets.CIFAR10 to auto-download inside your script, or
- Place dataset files under data/ (ignored by git).

## Run Experiments

Check available arguments:
- python run_experiment.py --help

Example patterns (adjust to your actual CLI arguments):
- python run_experiment.py --method baseline --seed 0 --subset 0.1
- python run_experiment.py --method composite --seed 1 --subset 0.2

Suggested outputs:
- TensorBoard logs: runs/
- Model checkpoints: checkpoints/
- Metrics / CSV logs: results/

## Plot Results

After you have produced logs/CSVs under results/, run:
- python plot_all.py

Expected outputs:
- Figures saved to figures/ (optional to track in git)

## Visualize Augmentations
- python visualize_aug.py

## Generate Appendix Tables (LaTeX)

Regenerate appendix tables from results/:
- python make_appendix_tables.py

This should regenerate:
- appendix_per_seed_table_10.tex
- appendix_per_seed_table_20.tex

## Repository Structure

CIFAR_augmentation/
  .gitignore
  README.md
  run_experiment.py
  plot_all.py
  visualize_aug.py
  make_appendix_tables.py
  appendix_per_seed_table_10.tex
  appendix_per_seed_table_20.tex
  data/           (ignored)
  checkpoints/    (ignored)
  runs/           (ignored)
  results/        (ignored)
  figures/        (optional)

## Notes

- To keep the repo lightweight and reproducible, avoid committing datasets, checkpoints, and large logs.
- If you want to publish final figures, remove figures/ from .gitignore.
