# Residual Correlation as a Diagnostic for Joint-Uncertainty Gains from GP Coregionalisation

Code, data and result files for the ACML 2026 paper

> Fangqin Zhou and Joaquin Vanschoren. *Residual Correlation as a Diagnostic for
> Joint-Uncertainty Gains from GP Coregionalisation.* Proceedings of the 18th Asian
> Conference on Machine Learning (ACML 2026), PMLR 353.

## Summary

Correlated targets are often modelled jointly with a coregionalised Gaussian process
(GP-ICM). Across synthetic data, 16 multi-target benchmarks and frozen image
representations, we find that coupling mainly improves **joint predictive uncertainty**,
not point prediction or representation selection, and that the gain is predicted by the
**residual correlation** left by independent per-target GPs rather than by raw target
correlation. We summarise it with

```
D_logdet = -1/2 * log det(R_res)
```

where `R_res` is the correlation matrix of the independent GPs' standardised
out-of-fold residuals. Across the 16 benchmarks, `D_logdet` is strongly associated with
the observed ICM joint-NLL gain (Spearman ρ = −0.83), whereas raw target correlation
(−0.07) and the feature-to-sample ratio p/n (−0.46) are not. **Residual-ICM** keeps the
independent GP means and marginal variances and adds only the estimated residual
correlation to the joint predictive covariance; it attains the best mean joint NLL among
the compared methods.

## Main result

Joint-NLL change relative to independent GPs (ΔNLL; lower is better), five-seed means
(Table 4 of the paper). Residual-ICM attains the best mean ΔNLL and mean rank, and does
not fail in the high-accuracy "ceiling" regime (sarcos, tecator) where jointly fitted ICM
worsens joint NLL.

| Dataset | Indep | ICM (r=1) | ICM (r=T−1) | Residual-ICM | Gate |
|---|---:|---:|---:|---:|---:|
| scm20d (strong) | 0.00 | −3.59 | −5.95 | −5.41 | −3.59 |
| atp1d (strong) | 0.00 | −1.14 | −1.56 | −1.93 | −1.14 |
| rf1 (R²<sub>Indep</sub> = 0.93) | 0.00 | −1.09 | −1.21 | −0.86 | 0.00 |
| sarcos (ceiling) | 0.00 | +0.44 | −0.06 | −0.83 | 0.00 |
| tecator (ceiling) | 0.00 | +0.78 | +0.62 | −0.45 | 0.00 |
| wq (low structure) | 0.00 | −0.14 | −0.06 | −0.10 | −0.14 |
| **Mean ΔNLL (16 datasets)** | 0.00 | −0.61 | −0.93 | **−1.13** | −0.61 |
| **Mean rank (1 = best)** | 4.34 | 2.97 | 2.56 | **1.94** | 3.19 |

The gate fits rank-1 ICM only when `D_logdet` exceeds a threshold chosen by
leave-one-dataset-out cross-validation and R²<sub>Indep</sub> < 0.9. Values are in
`results/legJ/table4_matched.csv`.

## Setup

```bash
pip install -r requirements.txt
```

The tabular experiments need only NumPy, SciPy, scikit-learn, pandas and matplotlib.
The image experiments additionally need PyTorch, timm, Pillow and pycocotools.
The results in this repository were produced with Python 3.10.

All Python code is in `src/`. Run scripts from the repository root, e.g.

```bash
python src/exp_residual_diagnostic_multiseed.py
```

Each script writes its outputs under `results/<experiment>/`.

## Repository layout

```
src/            all Python code (core modules, experiments, plotting)
data_public/    public multi-target benchmarks as .npz (X, Y)
results/        result files behind the paper's tables and figures
REPRODUCE.md    data sources and which script produces each table and figure
```

## Reproducing the paper

[REPRODUCE.md](REPRODUCE.md) describes the datasets (the 12 public benchmarks are included
in `data_public/`), lists the core modules, and maps every table and figure in the paper
to the script that produces it and the result file it writes.

## Citation

```bibtex
@inproceedings{zhou2026residual,
  title     = {Residual Correlation as a Diagnostic for Joint-Uncertainty Gains from {GP} Coregionalisation},
  author    = {Zhou, Fangqin and Vanschoren, Joaquin},
  booktitle = {Proceedings of the 18th Asian Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {353},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`). The datasets in `data_public/` remain under the terms of their original sources listed in [REPRODUCE.md](REPRODUCE.md#data).
