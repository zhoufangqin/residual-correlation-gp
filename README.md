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

## Setup

```bash
pip install -r requirements.txt
```

The tabular experiments need only NumPy, SciPy, scikit-learn, pandas and matplotlib.
The image experiments additionally need PyTorch, timm, Pillow and pycocotools.
The results in this repository were produced with Python 3.10.

Run every script from the repository root; each one writes its outputs under
`results/<experiment>/`.

## Data

| Datasets | Location | Source |
|---|---|---|
| atp1d, atp7d, edm, jura, rf1, rf2, scm1d, scm20d, wq, sarcos | `data_public/*.npz` (included) | Multi-target regression benchmarks: [MULAN](https://mulan.sourceforge.net/datasets-mtr.html) (Spyromitros-Xioufis et al., 2016); sarcos from the [GPML book site](http://gaussianprocess.org/gpml/data/) |
| energy | `data_public/energy.npz` (included) | [UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| tecator | `data_public/tecator.npz` (included) | [StatLib Tecator](http://lib.stat.cmu.edu/datasets/tecator) (Borggaard and Thodberg, 1992) |
| SH, WUR, rice, wheat (PHO) | not included | Hyperspectral leaf spectra with gas-exchange V<sub>cmax</sub>/J<sub>max</sub>; available from the authors on request |
| MPII, AFLW2000-3D, COCO keypoints | not included | Download the images yourself into `imgdata/{mpii,aflw,coco}/`, then run the `embed_*.py` scripts |

Each `.npz` holds `X` (n × p features) and `Y` (n × T targets); some also hold
`target_names`. Experiments subsample to n = 280 per seed, as described in the paper.

Scripts skip any dataset whose data are unavailable, so without the plant data they run
on the 12 public benchmarks. To include the plant datasets, set `CALIB_PAPER_DIR` to the
directory that provides the companion `data_utils` loaders, or place
`sh.npz`, `wur.npz`, `rice.npz` and `wheat.npz` in `data_public/`.

## Core modules

| File | Contents |
|---|---|
| `legJ_jointcalib.py` | Independent GPs, rank-1 GP-ICM (Kronecker eigendecomposition), joint metrics, data loading |
| `exp_residual_diagnostic.py` | `D_logdet` from standardised out-of-fold residuals |
| `exp_residual_icm.py` | Residual-ICM predictive covariance |
| `exp_hardening_ablation.py` | Rank-r ICM fitting and the npz-first data loader |
| `gateB_recovery.py`, `e0_synthetic.py` | Linear/MLP probes and LogME, used by the selection and recovery experiments |
| `embed_coco.py` | Frozen-encoder embedding code shared by the image scripts |

## Reproducing the paper

| Paper item | Script(s) | Output |
|---|---|---|
| Dataset table and `D_logdet` association (§4) | `exp_residual_diagnostic_multiseed.py` | `results/legJ/residual_diagnostic_multiseed_*.csv` |
| Partial correlations (`tab:partial`), robustness (`tab:robustness`), gate-ceiling sweep (`tab:ceiling`) | `exp_dlogdet_standardised.py`, then `exp_rz_consistency_sweep.py` (prints every statistic in these tables) | `results/legJ/dlogdet_standardised*.csv`, `results/legJ/rz_sweep_inputs.csv` |
| Synthetic phase diagram with real-data overlay (`fig:phase`) | `phase_prediction.py`, `plot_phase_overlay.py` | `results/phase/` |
| Sequential T′-growth experiment (`fig:sequential`) | `exp_sequential.py` | `results/sequential/` |
| Method comparison: Indep, ICM rank 1 and rank T−1, Residual-ICM, gate (`tab:method`) | `exp_residual_icm_multiseed.py`, `exp_hardening_multiseed.py`, `exp_resicm_matched_table3.py`, then `rebuild_table4.py` | `results/legJ/table4_matched.{csv,tex}` |
| Representation selection (`tab:selection`, `tab:selection_full`) | `legS_selection.py` | `results/legS/` |
| Frozen image encoders: MPII, AFLW2000-3D, COCO (§6.2, `fig:mpii_dlogdet`) | `prefetch_encoders.py`, `embed_{mpii,aflw,coco}.py`, `exp_{mpii,aflw,coco}_instrument.py`, `plot_mpii_scatter.py` | `results/{mpii,aflw,coco}/` |
| App. B: ICM configuration ablation (`tab:hardening`), rank sweep vs Residual-ICM (`tab:rank_resicm`) | `exp_hardening_multiseed.py`, `analysis_rank_vs_resicm.py`, `exp_resicm_zresid.py` | `results/legJ/hardening_multiseed.csv`, `rank_vs_resicm_5seed.csv` |
| App. C.1: calibration of the single-output GP (`tab:calibration`) | `legC_calibration_std.py` (uses `legC_calibration.py`) | `results/legC/calibration_std.csv` |
| App. C.2: cross-target recovery vs probes (`tab:gateb`) | `gateB_recovery.py` | `results/gateB/` |
| App. C.3: residual covariance in neural predictors (`tab:neural`) | `exp_deepens_coupling.py`, `exp_deepens_dlogdet.py` | `results/deepens/` |
| App. D: phase diagrams across ρ (`fig:phase_rho`) | `phase_rho_sweep.py` (`--plot-only` redraws from the CSV) | `results/phase/phase_rho_sweep.*` |
| App. D: scope stress tests (`tab:scope_tests`, `tab:nonstationary`, `tab:nonlinear`) | `exp_diagnostic_scope.py`, `exp_scope_partb_fix.py` | `results/scope/` |
| App. E: shrinkage (`fig:shrinkage_synth`, `tab:shrinkage`) | `synthetic_collinearity_shrinkage.py`, `exp_lw_multiseed.py` | `results/shrinkage/`, `results/legJ/lw_multiseed.csv` |

Notes on specific result files:

- `results/legJ/realdata_full.csv` (the overlay points in `fig:phase`) collates the
  per-dataset five-seed means from `residual_icm_multiseed_summary.csv`.
- `results/shrinkage/tabular_dlogdet_lw.csv` (tabular row of `tab:shrinkage`) is the
  seed-0 output of `exp_residual_diagnostic.py`.
- `legS_selection.py` produces the GP-LML, LogME and capacity columns of the selection
  tables. The GP-ICM selection column is not produced by a script in this repository.
- SH and WUR rows in `tab:method` keep their published values. Their GP fits are
  sensitive to BLAS threading because p ≫ n; see `rebuild_table4.py`.

Several experiments (the T = 16 rank sweeps and the image embeddings) take hours and
were run on a GPU cluster.

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

Code: MIT (see `LICENSE`). The datasets in `data_public/` remain under the terms of their original sources listed above.
