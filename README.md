# PAMC25614-RSM-BO

Code and data accompanying the manuscript
"Experimental comparison of response surface methodology and Bayesian optimization for medium improvement in cold-adapted protease production by Antarctic *Pseudomonas* sp." (Han SJ, Choi J-I).

## Contents
| Folder | Content |
|---|---|
| `data/` | Measured design-point means: 30-point CCD (Table S2), 20-point closed-loop BO campaign (Tables 2, S5), 8 neutral external points (Tables S8, S9) |
| `scripts/model_comparison_metrics.py` | RSM and GP (Matérn 5/2 ARD + white noise) predictions; LOO-CV; repeated 5-fold CV on the pooled 58 points; RMSE, MAE, bias, R²/Q², AAD, MPE, TIC (Tables 6, S7, S9, S11) |
| `scripts/simulation_gp_ga_pso.py` | In-silico comparison of GP/EI, GA, PSO, uniform random and single LHS designs on the fitted quadratic surface (Table S12, Fig. S1) |
| `scripts/figure_*.py` | Scripts for Fig. 2, Fig. 5 and Fig. S1 |
| `results/` | Output JSON files |

**TODO before publication:** add the original closed-loop BO campaign code (Latin hypercube generation, GP refitting and EI maximization at each step; random seeds in Table S3) and the original simulation code for Fig. 8.

## Requirements
Python 3.11; numpy, scipy, scikit-learn (1.8), matplotlib.

## Usage
```
python scripts/model_comparison_metrics.py
python scripts/simulation_gp_ga_pso.py results/res_bo.json GP_EI
python scripts/simulation_gp_ga_pso.py results/res_other.json GA,PSO,Random,LHS
```
