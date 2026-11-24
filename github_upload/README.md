# Smart-meter security simulation

Simulation code for the project report *Security of Smart Metering in Smart Grids*. It reproduces every result and figure in Chapters 7–10: synthetic AMI load data, attack injection (scenarios S1–S4), explainable detectors and their evaluation.

## Requirements
Python 3.10+ with `numpy`, `pandas`, `matplotlib` (no other libraries):

```bash
pip install -r requirements.txt
```

## Run
```bash
python run_experiments.py all    # main (120 runs), sensitivity (200 runs), threshold study (~7 min on 2 cores)
python make_figures.py all       # Figures 12-29 -> fig/
python make_equations.py         # Equations (8)-(16) as images -> eq/
python quick_test.py             # fast smoke test
```
Seeds are fixed, so results are identical on every run.

## Layout
| File | Purpose |
|---|---|
| `config.py` | All parameters (Tables 11, 17, 18, 23) |
| `generator.py` | Synthetic load model, Eq. (1), plus benign events (A11) |
| `attacks.py` | Scenarios S1–S4 with ground-truth labels and attack events |
| `detectors.py` | D1 rolling z-score, D2 rules, D3 neighbourhood, E1 CUSUM |
| `metrics.py` | Detection rate, FPR, delay, false-positive sources |
| `run_experiments.py` | Experiments of Table 20 → `results/*.csv` |
| `make_figures.py` | All figures |
| `make_equations.py` | Equation images |
| `results/` | Raw and summary CSV results used in Chapter 9 |
| `fig/` | Figures 12–29 (PNG, 220 dpi) |
| `eq/` | Equations (8)–(16) (PNG, 300 dpi) |

Figure file names follow the draft order; in the report `fig29_cost_benefit.png` is Figure 28 and `fig28_roadmap.png` is Figure 29.
