# Multi-representation learning for lithography hotspot detection (ICCAD-12)


## Question

Does combining three layout representations (original image, Canny edges, Gaussian density) improve a compact CNN hotspot detector, when models are selected on validation data only and differences are tested across seeds?

## Answer (short)

No significant improvement on any benchmark. Results over six seeds on all five ICCAD-12 benchmarks:

- On ICCAD-3, three-way early fusion is significantly **worse** than the validation-selected single representation (exact Wilcoxon p = 0.031, lower in all six seeds).
- On ICCAD-2, original-only beats original+edge in all six seeds (p = 0.031).
- On ICCAD-1, the best fusion method (edge+density, BA 0.847 ± 0.033) is not significantly better than the original representation (BA 0.825 ± 0.009, p = 0.094).

Earlier single-seed claims (+5.5% on ICCAD-1, p = 0.016) were wrong and are withdrawn. See `docs/CORRECTIONS.md`.

## Repository contents

| Path | What it is |
|---|---|
| `run_experiments_v3_efficient.py` | Experiment script (training, evaluation, summary). Same protocol as v2 for ICCAD-1/2. |
| `scripts/make_figures.py` | Builds every figure in `figures/` from `results/`. |
| `results/summary.md` | Mean ± std and Wilcoxon tests for all five benchmarks. |
| `results/iccad{1,2}_seed{0..5}.json` | Per-seed results for ICCAD-1 and ICCAD-2. |
| `figures/` | Generated charts (bar, histogram, pie). |
| `docs/CORRECTIONS.md` | Earlier claims that were withdrawn, and why. |
| `data/README.md` | Where the dataset goes and how it is laid out. |

Per-seed JSON files for ICCAD-3, 4, and 5 are not included in this repository; only their summary tables are in `results/summary.md`. Add them to `results/` if you have them.

## Method (short)

- **Data:** ICCAD-12 benchmarks ICCAD-1 to ICCAD-5 (not included; see `data/README.md`).
- **Preprocessing:** grayscale, resized to 224×224. Three channels are derived: original (intensity/255), Canny edges (thresholds 50/150), and Gaussian density (binarise at 127, 11×11 kernel, σ = 5).
- **Split:** per seed, the training set is split 80/20 with stratification. The provided test set is used once per model.
- **Model:** two blocks of three 3×3 convolutions (12 filters, ELU, BatchNorm) with max-pooling, then Dense(256) and Dense(2). 2,415,926 parameters for one channel; 7,246,241 for learned-weight fusion.
- **Training:** Adam (lr 1e-3), batch 16, at most 10 epochs, early stopping (patience 4) on validation balanced accuracy.
- **Selection:** threshold chosen on validation (grid 0.05–0.95). SEL = the single representation with the best validation balanced accuracy, chosen per seed.
- **Seeds:** 0, 1, 2, 3, 4, 5. Each seed controls the split and model initialisation.
- **Statistics:** exact two-sided Wilcoxon signed-rank test over the six paired seeds. The smallest achievable p-value with six seeds is 0.031.

## Reproduce

```bash
# 1. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # install torch separately if needed: https://pytorch.org

# 2. set DATA_ROOT inside run_experiments_v3_efficient.py to your ICCAD-12 folder

# 3. check that the preprocessing matches the original v2 code
python run_experiments_v3_efficient.py --verify --benches iccad1

# 4. run the benchmarks (finished seeds are skipped)
python run_experiments_v3_efficient.py --benches iccad3 iccad4 iccad5

# 5. figures
python scripts/make_figures.py
```

Outputs go to the folder set by `OUT` in the script (`results_v2_cached/` by default). `summary.md` is written there.

Storage: the grayscale cache uses 50 KB per image (about 8 GB for all five benchmarks). Runtime depends on hardware and was not benchmarked.

## Limitations

- ICCAD-1 validation saturates (20 hotspots in validation; balanced accuracy reaches 1.0 for most seeds), so thresholds are weakly determined.
- Several ICCAD-1 operating points have recall near 1.0 and precision about 0.12–0.15, which means most flagged clips are false alarms.
- Six seeds give p ≥ 0.031. Non-significant results do not show equal performance.
- ICCAD-4 and ICCAD-5 have very few training hotspots (95 and 26).
- Only balanced accuracy is compared with published results. Published recall, false-alarm, and F1 values have not been collected.
- Hyperparameters and Canny/Gaussian settings were not tuned.
- Images are downsampled from 1200 to 224 pixels.
- Learned-weight fusion uses one global weight per representation, not per-sample attention.


