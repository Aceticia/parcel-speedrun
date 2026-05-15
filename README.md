# parcel-speedrun

A tiny autoresearch playground: pretrain a transformer on Schaefer-1000 parcel BOLD time series via masked time-step reconstruction, then probe the learned representation for two subject-level labels (`Gender`, `Child_Adult`).

The point of the repo is the **loop** — drop Claude Code in, point at `CLAUDE.md`, walk away. Each experiment is time-budgeted to ~10 min and produces a parseable summary block.

## Quickstart

```sh
uv sync
uv run python main.py
```

First run will warm `~/nilearn_data/` and `nilearn_cache/` (a few minutes of downloads + parcellation). Subsequent runs are fast.

## Layout

- `simple_model.py` — the encoder: `SimpleBOLDEncoder` with a learned mask token, EvaAttention temporal blocks, 1D RoPE over time, a width-direction FFW that updates everything except the input slice.
- `main.py` — data fetch + subject-level deterministic split + Lightning training (`LitBOLD`, masked recon) + LDA probe + parseable summary.
- `CLAUDE.md` — the autoresearch protocol: what's editable, what's frozen, how runs are scored, and the experiment loop.

## Data

[nilearn `development_fmri`](https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_development_fmri.html) — 155 subjects (~25 adults, ~130 children aged 3–5) viewing a Pixar short. Parcellated to Schaefer-2018 1000 ROIs with z-scored time series.

## Baseline (one run, MPS, ~10 min)

| metric                  | value             |
| ----------------------- | ----------------- |
| `probe_ca_test_bal`     | ~0.94             |
| `probe_gender_test_bal` | ~0.57 (≈ chance)  |
| `num_params_M`          | 11.77             |

Numbers are noisy — the test set is ~30 subjects. The point of the loop is to push `probe_gender_test_bal` up from this baseline **without degrading `probe_ca_test_bal`**.

## Autoresearch

Start a new branch and let the loop run:

```sh
git checkout -b autoresearch/<tag>
# then in Claude Code: "begin the autoresearch loop per CLAUDE.md"
```

See `CLAUDE.md` for the full protocol.
