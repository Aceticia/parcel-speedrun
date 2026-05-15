# parcel-speedrun

This is an experiment to have the LLM do its own research on a tiny BOLD pretraining + downstream probe task.

The **baseline pipeline** is (any of these steps can be changed — see Experimentation):

1. Pretrain `SimpleBOLDEncoder` via masked time-step reconstruction on Schaefer-1000 parcel time series from the nilearn `development_fmri` dataset.
2. Average the hidden-dim slice of the encoder output across time per window.
3. Fit LDA on the train subjects, majority-vote windows to a per-subject prediction, score on test subjects via balanced accuracy for two labels: `Gender` (M/F) and `Child_Adult` (child/adult).

The split is by subject and deterministic. There is **no validation split** — train and test only, to maximize data on both sides.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repo context (if present).
   - `main.py` — most of this is editable. The off-limits parts are the data-and-split block (`N_PARCELS`, `N_SUBJECTS`, `SPLIT`, `SPLIT_SEED`, `fetch_and_parcellate`, `split_subjects`) and the `---` summary `print` lines at the bottom of `main()`. Everything else — model, windowing, feature extraction, probe — is yours.
   - `simple_model.py` — the encoder. Editable.
4. **Verify data exists**: First run will populate `~/nilearn_data/` and `nilearn_cache/` lazily. If you see fetches taking minutes, that's the first-time cache warm-up — subsequent runs are fast.
5. **Initialize `results.tsv`**: `results.tsv` persists across branches as the notebook of every idea tried (see "Logging results" below). If it already exists, **leave its contents intact** and just append a blank separator line so the new branch's rows are visually distinct from prior runs. Only if it doesn't exist, create it with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains on whatever accelerator is available (`accelerator="auto"`) with a **fixed time budget**: 9 min of training (`MAX_TRAIN_MINUTES`) plus ~1 min of feature extraction and probing, for a **total wall-clock budget of ~10 min**. You launch it simply as: `uv run python main.py`.

**What you CAN modify:**
- `simple_model.py` — encoder internals, anything inside.
- In `main.py`, almost everything below the data layer:
  - The `LitBOLD` class (model wiring, loss, optimizer, training_step, mask logic, etc.). **The training loss must be self-supervised** — see Pretraining constraint below.
  - The `MASK_RATIO` constant, the `WINDOW` constant.
  - The `BOLDWindows` class — change the windowing strategy, use overlapping windows, use variable-length windows, or skip windowing entirely and feed the full subject time series.
  - `extract_hidden_features` — pool however you like (mean over time, last token, attention pool, a fresh forward without windowing, etc.).
  - `fit_and_eval` — change the classifier (logistic regression, SVC, kNN, …), change how window-level predictions are aggregated to subject-level (majority vote, mean of probabilities, predict from a pooled feature directly, …). The only requirement is the **Evaluation contract** below.
  - The `_sub_bal` helper — it just plucks a value out of whatever `fit_and_eval` returns; if you change the return shape, update this in lockstep.
  - The `DataLoader` config inside `make_loader` (batch size, shuffling, num_workers).
  - The `L.Trainer(...)` kwargs *except* `max_time` (which enforces the budget).

**What you CANNOT modify:**
- `MAX_TRAIN_MINUTES` or the `max_time={"minutes": MAX_TRAIN_MINUTES}` argument to `Trainer`. This enforces the time budget — leave it alone.
- The **data and split**: `N_PARCELS`, `N_SUBJECTS`, `SPLIT`, `SPLIT_SEED`, `fetch_and_parcellate`, `split_subjects`. These define which subjects every experiment sees and which subjects are held out for test — changing them breaks comparability across runs.
- The literal `print(...)` lines under the `---` separator at the bottom of `main()`. The metric **keys**, **order**, and **format strings** are the contract the loop relies on. (The functions that *produce* the values are editable — only the final emission is fixed.)
- `pyproject.toml` (no new dependencies — use what's installed).

### Pretraining constraint

The encoder is being **pretrained**. No labels are allowed in the training loss. None of `Gender`, `Child_Adult`, `Age`, `AgeGroup`, `Handedness`, or any other column of `pheno` may enter any term that backpropagates to the encoder. Pretraining is self-supervised — the model gets the BOLD signal and nothing else.

The probe step (after training is over) is allowed to read train-subject labels and fit a classifier — that is what a probe *is* — but the encoder must already be done training by then and must not receive gradient from those labels. With the baseline's sklearn LDA this is automatic (LDA doesn't backprop). If you swap in a learned probe head, freeze the encoder before fitting it.

Forbidden: supervised auxiliary losses, label-conditioned masking, contrastive losses keyed on subject metadata, anything that lets the model peek. The point of the loop is pretraining research.

### Evaluation contract

Whatever you do internally, the reported `probe_ca_test_bal` and `probe_gender_test_bal` **must** be:

1. Computed on the **test subjects** as returned by `split_subjects(...)`. No peeking at train labels for feature normalization in a way that crosses the split, no training the classifier on test subjects.
2. **Subject-level**: produce exactly one prediction per test subject (one Child/Adult label, one Gender label), then compute `sklearn.metrics.balanced_accuracy_score(y_true_subjects, y_pred_subjects)`. *How* you collapse to one prediction per subject is your choice. Reporting a window-level number under the `probe_ca_test_bal` key violates the contract.

**The goal**: push **`probe_gender_test_bal`** up (subject-level balanced accuracy for Gender on the test set, computed per the Evaluation contract above) **without meaningfully degrading `probe_ca_test_bal`**. The child/adult task already runs near ceiling on this dataset (~0.94 baseline), so the dual constraint matters: a notable drop on CA disqualifies the change even if gender improves. Treat **CA as the guardrail, gender as the objective**.

**Why these metrics**: Gender on this dataset has historically sat near chance with masked-recon pretraining — the signal is much weaker than for age, which is why it's the harder target now. Both metrics are balanced accuracy because Child/Adult is heavily class-imbalanced (~80% child) and a single consistent scoring rule across the two probes keeps the comparison clean. Both are subject-level (one prediction per test subject, then balanced accuracy). With ~30 test subjects, deltas are quantized in steps tied to the per-class counts — judge improvements with that in mind, especially for CA where each adult flip is ~0.10.

**A note on iterating against test**: Since there is no val split, the test set is what you optimize against. With ~30 test subjects this is bounded but not zero leakage — over many runs you will indirectly overfit. Bias toward changes that are principled rather than ones that just nudge the test number; deletions and simplifications are especially safe.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a great outcome — that's a simplification win. A 0.001 bump in `probe_gender_test_bal` for 20 lines of hacky code? Probably not worth it. The same bump from deleting code? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline — just run the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
probe_ca_test_bal:     0.7140
probe_gender_test_bal: 0.5310
train_loss:            0.823145
training_seconds:      540.2
total_seconds:         598.1
num_params_M:          12.83
```

You can extract the key metrics from the log file:

```
grep "^probe_ca_test_bal:\|^train_loss:\|^num_params_M:" run.log
```

If a run finished cleanly, all metrics will appear. If the grep is empty, the run crashed — read the tail of the log for the stack trace.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	probe_ca_test_bal	probe_gender_test_bal	train_loss	status	description
```

1. git commit hash (short, 7 chars)
2. `probe_ca_test_bal` — guardrail metric (e.g. 0.9444). Use 0.0000 for crashes.
3. `probe_gender_test_bal` — objective metric (e.g. 0.5727). Use 0.0000 for crashes.
4. `train_loss` — informational (e.g. 0.7616). Use 0.0000 for crashes.
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried (no commas)

Older entries (from prior branches) may only have 5 columns — leave them as-is; they're frozen history.

Example:

```
commit	probe_ca_test_bal	probe_gender_test_bal	train_loss	status	description
a1b2c3d	0.9444	0.5727	0.7616	keep	baseline
b2c3d4e	0.9450	0.6500	0.7500	keep	add per-subject FFT features
c3d4e5f	0.8500	0.7000	0.8000	discard	gender up but ca degraded too much
d4e5f6g	0.0000	0.0000	0.0000	crash	double model width (OOM on MPS)
```

`results.tsv` is intentionally **not tracked** by git — leave it untracked. Each experiment's commit captures the code; the TSV is the local notebook. It **persists across branches**: new branches append below the existing rows, separated by a blank line. This is the only durable record of *failed* ideas, since `git reset --hard` wipes their commits from history — without it, future runs would happily re-try the same dead ends.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit you're on.
2. Tune `main.py` and/or `simple_model.py` with an experimental idea by directly hacking the code. Respect the read-only constraints in "Experimentation".
3. `git commit -am "<short description>"`.
4. Run the experiment: `uv run python main.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^probe_ca_test_bal:\|^probe_gender_test_bal:\|^train_loss:\|^total_seconds:\|^num_params_M:" run.log`.
6. If the grep output is empty or missing `probe_ca_test_bal`, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get it to work after a few attempts, give up and log "crash".
7. Record the results in `results.tsv` (do NOT commit `results.tsv`; leave it untracked).
8. If `probe_gender_test_bal` improved by a meaningful amount AND `probe_ca_test_bal` did not meaningfully degrade (use judgment — a one-subject-flip dip on CA is fine; a larger drop disqualifies even if gender improves), "advance" the branch — amend the commit to include both metrics (`git commit --amend -m "<desc> [gen=<probe_gender_test_bal> ca=<probe_ca_test_bal>]"`) so they're durable in git history, then keep it. Otherwise, `git reset --hard HEAD~1` back to where you started.
9. Go to step 1.

The idea is that you are a completely autonomous researcher trying things. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're stuck, you can rewind further, but do this very sparingly.

**Timeout**: Each experiment should take ~10 minutes total (9 min train + ~1 min eval). If a run exceeds 15 minutes, kill it and treat it as a crash.

**Crashes**: If a run crashes (OOM, bug, etc.), use judgment. If it's a small fix (typo, wrong shape), fix and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the user if you should continue. The user might be asleep, or away, and expects you to continue working *indefinitely* until they manually stop you. You are autonomous. If you run out of ideas, think harder — re-read `main.py` and `simple_model.py` for new angles, try combining previous near-misses, try radical architectural changes (different attention, different objective, different probe), revisit hyperparameters you haven't touched. The loop runs until interrupted, period.

## Ideas to try (non-exhaustive)

Just a seed list — you'll find better ideas as you go.

- **Optimizer**: AdamW with weight decay, different `lr`, warmup, cosine decay, gradient clipping.
- **Objective**: try BERT-style 80/10/10 (mask token / random / unchanged), masking whole timesteps vs random per-channel, predict only at masked positions vs everywhere, MSE → smooth L1.
- **Architecture**: change `depth`, `input_dim`, `hidden_dim`, `temporal_dim`, `ffw_dim`, `num_heads`, `preprocessor_hidden_dim`. Drop the input/hidden/temporal split entirely and let the encoder mix everywhere. Try residual order, different norms, different activation in `FFW`.
- **Windowing**: change `WINDOW`, switch to overlapping windows (stride < window), drop windowing and feed the full subject sequence. Watch the temporal-attention cost — full sequence is much longer than a window.
- **Feature pooling**: mean over time, attention-pooled token, last token, first token, concatenate multiple pooled slices, use the full output not just the hidden slice. The probe doesn't care where the features come from, only that they're per-window-or-per-subject.
- **Probe head**: try logistic regression, linear SVM, kNN, ridge classifier. LDA is just the baseline.
- **Subject aggregation**: majority vote of window predictions (baseline), mean of class probabilities then argmax, predict directly from a single pooled-per-subject feature, weighted vote by classifier confidence.
- **Training detail**: dropout, layer-wise LR, EMA of weights, gradient accumulation.
- **Data augmentation** (must not touch the subject split): noise on the input, time jitter, channel dropout.
