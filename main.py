"""MLM pretrain on Schaefer parcels, then LDA probe (gender, child/adult).

Pretrains `SimpleBOLDEncoder` with masked time-step reconstruction, then averages
the hidden-dim slice across time per window and fits LDA on a subject-level
train/test split.
"""

import time

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from simple_model import SimpleBOLDEncoder

N_PARCELS = 1000
WINDOW = 32
MASK_RATIO = 0.25
N_SUBJECTS = None  # None -> all available
SPLIT = (0.8, 0.2)  # train, test
SPLIT_SEED = 0
MAX_TRAIN_MINUTES = 9  # leaves ~1 min for probe; keeps total under 10 min


# ---------- data ----------


def fetch_and_parcellate(n_subjects=N_SUBJECTS, n_rois=N_PARCELS):
    """Returns (series, subject_ids, phenotypic) in fetch order."""
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    fmri = datasets.fetch_development_fmri(n_subjects=n_subjects)
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        labels=atlas.labels,
        standardize="zscore_sample",
        memory="nilearn_cache",
        verbose=0,
    )
    series = [
        torch.from_numpy(masker.fit_transform(f, confounds=c)).float()
        for f, c in zip(fmri.func, fmri.confounds)
    ]
    subject_ids = list(fmri.phenotypic["participant_id"])
    pheno = fmri.phenotypic.reset_index(drop=True)
    return series, subject_ids, pheno


def split_subjects(n, fractions=SPLIT, seed=SPLIT_SEED):
    """Deterministic subject-level split. Returns (train_idx, test_idx)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(round(fractions[0] * n))
    return perm[:n_train], perm[n_train:]


class BOLDWindows(Dataset):
    def __init__(self, series, window=WINDOW, stride=None):
        self.series = series
        self.window = window
        if stride is None:
            stride = window
        self.index = [
            (i, s)
            for i, ts in enumerate(series)
            for s in range(0, ts.shape[0] - window + 1, stride)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sub, start = self.index[i]
        return self.series[sub][start : start + self.window]


# ---------- model ----------


class LitBOLD(L.LightningModule):
    def __init__(
        self,
        n_parcels=N_PARCELS,
        input_dim=512,
        hidden_dim=256,
        temporal_dim=256,
        ffw_dim=1024,
        depth=6,
        num_heads=4,
        preprocessor_hidden_dim=512,
        mask_ratio=MASK_RATIO,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = SimpleBOLDEncoder(
            n_parcels=n_parcels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            temporal_dim=temporal_dim,
            ffw_dim=ffw_dim,
            depth=depth,
            num_heads=num_heads,
            preprocessor_hidden_dim=preprocessor_hidden_dim,
        )
        self.head = nn.Linear(input_dim + hidden_dim + temporal_dim, n_parcels)

    def _masked_recon_loss(self, x):
        B, T, _ = x.shape
        mask = torch.rand(B, T, device=x.device) < self.hparams.mask_ratio
        if not mask.any():
            mask[:, 0] = True
        pred = self.head(self.encoder(x, mask=mask))
        return F.mse_loss(pred[mask], x[mask])

    def training_step(self, batch, batch_idx):
        loss = self._masked_recon_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)


# ---------- linear probe ----------


def extract_hidden_features(model, series_list):
    """Per window, mean of the hidden-dim slice across time.

    Returns:
        features: [N, hidden_dim] float array, one row per window.
        subject:  [N] int array, the subject index for each row.
    """
    in_dim = model.hparams.input_dim
    hid_dim = model.hparams.hidden_dim
    device = next(model.parameters()).device
    feats, subjs = [], []
    model.eval()
    with torch.no_grad():
        for subj_i, series in enumerate(series_list):
            ds = BOLDWindows([series], stride=WINDOW // 2)
            if len(ds) == 0:
                continue
            x = torch.stack([ds[i] for i in range(len(ds))]).to(device)
            h = model.encoder(x)  # [n_windows, T, in+hid+t]
            hid = h[..., in_dim : in_dim + hid_dim].mean(dim=1)  # [n_windows, hid]
            feats.append(hid.cpu().numpy())
            subjs.extend([subj_i] * len(ds))
    return np.concatenate(feats, axis=0), np.array(subjs)


def fit_and_eval(label_name, y_subject, train_idx, test_idx, X_all, subj_all):
    """Fit LDA on train windows; report window + subject-vote metrics on train and test.

    Returns a dict keyed by split ('train'/'test') with metric subdicts.
    """

    def collect(idx):
        keep = np.isin(subj_all, idx)
        return X_all[keep], y_subject[subj_all[keep]], subj_all[keep]

    X_tr, y_tr, s_tr = collect(train_idx)
    X_te, y_te, s_te = collect(test_idx)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_tr, y_tr)

    results = {}

    def report(split, X, y, s):
        key = split.strip()
        if len(X) == 0:
            print(f"  {split}: no samples")
            results[key] = None
            return
        pred = lda.predict(X)
        win_acc = accuracy_score(y, pred)
        win_bal = balanced_accuracy_score(y, pred)
        # Subject majority vote
        subj_pred, subj_true = [], []
        for sj in np.unique(s):
            mask = s == sj
            vals, counts = np.unique(pred[mask], return_counts=True)
            subj_pred.append(vals[counts.argmax()])
            subj_true.append(y[mask][0])
        sub_acc = accuracy_score(subj_true, subj_pred)
        sub_bal = balanced_accuracy_score(subj_true, subj_pred)
        results[key] = dict(
            win_acc=win_acc, win_bal=win_bal,
            sub_acc=sub_acc, sub_bal=sub_bal,
            n_windows=len(X), n_subjects=len(subj_true),
        )
        print(
            f"  {split}: window acc={win_acc:.3f} bal={win_bal:.3f}"
            f" | subject acc={sub_acc:.3f} bal={sub_bal:.3f}"
            f" ({len(X)} windows, {len(subj_true)} subjects)"
        )

    print(f"=== {label_name} (classes={list(lda.classes_)}) ===")
    print(f"  train labels: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    report("train", X_tr, y_tr, s_tr)
    report("test ", X_te, y_te, s_te)
    return results


# ---------- entrypoint ----------


def _sub_bal(metrics, split):
    if metrics is None or metrics.get(split) is None:
        return float("nan")
    return metrics[split]["sub_bal"]


def main():
    t0 = time.monotonic()
    series, subject_ids, pheno = fetch_and_parcellate()
    train_idx, test_idx = split_subjects(len(series))
    print(f"train subjects ({len(train_idx)}): {[subject_ids[i] for i in train_idx]}")
    print(f"test  subjects ({len(test_idx)}): {[subject_ids[i] for i in test_idx]}")

    def make_loader(idx, shuffle):
        ds = BOLDWindows([series[i] for i in idx], stride=WINDOW // 2)
        return DataLoader(ds, batch_size=4, shuffle=shuffle, num_workers=0)

    model = LitBOLD()
    trainer = L.Trainer(
        max_time={"minutes": MAX_TRAIN_MINUTES},
        accelerator="auto",
        log_every_n_steps=10,
        enable_progress_bar=False,
        gradient_clip_val=1.0,
    )
    t_fit_start = time.monotonic()
    trainer.fit(model, make_loader(train_idx, True))
    t_train = time.monotonic() - t_fit_start
    train_loss = float(trainer.callback_metrics.get("train_loss", float("nan")))

    X_all, subj_all = extract_hidden_features(model, series)
    print(f"\nfeature matrix: {X_all.shape}, hidden_dim={model.hparams.hidden_dim}")

    gender = pheno["Gender"].to_numpy()
    child_adult = pheno["Child_Adult"].to_numpy()
    gen_m = fit_and_eval("Gender (M/F)", gender, train_idx, test_idx, X_all, subj_all)
    ca_m = fit_and_eval(
        "Child vs Adult", child_adult, train_idx, test_idx, X_all, subj_all
    )

    n_params = sum(p.numel() for p in model.parameters())
    t_total = time.monotonic() - t0

    # Parseable summary (one metric per line). Primary metric: probe_ca_test_bal (subject vote).
    print()
    print("---")
    print(f"probe_ca_test_bal:     {_sub_bal(ca_m, 'test'):.4f}")
    print(f"probe_gender_test_bal: {_sub_bal(gen_m, 'test'):.4f}")
    print(f"train_loss:            {train_loss:.6f}")
    print(f"training_seconds:      {t_train:.1f}")
    print(f"total_seconds:         {t_total:.1f}")
    print(f"num_params_M:          {n_params / 1e6:.2f}")


if __name__ == "__main__":
    main()
