"""
run_experiments_v2.py  -  G1 multi-representation hotspot detection (v2)

What changed vs v1:
  * one preprocessing protocol for every method (224x224, same epochs, same split rule)
  * threshold and epoch chosen on the validation split only; test set scored once
  * 6 seeds (allows an exact two-sided Wilcoxon p-value below 0.05 in principle; significance is not guaranteed)\n  * persistent disk cache for resized/original, edge, and density channels
  * true fusion: early concat (3ch), pairwise concat (= leave-one-representation-out),
    late fusion (mean of single-model probabilities), learned-weight fusion
  * per-channel statistics to test the scale-mismatch explanation

Usage:
  1. set DATA_ROOT below to the folder that contains iccad1 ... iccad5
  2. python run_experiments_v2.py
Outputs: /mnt/c/Outputs/results_v2_cached/<bench>_seed<k>.json and summary.md\nPreprocessed image cache: /mnt/c/Outputs/preprocessed_cache_224/
"""
import os, json, random, hashlib
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from scipy.stats import wilcoxon

# ----------------------------- configuration --------------------------------
DATA_ROOT = "/mnt/c/Outputs/iccad_official/iccad-official"
OUT = "/mnt/c/Outputs/results_v2_cached"
BENCHES = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
SEEDS = [0, 1, 2, 3, 4, 5]
SIZE = 224                       # all images resized to SIZE x SIZE (same for every method)
EPOCHS, PATIENCE, BS, LR = 10, 4, 16, 1e-3  # 10 max epochs; early stopping; batch size 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/mnt/c/Outputs/preprocessed_cache_224"  # persistent cache shared across runs
CACHE_VERSION = "v1"
SINGLE = {"orig": [0], "edge": [1], "dens": [2]}
PAIRS = {"pair_orig_edge": [0, 1], "pair_orig_dens": [0, 2], "pair_edge_dens": [1, 2]}
CONCAT3 = [0, 1, 2]
THRS = np.linspace(0.05, 0.95, 91)
FEAT = 12 * (SIZE // 8) ** 2     # 9408 flattened features per branch at 224x224


# ------------------------------ data ----------------------------------------
def list_split(bench):
    base = os.path.join(DATA_ROOT, bench)

    def grab(sub, label):
        d = os.path.join(base, sub)
        return [(os.path.join(d, f), label) for f in sorted(os.listdir(d))]

    train = grab("train/train_hs", 1) + grab("train/train_nhs", 0)
    test = grab("test/test_hs", 1) + grab("test/test_nhs", 0)
    return train, test


def make_channels(path):
    """Return (3, SIZE, SIZE) float32 in [0,1]: [original, edge, density]."""
    img = np.array(Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)
    orig = img.astype(np.float32) / 255.0
    edge = cv2.Canny(img, 50, 150).astype(np.float32) / 255.0
    binary = (img > 127).astype(np.float32)
    dens = cv2.GaussianBlur(binary, (11, 11), 5)
    dens = dens / (dens.max() + 1e-8)
    return np.stack([orig, edge, dens]).astype(np.float32)


def cached_channels(path):
    """
    Persistent disk cache for deterministic preprocessing.
    Each source image is processed once for a given file version and SIZE.
    If an image changes, its modification time/size changes the cache key.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    abs_path = os.path.abspath(path)
    try:
        stat = os.stat(abs_path)
        source_version = f"{stat.st_mtime_ns}_{stat.st_size}"
    except OSError:
        source_version = "unknown"
    key_text = f"{abs_path}|{source_version}|size={SIZE}|{CACHE_VERSION}"
    key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    cache_path = os.path.join(CACHE_DIR, key + ".npy")

    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path, allow_pickle=False)
            if arr.shape == (3, SIZE, SIZE) and arr.dtype == np.float32:
                return arr
        except (OSError, ValueError):
            pass  # corrupt/incomplete cache: regenerate it

    arr = make_channels(path)
    temp_path = cache_path + f".{os.getpid()}.tmp.npy"
    np.save(temp_path, arr, allow_pickle=False)
    os.replace(temp_path, cache_path)
    return arr


class RepDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return torch.from_numpy(cached_channels(p)), y


def loaders(bench, seed):
    train, test = list_split(bench)
    ys = np.array([y for _, y in train])
    itr, iva = train_test_split(np.arange(len(train)), test_size=0.2, stratify=ys, random_state=seed)
    full = RepDS(train)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=BS, shuffle=shuffle, num_workers=0)
    return (mk(Subset(full, itr), True), mk(Subset(full, iva), False),
            mk(RepDS(test), False), ys[iva], np.array([y for _, y in test]))


# ------------------------------ models --------------------------------------
def block(c_in):
    return nn.Sequential(
        nn.Conv2d(c_in, 12, 3, padding=1), nn.ELU(),
        nn.Conv2d(12, 12, 3, padding=1), nn.ELU(),
        nn.Conv2d(12, 12, 3, padding=1), nn.BatchNorm2d(12), nn.ELU(),
        nn.MaxPool2d(2))


def extractor(c_in):
    return nn.Sequential(block(c_in), nn.MaxPool2d(2), block(12), nn.Flatten())


class CNN(nn.Module):
    """Single representation or early (channel) concatenation."""
    def __init__(self, c_in):
        super().__init__()
        self.net = nn.Sequential(extractor(c_in), nn.Dropout(0.3),
                                 nn.Linear(FEAT, 256), nn.ELU(), nn.Linear(256, 2))

    def forward(self, x):
        return self.net(x)


class LearnedFusion(nn.Module):
    """One branch per representation; features scaled by a learned global softmax weight."""
    def __init__(self, n=3):
        super().__init__()
        self.branches = nn.ModuleList([extractor(1) for _ in range(n)])
        self.gate = nn.Parameter(torch.zeros(n))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(n * FEAT, 256), nn.ELU(), nn.Linear(256, 2))

    def forward(self, x):
        w = torch.softmax(self.gate, 0)
        feats = [w[i] * b(x[:, i:i + 1]) for i, b in enumerate(self.branches)]
        return self.head(torch.cat(feats, 1))


# ----------------------------- training -------------------------------------
def ba(y, p, thr=0.5):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))


@torch.no_grad()
def predict(model, loader, chans):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        x_input = x[:, chans].to(DEVICE) if len(chans) > 1 else x[:, chans[0]:chans[0]+1].to(DEVICE)
        P.append(torch.softmax(model(x_input), 1)[:, 1].cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def train_model(model, chans, tr, va):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best, best_state, bad = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        for x, y in tr:
            opt.zero_grad()
            x_input = x[:, chans].to(DEVICE) if len(chans) > 1 else x[:, chans[0]:chans[0]+1].to(DEVICE)
            loss = crit(model(x_input), y.to(DEVICE))
            loss.backward()
            opt.step()
        pv, yv = predict(model, va, chans)
        score = ba(yv, pv)                      # early stopping on VALIDATION only
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec, spec, prec = tp / (tp + fn + 1e-9), tn / (tn + fp + 1e-9), tp / (tp + fp + 1e-9)
    return {"ba": 0.5 * (rec + spec), "recall": rec, "specificity": spec, "precision": prec,
            "f1": f1_score(y, pred, zero_division=0), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "thr": float(thr)}


def best_thr(y, p):
    """Threshold chosen to maximise VALIDATION balanced accuracy."""
    return float(max(THRS, key=lambda t: ba(y, p, t)))


def run_seed(bench, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tr, va, te, _, _ = loaders(bench, seed)
    store = {}
    for name, chans in SINGLE.items():
        m = train_model(CNN(1), chans, tr, va)
        store[name] = {"val": predict(m, va, chans), "test": predict(m, te, chans)}
    for name, chans in {**PAIRS, "concat3": CONCAT3}.items():
        m = train_model(CNN(len(chans)), chans, tr, va)
        store[name] = {"val": predict(m, va, chans), "test": predict(m, te, chans)}
    m = train_model(LearnedFusion(3), CONCAT3, tr, va)
    store["learned_fusion"] = {"val": predict(m, va, CONCAT3), "test": predict(m, te, CONCAT3)}
    # late fusion: average of the three single-model probabilities (no extra training)
    store["late_fusion"] = {s: (np.mean([store[k][s][0] for k in SINGLE], 0), store["orig"][s][1])
                            for s in ("val", "test")}
    rows = {}
    for name, d in store.items():
        thr = best_thr(d["val"][1], d["val"][0])
        rows[name] = {"val_ba": ba(d["val"][1], d["val"][0], thr), **metrics(d["test"][1], d["test"][0], thr)}
    return rows


def channel_stats(bench, n=200):
    """Per-representation scale statistics on a random training sample (evidence for scale mismatch)."""
    tr, _ = list_split(bench)
    sample = random.Random(0).sample(tr, min(n, len(tr)))
    arr = np.stack([cached_channels(p) for p, _ in sample])
    return {c: {"mean": float(arr[:, i].mean()), "std": float(arr[:, i].std())}
            for i, c in enumerate(["orig", "edge", "dens"])}


# ----------------------------- summary --------------------------------------
def summarize(allres):
    names = list(next(iter(allres.values())).keys())
    lines = ["# Results (mean ± std over %d seeds; threshold chosen on validation)" % len(SEEDS), ""]
    lines.append("Parameter count, single CNN: %d" % sum(p.numel() for p in CNN(1).parameters()))
    lines.append("Parameter count, learned fusion: %d" % sum(p.numel() for p in LearnedFusion(3).parameters()))
    for bench in BENCHES:
        lines += ["", f"## {bench}", "", "| Method | BA | Recall | Precision | Specificity | F1 |",
                  "|---|---|---|---|---|---|"]
        for name in names:
            cols = []
            for k in ["ba", "recall", "precision", "specificity", "f1"]:
                v = np.array([allres[(bench, s)][name][k] for s in SEEDS])
                cols.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}")
            lines.append(f"| {name} | " + " | ".join(cols) + " |")
        sel = []
        for s in SEEDS:
            r = allres[(bench, s)]
            best = max(SINGLE, key=lambda k: r[k]["val_ba"])   # chosen on validation
            sel.append(r[best]["ba"])
        sel = np.array(sel)
        lines.append("")
        lines.append(f"Validation-selected single representation: BA {sel.mean():.3f} ± {sel.std(ddof=1):.3f}")
        for other in ["concat3", "learned_fusion", "late_fusion"]:
            o = np.array([allres[(bench, s)][other]["ba"] for s in SEEDS])
            try:
                p = wilcoxon(sel, o).pvalue
            except ValueError:
                p = float("nan")
            lines.append(f"- vs {other}: mean diff {np.mean(sel - o):+.3f}, Wilcoxon p = {p:.4f}")
    lines += ["", "## Channel statistics (scale evidence)"]
    for bench in BENCHES:
        st = channel_stats(bench)
        lines.append(f"- {bench}: " + ", ".join(f"{c} mean={v['mean']:.3f} std={v['std']:.3f}" for c, v in st.items()))
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Settings: SIZE={SIZE}, EPOCHS={EPOCHS}, PATIENCE={PATIENCE}, BS={BS}, SEEDS={SEEDS}")
    print(f"Preprocessing cache: {CACHE_DIR}")
    print(f"Results directory: {OUT}")
    allres = {}
    for bench in BENCHES:
        for seed in SEEDS:
            path = os.path.join(OUT, f"{bench}_seed{seed}.json")
            if os.path.exists(path):
                with open(path) as f:
                    rows = json.load(f)
            else:
                rows = run_seed(bench, seed)
                with open(path, "w") as f:
                    json.dump(rows, f, indent=2)
            allres[(bench, seed)] = rows
            print(bench, "seed", seed, "done", flush=True)
    summarize(allres)


if __name__ == "__main__":
    main()
