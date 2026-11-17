"""
run_experiments_v3_efficient.py - G1 multi-representation hotspot detection

Same protocol as run_experiments_v2_cached_ac0566.py (splits, CNN, training budget,
threshold rule, seeds). Only the storage and evaluation pipeline changed:

  * Each image is stored ONCE as 224x224 uint8 grayscale (50 KB), in one memmap per split
    (instead of one 602 KB float32 3-channel .npy per image).
  * The three channels (original, Canny edge, Gaussian density) are derived from that
    grayscale on the fly. The derivation is bit-identical to the v2 code, so ICCAD-1/2
    results stay reproducible.
  * Validation/test batches use a larger batch size (EVAL_BS). model.eval() makes outputs
    independent of batch size, so results do not change.
  * Finished (benchmark, seed) JSON files are skipped, so ICCAD-1/2 are not rerun.

Usage:
  python run_experiments_v3_efficient.py --verify                    # check derivation == v2
  python run_experiments_v3_efficient.py --benches iccad3 iccad4 iccad5
  python run_experiments_v3_efficient.py                             # everything missing
"""
import os, json, random, argparse, hashlib
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
DATA_ROOT = "/mnt/c/Outputs/iccad_official/iccad-official"   # folder containing iccad1 ... iccad5
OUT = "/mnt/c/Outputs/results_v2_cached"                     # same folder: finished seeds are reused
CACHE_DIR = "/mnt/c/Outputs/gray224_cache"                   # uint8 grayscale cache (~8 GB for all 5)
BENCHES = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
SEEDS = [0, 1, 2, 3, 4, 5]
SIZE = 224
EPOCHS, PATIENCE, BS, LR = 10, 4, 16, 1e-3   # training protocol: unchanged from v2
EVAL_BS = 256                                 # evaluation only; does not change outputs
NUM_WORKERS = 0                               # keep 0 on Windows unless tested
CACHE_VERSION = "gray224-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SINGLE = {"orig": [0], "edge": [1], "dens": [2]}
PAIRS = {"pair_orig_edge": [0, 1], "pair_orig_dens": [0, 2], "pair_edge_dens": [1, 2]}
CONCAT3 = [0, 1, 2]
THRS = np.linspace(0.05, 0.95, 91)
FEAT = 12 * (SIZE // 8) ** 2                  # 9408 flattened features per branch


# ------------------------------ data ----------------------------------------
def list_split(bench):
    base = os.path.join(DATA_ROOT, bench)

    def grab(sub, label):
        d = os.path.join(base, sub)
        return [(os.path.join(d, f), label) for f in sorted(os.listdir(d))]

    train = grab("train/train_hs", 1) + grab("train/train_nhs", 0)
    test = grab("test/test_hs", 1) + grab("test/test_nhs", 0)
    return train, test


def load_gray(path):
    """1200x1200 PNG -> 224x224 uint8 grayscale (same resize as v2)."""
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def derive_channels(gray):
    """(3, SIZE, SIZE) float32 in [0,1]: [original, edge, density]. Identical to v2 make_channels()."""
    orig = gray.astype(np.float32) / 255.0
    edge = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    binary = (gray > 127).astype(np.float32)
    dens = cv2.GaussianBlur(binary, (11, 11), 5)
    dens = dens / (dens.max() + 1e-8)
    return np.stack([orig, edge, dens]).astype(np.float32)


def _manifest(items):
    """Fingerprint of the source files and settings; any change forces a rebuild."""
    h = hashlib.sha256()
    for p, _ in items:
        st = os.stat(p)
        h.update(f"{os.path.abspath(p)}|{st.st_mtime_ns}|{st.st_size}\n".encode())
    return {"version": CACHE_VERSION, "size": SIZE, "n": len(items), "digest": h.hexdigest()}


_GRAY = {}   # in-process memo: name -> memmap


def build_gray_cache(name, items):
    """One uint8 memmap per split, rows in list order. Rebuilt only if sources or settings change."""
    if name in _GRAY:
        return _GRAY[name]
    os.makedirs(CACHE_DIR, exist_ok=True)
    npy = os.path.join(CACHE_DIR, name + ".npy")
    meta = os.path.join(CACHE_DIR, name + ".json")
    want = _manifest(items)
    if os.path.exists(npy) and os.path.exists(meta):
        with open(meta) as f:
            if json.load(f) == want:
                _GRAY[name] = np.load(npy, mmap_mode="r")
                return _GRAY[name]
    tmp = npy + ".tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(items), SIZE, SIZE))
    for i, (p, _) in enumerate(items):
        arr[i] = load_gray(p)
    arr.flush()
    del arr
    os.replace(tmp, npy)                       # only a complete file ever gets the final name
    with open(meta, "w") as f:
        json.dump(want, f)
    _GRAY[name] = np.load(npy, mmap_mode="r")
    return _GRAY[name]


class ChannelDS(Dataset):
    def __init__(self, items, gray):
        self.items, self.gray = items, gray

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return torch.from_numpy(derive_channels(np.array(self.gray[i]))), self.items[i][1]


def make_loaders(bench, seed):
    train, test = list_split(bench)
    gtr = build_gray_cache(f"{bench}_train", train)
    gte = build_gray_cache(f"{bench}_test", test)
    ys = np.array([y for _, y in train])
    itr, iva = train_test_split(np.arange(len(train)), test_size=0.2, stratify=ys, random_state=seed)
    full = ChannelDS(train, gtr)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    tr = DataLoader(Subset(full, itr), batch_size=BS, shuffle=True, **kw)
    va = DataLoader(Subset(full, iva), batch_size=EVAL_BS, shuffle=False, **kw)
    te = DataLoader(ChannelDS(test, gte), batch_size=EVAL_BS, shuffle=False, **kw)
    return tr, va, te, ys[iva], np.array([y for _, y in test])


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
    def __init__(self, c_in):
        super().__init__()
        self.net = nn.Sequential(extractor(c_in), nn.Dropout(0.3),
                                 nn.Linear(FEAT, 256), nn.ELU(), nn.Linear(256, 2))

    def forward(self, x):
        return self.net(x)


class LearnedFusion(nn.Module):
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
def select(x, chans):
    return x[:, chans] if len(chans) > 1 else x[:, chans[0]:chans[0] + 1]


def ba(y, p, thr=0.5):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))


@torch.inference_mode()
def predict(model, loader, chans):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(torch.softmax(model(select(x, chans).to(DEVICE)), 1)[:, 1].cpu().numpy())
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
            loss = crit(model(select(x, chans).to(DEVICE)), y.to(DEVICE))
            loss.backward()
            opt.step()
        pv, yv = predict(model, va, chans)
        score = ba(yv, pv)                     # early stopping on VALIDATION only
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
    tr, va, te, _, _ = make_loaders(bench, seed)
    store = {}
    for name, chans in SINGLE.items():
        m = train_model(CNN(1), chans, tr, va)
        store[name] = {"val": predict(m, va, chans), "test": predict(m, te, chans)}
    for name, chans in {**PAIRS, "concat3": CONCAT3}.items():
        m = train_model(CNN(len(chans)), chans, tr, va)
        store[name] = {"val": predict(m, va, chans), "test": predict(m, te, chans)}
    m = train_model(LearnedFusion(3), CONCAT3, tr, va)
    store["learned_fusion"] = {"val": predict(m, va, CONCAT3), "test": predict(m, te, CONCAT3)}
    # late fusion: mean of the three single-model probabilities (no extra training)
    store["late_fusion"] = {s: (np.mean([store[k][s][0] for k in SINGLE], 0), store["orig"][s][1])
                            for s in ("val", "test")}
    rows = {}
    for name, d in store.items():
        thr = best_thr(d["val"][1], d["val"][0])
        rows[name] = {"val_ba": ba(d["val"][1], d["val"][0], thr), **metrics(d["test"][1], d["test"][0], thr)}
    return rows


# ------------------------- checks and summary --------------------------------
def v2_make_channels(path):
    """Verbatim preprocessing from run_experiments_v2_cached_ac0566.py, used only for --verify."""
    img = np.array(Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)
    orig = img.astype(np.float32) / 255.0
    edge = cv2.Canny(img, 50, 150).astype(np.float32) / 255.0
    binary = (img > 127).astype(np.float32)
    dens = cv2.GaussianBlur(binary, (11, 11), 5)
    dens = dens / (dens.max() + 1e-8)
    return np.stack([orig, edge, dens]).astype(np.float32)


def verify(bench, n=5):
    train, _ = list_split(bench)
    gray = build_gray_cache(f"{bench}_train", train)
    for i in random.Random(1).sample(range(len(train)), n):
        assert np.array_equal(v2_make_channels(train[i][0]), derive_channels(np.array(gray[i]))), train[i][0]
    print(f"{bench}: derivation identical to v2 on {n} images")


def load_results():
    res = {}
    for b in BENCHES:
        for s in SEEDS:
            p = os.path.join(OUT, f"{b}_seed{s}.json")
            if os.path.exists(p):
                with open(p) as f:
                    res[(b, s)] = json.load(f)
    return res


def summarize(res):
    lines = ["# Results (mean ± std over seeds; threshold chosen on validation)", ""]
    lines.append("Parameter count, single CNN: %d" % sum(p.numel() for p in CNN(1).parameters()))
    lines.append("Parameter count, learned fusion: %d" % sum(p.numel() for p in LearnedFusion(3).parameters()))
    for bench in BENCHES:
        seeds = [s for s in SEEDS if (bench, s) in res]
        if not seeds:
            continue
        names = list(res[(bench, seeds[0])].keys())
        lines += ["", f"## {bench} ({len(seeds)} seeds)", "",
                  "| Method | BA | Recall | Precision | Specificity | F1 |", "|---|---|---|---|---|---|"]
        for name in names:
            cols = []
            for k in ["ba", "recall", "precision", "specificity", "f1"]:
                v = np.array([res[(bench, s)][name][k] for s in seeds])
                cols.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}")
            lines.append(f"| {name} | " + " | ".join(cols) + " |")
        if len(seeds) < 2:
            continue
        sel = np.array([res[(bench, s)][max(SINGLE, key=lambda k: res[(bench, s)][k]["val_ba"])]["ba"]
                        for s in seeds])
        lines += ["", f"Validation-selected single representation: BA {sel.mean():.3f} ± {sel.std(ddof=1):.3f}"]
        for other in ["concat3", "learned_fusion", "late_fusion"]:
            o = np.array([res[(bench, s)][other]["ba"] for s in seeds])
            try:
                p = wilcoxon(o, sel, alternative="two-sided", method="exact").pvalue
            except ValueError:
                p = float("nan")
            lines.append(f"- {other} vs SEL: mean diff {np.mean(o - sel):+.4f}, Wilcoxon exact two-sided p = {p:.4f}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", default=BENCHES)
    ap.add_argument("--verify", action="store_true", help="check derivation against v2 and exit")
    args = ap.parse_args()
    print(f"Device: {DEVICE}  BS={BS}  EVAL_BS={EVAL_BS}  EPOCHS={EPOCHS}  PATIENCE={PATIENCE}")
    print(f"Cache: {CACHE_DIR}   Results: {OUT}")
    if args.verify:
        for b in args.benches:
            verify(b)
        return
    os.makedirs(OUT, exist_ok=True)
    for bench in args.benches:
        for seed in SEEDS:
            path = os.path.join(OUT, f"{bench}_seed{seed}.json")
            if os.path.exists(path):
                print(bench, "seed", seed, "already done, skipping", flush=True)
                continue
            rows = run_seed(bench, seed)
            with open(path, "w") as f:
                json.dump(rows, f, indent=2)
            print(bench, "seed", seed, "done", flush=True)
    summarize(load_results())


if __name__ == "__main__":
    main()