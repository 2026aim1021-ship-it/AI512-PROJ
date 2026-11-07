"""
rgbt_eval.py -- Evaluation toolkit for the CMX / CMRF RGB-Thermal project (PST900)

1. PST900Dataset  : reads images exactly like the official CMX repo (OpenCV;
                    BGR channel order by default), labels at ORIGINAL resolution,
                    out-of-range labels -> 255 (ignored, never clipped into a class),
                    optional seeded corruption hook.
2. Corruptions    : fixed RGB / thermal degradations, 3 severities, global and LOCALIZED.
3. SegMetrics     : confusion-matrix mIoU, Dice, Sensitivity, Specificity
                    (per class / mean over all classes / mean over object classes).
4. evaluate(), run_benchmark(), robustness_summary()
                    sliding-window inference (like the CMX evaluator), clean + corrupted.
5. model_cost()   : params, GFLOPs (fvcore), latency, FPS.
"""

import os
import csv
import glob
import time
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = 255

# Verify this order against the official PST900 README before reporting results.
PST900_CLASSES = ["Background", "Fire-Extinguisher", "Backpack", "Hand-Drill", "Survivor"]
PST900_PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]  # RGB

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# =============================================================================
# 1. Corruptions (uint8 numpy; RGB image is HxWx3 in RGB order, thermal HxW)
# =============================================================================
# Keep these FIXED for the whole project so E1-E4 are scored on identical inputs.
SEVERITY = {
    "low_light": [(0.45, 1.5, 4), (0.30, 2.0, 8), (0.15, 2.5, 12)],  # brightness, gamma, noise std
    "noise":     [10, 20, 35],            # gaussian noise std (0-255)
    "blur":      [2.0, 4.0, 7.0],         # gaussian sigma in px (at 720x1280)
    "occlusion": [0.10, 0.20, 0.35],      # fraction of image area covered by patches
}

CORRUPTIONS = {
    "rgb_low_light":     ("rgb", "low_light"),
    "rgb_noise":         ("rgb", "noise"),
    "rgb_blur":          ("rgb", "blur"),
    "rgb_occlusion":     ("rgb", "occlusion"),      # localized
    "thermal_noise":     ("thermal", "noise"),
    "thermal_blur":      ("thermal", "blur"),
    "thermal_occlusion": ("thermal", "occlusion"),  # localized
}


def _gaussian_noise(img, std, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0.0, std, img.shape), 0, 255).astype(np.uint8)


def _blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _low_light(img, brightness, gamma, noise_std, rng):
    x = np.power((img.astype(np.float32) / 255.0) * brightness, gamma) * 255.0
    x = x + rng.normal(0.0, noise_std, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def _occlusion(img, area_frac, rng, fill="noise"):
    out = img.copy()
    h, w = img.shape[:2]
    n = int(rng.integers(3, 7))
    patch_area = area_frac * h * w / n
    for _ in range(n):
        ar = rng.uniform(0.5, 2.0)
        ph = int(min(h, max(4, np.sqrt(patch_area / ar))))
        pw = int(min(w, max(4, patch_area / max(ph, 1))))
        y0, x0 = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
        region = out[y0:y0 + ph, x0:x0 + pw]
        out[y0:y0 + ph, x0:x0 + pw] = rng.integers(0, 256, region.shape) if fill == "noise" else 0
    return out


def apply_corruption(rgb, thermal, name, severity, rng):
    """Corrupt ONE modality. rgb: HxWx3 uint8, thermal: HxW uint8."""
    if name in (None, "clean"):
        return rgb, thermal
    modality, kind = CORRUPTIONS[name]
    s = int(severity) - 1
    x = rgb if modality == "rgb" else thermal
    if kind == "noise":
        x = _gaussian_noise(x, SEVERITY["noise"][s], rng)
    elif kind == "blur":
        x = _blur(x, SEVERITY["blur"][s])
    elif kind == "low_light":
        x = _low_light(x, *SEVERITY["low_light"][s], rng)
    elif kind == "occlusion":
        x = _occlusion(x, SEVERITY["occlusion"][s], rng, "black" if modality == "rgb" else "noise")
    return (x, thermal) if modality == "rgb" else (rgb, x)


def random_training_corruption(rgb, thermal, rng, p_corrupt=0.5):
    """For E3: with prob p_corrupt, corrupt one randomly chosen modality at random severity."""
    if rng.random() > p_corrupt:
        return rgb, thermal, "clean"
    name = str(rng.choice(list(CORRUPTIONS)))
    sev = int(rng.integers(1, 4))
    rgb, thermal = apply_corruption(rgb, thermal, name, sev, rng)
    return rgb, thermal, f"{name}@{sev}"


# =============================================================================
# 2. Dataset
# =============================================================================
def read_rgb(path):
    """Returns RGB-order uint8 HxWx3."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_thermal(path):
    """Same as the CMX repo: OpenCV grayscale (16-bit files are reduced to 8-bit)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_label(path):
    lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(path)
    if lab.ndim == 3:   # colour-coded label: not an index map -> must be converted first
        raise ValueError(f"{path} is a 3-channel label; convert it to a class-index map first.")
    return lab


def _normalize(img):
    return (img.astype(np.float64) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


class PST900Dataset(Dataset):
    """
    Layout: <root>/<split>/{rgb,thermal,labels}/*.png
    channel_order: "BGR" to match a model trained with the official CMX repo
                   (its loader feeds BGR), "RGB" for your own pipelines.
    Eval mode returns the FULL-resolution image (sliding-window inference in evaluate()).
    """

    def __init__(self, root_dir, split="test", is_training=False, crop_size=(480, 640),
                 scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75), corruption=None, severity=1,
                 seed=0, train_corruption_p=0.0, channel_order="BGR", num_classes=5):
        super().__init__()
        self.is_training, self.crop_size, self.scales = is_training, crop_size, scales
        self.corruption, self.severity, self.seed = corruption, severity, seed
        self.train_corruption_p, self.channel_order = train_corruption_p, channel_order
        self.num_classes = num_classes

        candidates = [os.path.join(root_dir, split),
                      os.path.join(root_dir, "PST900_RGBT_Dataset", split)]
        self.split_dir = next((p for p in candidates if os.path.isdir(p)), None)
        if self.split_dir is None:
            raise FileNotFoundError(f"Cannot locate '{split}' split in {root_dir}")
        self.rgb_dir = os.path.join(self.split_dir, "rgb")
        self.thermal_dir = os.path.join(self.split_dir, "thermal")
        self.label_dir = os.path.join(self.split_dir, "labels")
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, "*.png")) +
                                glob.glob(os.path.join(self.rgb_dir, "*.jpg")))
        if not self.rgb_files:
            raise RuntimeError(f"No RGB images in {self.rgb_dir}")

    def __len__(self):
        return len(self.rgb_files)

    @staticmethod
    def find(directory, basename):
        p = os.path.join(directory, basename)
        if os.path.exists(p):
            return p
        c = glob.glob(os.path.join(directory, os.path.splitext(basename)[0] + ".*"))
        if not c:
            raise FileNotFoundError(f"No match for {basename} in {directory}")
        return c[0]

    def load_raw(self, idx):
        f = self.rgb_files[idx]
        name = os.path.basename(f)
        rgb = read_rgb(f)
        th = read_thermal(self.find(self.thermal_dir, name))
        lab = read_label(self.find(self.label_dir, name)).astype(np.int64)
        lab[(lab < 0) | (lab >= self.num_classes)] = IGNORE_INDEX
        return rgb, th, lab, name

    def _train_aug(self, rgb, th, lab):
        if random.random() >= 0.5:
            rgb, th, lab = rgb[:, ::-1], th[:, ::-1], lab[:, ::-1]
        s = random.choice(self.scales)
        h, w = int(rgb.shape[0] * s), int(rgb.shape[1] * s)
        rgb = cv2.resize(np.ascontiguousarray(rgb), (w, h), interpolation=cv2.INTER_LINEAR)
        th = cv2.resize(np.ascontiguousarray(th), (w, h), interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(np.ascontiguousarray(lab).astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
        ch, cw = self.crop_size
        ph, pw = max(0, ch - h), max(0, cw - w)
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)))
            th = np.pad(th, ((0, ph), (0, pw)))
            lab = np.pad(lab, ((0, ph), (0, pw)), constant_values=IGNORE_INDEX)  # padded = ignored
        H, W = lab.shape
        y, x = random.randint(0, H - ch), random.randint(0, W - cw)
        return rgb[y:y + ch, x:x + cw], th[y:y + ch, x:x + cw], lab[y:y + ch, x:x + cw]

    def __getitem__(self, idx):
        rgb, th, lab, name = self.load_raw(idx)

        if self.is_training and self.train_corruption_p > 0:
            rgb, th, _ = random_training_corruption(rgb, th, np.random.default_rng(),
                                                    self.train_corruption_p)
        elif not self.is_training and self.corruption not in (None, "clean"):
            rng = np.random.default_rng(self.seed * 100003 + idx)  # identical every run
            rgb, th = apply_corruption(rgb, th, self.corruption, self.severity, rng)

        if self.is_training:
            rgb, th, lab = self._train_aug(rgb, th, lab)

        if self.channel_order == "BGR":
            rgb = rgb[..., ::-1]
        th3 = np.stack([th] * 3, axis=-1)
        rgb_t = torch.from_numpy(np.ascontiguousarray(_normalize(rgb).transpose(2, 0, 1))).float()
        th_t = torch.from_numpy(np.ascontiguousarray(_normalize(th3).transpose(2, 0, 1))).float()
        return {"rgb": rgb_t, "thermal": th_t,
                "label": torch.from_numpy(np.ascontiguousarray(lab)).long(), "name": name}


def inspect_dataset(root_dir, split="train", n=50):
    """Prints label values, image modes and sizes. Run once before training."""
    ds = PST900Dataset(root_dir, split)
    vals, info = set(), {}
    for f in ds.rgb_files[:n]:
        name = os.path.basename(f)
        lab = cv2.imread(ds.find(ds.label_dir, name), cv2.IMREAD_UNCHANGED)
        th = cv2.imread(ds.find(ds.thermal_dir, name), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        vals |= set(np.unique(lab).tolist())
        info = {"rgb": (rgb.shape, rgb.dtype), "thermal": (th.shape, th.dtype, int(th.min()), int(th.max())),
                "label": (lab.shape, lab.dtype)}
    print(f"[{split}] {len(ds)} samples")
    print(f"  label values (first {n}): {sorted(vals)}   <- expect a subset of 0..4 (and maybe 255)")
    for k, v in info.items():
        print(f"  {k:8s}: {v}")
    if any(v >= ds.num_classes and v != IGNORE_INDEX for v in vals):
        print("  WARNING: label values outside 0..4 -> check the label encoding before training!")
    return sorted(vals)


# =============================================================================
# 3. Metrics
# =============================================================================
class SegMetrics:
    """Confusion matrix: rows = ground truth, cols = prediction."""

    def __init__(self, num_classes, class_names=None, ignore_index=IGNORE_INDEX, bg_index=0):
        self.n = num_classes
        self.names = class_names or [str(i) for i in range(num_classes)]
        self.ignore_index, self.bg_index = ignore_index, bg_index
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred, target):
        pred, target = pred.reshape(-1).cpu(), target.reshape(-1).cpu()
        keep = target != self.ignore_index
        idx = target[keep] * self.n + pred[keep]
        self.cm += torch.bincount(idx, minlength=self.n ** 2).view(self.n, self.n).double()

    def compute(self):
        cm = self.cm
        tp = cm.diag()
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        tn = cm.sum() - tp - fp - fn
        present = cm.sum(1) > 0

        def safe(a, b):
            return torch.where(b > 0, a / b.clamp(min=1e-12), torch.full_like(a, float("nan")))

        per = {"IoU": safe(tp, tp + fp + fn), "Dice": safe(2 * tp, 2 * tp + fp + fn),
               "Sensitivity": safe(tp, tp + fn), "Specificity": safe(tn, tn + fp)}
        obj = torch.tensor([i != self.bg_index for i in range(self.n)]) & present
        res = {"pixel_acc": (tp.sum() / cm.sum()).item()}
        for k, v in per.items():
            short = "mIoU" if k == "IoU" else f"m{k}"
            res[short] = torch.nanmean(v[present]).item()
            res[f"{short}_obj"] = torch.nanmean(v[obj]).item()
            for i, nm in enumerate(self.names):
                res[f"{k}/{nm}"] = v[i].item()
        return res


# =============================================================================
# 4. Inference & evaluation
# =============================================================================
def _default_forward(model, rgb, thermal):
    out = model(rgb, thermal)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def predict_probs(model, rgb, th, forward_fn=_default_forward, crop=(480, 640),
                  stride_rate=2 / 3, flip=False):
    """Sliding-window softmax probabilities at input resolution (B x C x H x W)."""
    def run(r, t):
        p = forward_fn(model, r, t).float().softmax(1)
        if flip:
            p = p + forward_fn(model, r.flip(-1), t.flip(-1)).float().softmax(1).flip(-1)
        return p

    B, _, H, W = rgb.shape
    if crop is None:
        return run(rgb, th)
    ch, cw = crop
    ph, pw = max(0, ch - H), max(0, cw - W)
    if ph or pw:
        rgb, th = F.pad(rgb, (0, pw, 0, ph)), F.pad(th, (0, pw, 0, ph))
    Hp, Wp = rgb.shape[-2:]
    sh, sw = int(np.ceil(ch * stride_rate)), int(np.ceil(cw * stride_rate))
    ys = list(range(0, max(Hp - ch, 0) + 1, sh))
    xs = list(range(0, max(Wp - cw, 0) + 1, sw))
    if ys[-1] != Hp - ch:
        ys.append(Hp - ch)
    if xs[-1] != Wp - cw:
        xs.append(Wp - cw)
    probs, count = None, torch.zeros(1, 1, Hp, Wp, device=rgb.device)
    for y in ys:
        for x in xs:
            p = run(rgb[..., y:y + ch, x:x + cw], th[..., y:y + ch, x:x + cw])
            if probs is None:
                probs = torch.zeros(B, p.shape[1], Hp, Wp, device=rgb.device)
            probs[..., y:y + ch, x:x + cw] += p
            count[..., y:y + ch, x:x + cw] += 1
    return (probs / count)[..., :H, :W]


@torch.no_grad()
def evaluate(model, loader, num_classes=5, forward_fn=_default_forward, device=None,
             crop=(480, 640), stride_rate=2 / 3, flip=False, class_names=PST900_CLASSES):
    device = device or next(model.parameters()).device
    model.eval()
    metrics = SegMetrics(num_classes, class_names)
    for batch in loader:
        rgb, th = batch["rgb"].to(device), batch["thermal"].to(device)
        label = batch["label"].to(device)
        probs = predict_probs(model, rgb, th, forward_fn, crop, stride_rate, flip)
        if probs.shape[-2:] != label.shape[-2:]:
            probs = F.interpolate(probs, size=label.shape[-2:], mode="bilinear", align_corners=False)
        metrics.update(probs.argmax(1), label)
    return metrics.compute()


def run_benchmark(model, root_dir, forward_fn=_default_forward, corruptions=None,
                  severities=(1, 2, 3), crop=(480, 640), stride_rate=2 / 3, flip=False,
                  channel_order="BGR", num_workers=2, seed=0, out_csv=None, tag="", verbose=True):
    """Clean + every corruption x severity on the test split. Returns list of dicts."""
    corruptions = corruptions if corruptions is not None else list(CORRUPTIONS)
    conditions = [("clean", 0)] + [(c, s) for c in corruptions for s in severities]
    rows = []
    for cname, sev in conditions:
        ds = PST900Dataset(root_dir, "test", corruption=cname, severity=max(sev, 1),
                           seed=seed, channel_order=channel_order)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
        r = evaluate(model, dl, len(PST900_CLASSES), forward_fn, crop=crop,
                     stride_rate=stride_rate, flip=flip)
        rows.append({"tag": tag, "condition": cname, "severity": sev, **r})
        if verbose:
            print(f"{tag:>6s} {cname:>18s} s{sev} | mIoU {r['mIoU']*100:6.2f} | mDice {r['mDice']*100:6.2f}"
                  f" | mSens {r['mSensitivity']*100:6.2f} | mSpec {r['mSpecificity']*100:6.2f}"
                  f" | mIoU(obj) {r['mIoU_obj']*100:6.2f}")
    if out_csv:
        new = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        if verbose:
            print(f"appended {len(rows)} rows to {out_csv}")
    return rows


def robustness_summary(rows):
    clean = next(r for r in rows if r["condition"] == "clean")
    corr = [r for r in rows if r["condition"] != "clean"]
    out = {}
    for k in ("mIoU", "mDice", "mSensitivity", "mSpecificity", "mIoU_obj"):
        avg = float(np.mean([r[k] for r in corr]))
        out[f"{k}_clean"], out[f"{k}_corrupt_avg"], out[f"{k}_drop"] = clean[k], avg, clean[k] - avg
    for mod in ("rgb", "thermal"):
        sub = [r["mIoU"] for r in corr if r["condition"].startswith(mod)]
        out[f"mIoU_{mod}_degraded_avg"] = float(np.mean(sub)) if sub else float("nan")
    return out


# =============================================================================
# 5. Cost
# =============================================================================
@torch.no_grad()
def model_cost(model, input_size=(480, 640), forward_fn=_default_forward, device=None,
               warmup=10, iters=50):
    device = device or next(model.parameters()).device
    model.eval()
    rgb = torch.randn(1, 3, *input_size, device=device)
    th = torch.randn(1, 3, *input_size, device=device)
    res = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    try:
        from fvcore.nn import FlopCountAnalysis

        class _Wrap(torch.nn.Module):
            def __init__(s, m):
                super().__init__()
                s.m = m

            def forward(s, a, b):
                return forward_fn(s.m, a, b)

        fl = FlopCountAnalysis(_Wrap(model), (rgb, th))
        fl.unsupported_ops_warnings(False)
        fl.uncalled_modules_warnings(False)
        res["GFLOPs"] = fl.total() / 1e9
    except Exception as e:
        res["GFLOPs"] = f"n/a ({type(e).__name__})"
    for _ in range(warmup):
        forward_fn(model, rgb, th)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        forward_fn(model, rgb, th)
    if device.type == "cuda":
        torch.cuda.synchronize()
    res["latency_ms"] = (time.perf_counter() - t0) / iters * 1000
    res["FPS"] = 1000 / res["latency_ms"]
    return res


# =============================================================================
# 6. Visual helpers
# =============================================================================
def colorize(label):
    out = np.zeros((*label.shape, 3), np.uint8)
    for i, c in enumerate(PST900_PALETTE):
        out[label == i] = c
    out[label == IGNORE_INDEX] = [128, 128, 128]
    return out


def corruption_grid(root_dir, idx=0, severity=2):
    """Returns an image: rows = clean + each corruption; columns = RGB | thermal."""
    ds = PST900Dataset(root_dir, "test")
    rgb0, th0, _, _ = ds.load_raw(idx)
    rows = []
    for c in ["clean"] + list(CORRUPTIONS):
        r, t = apply_corruption(rgb0, th0, c, severity, np.random.default_rng(idx))
        rows.append(np.concatenate([r, np.stack([t] * 3, -1)], axis=1))
    return np.concatenate(rows, axis=0), ["clean"] + list(CORRUPTIONS)
