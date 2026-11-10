"""
cmrf_engine.py -- CMX / CMRF training engine for PST900 (Team 41)

* CMRF v2 (fixes from the E2 diagnosis):
    1. every parameter is optimised (E2 bug: scale_t / scale_r were never trained)
    2. the cross-modal prediction loss trains ONLY the predictors (encoder inputs are detached),
       so it can no longer distort the pretrained encoders
    3. the residual is normalised (GroupNorm) before the gate; reflect padding removes border artefacts
* One engine for every experiment: E1 protocol (no CMRF), E2-v2, E3, control.
* Resumable: last.pth (model + optimizer + scaler + epoch) is written after every epoch and when
  the run is interrupted; best_model.pth (lowest validation loss); epoch_10/20/... snapshots.
  All writes are atomic (temp file + rename), so a disconnect cannot corrupt a checkpoint.
"""
import os
import re
import json
import time
import glob
import math
import random
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ============================================================================
# 1. CMRF v2 module
# ============================================================================
class _Predictor(nn.Sequential):
    def __init__(self, dim, hidden):
        super().__init__(
            nn.Conv2d(dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='reflect', groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, dim, 1))


class _Gate(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm_res = nn.GroupNorm(1, dim)           # fix 3: scale-free residual magnitude
        self.net = nn.Sequential(
            nn.Conv2d(2 * dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='reflect', groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, dim, 1), nn.Sigmoid())

    def forward(self, res, ctx):
        return self.net(torch.cat([self.norm_res(res.abs()), ctx], 1))


class CMRF(nn.Module):
    """Cross-Modal Residual Filtering, applied at one encoder stage before CMX's CM-FRM."""

    def __init__(self, dim, reduction=4, lam=0.1, init_scale=0.1):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.pred_r2t, self.pred_t2r = _Predictor(dim, hidden), _Predictor(dim, hidden)
        self.gate_t, self.gate_r = _Gate(dim, hidden), _Gate(dim, hidden)
        self.proj_t, self.proj_r = nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 1)
        self.scale_t = nn.Parameter(torch.full((1,), float(init_scale)))
        self.scale_r = nn.Parameter(torch.full((1,), float(init_scale)))
        self.lam = lam
        self.aux_loss = None
        self.last = {}

    @staticmethod
    def _nmse(pred, target):
        target = target.detach()
        return ((pred.float() - target.float()) ** 2).mean() / (target.float().pow(2).mean() + 1e-6)

    def forward(self, x_rgb, x_thm):
        t_hat, r_hat = self.pred_r2t(x_rgb), self.pred_t2r(x_thm)
        res_t, res_r = x_thm - t_hat, x_rgb - r_hat          # what the other camera cannot explain
        g_t, g_r = self.gate_t(res_t, x_rgb), self.gate_r(res_r, x_thm)
        out_rgb = x_rgb + self.scale_t * g_t * self.proj_t(res_t)
        out_thm = x_thm + self.scale_r * g_r * self.proj_r(res_r)
        if self.training and self.lam > 0:
            # fix 2: predictors learn from detached encoder features -> loss cannot reshape the encoders
            self.aux_loss = self.lam * (self._nmse(self.pred_r2t(x_rgb.detach()), x_thm) +
                                        self._nmse(self.pred_t2r(x_thm.detach()), x_rgb))
            self.last = {}
        else:
            self.aux_loss = None
            if not self.training:
                self.last = {'gate_t': g_t.mean(1).detach().float(), 'gate_r': g_r.mean(1).detach().float(),
                             'res_t': res_t.abs().mean(1).detach().float(), 'res_r': res_r.abs().mean(1).detach().float()}
        return out_rgb, out_thm


def patch_cmx(repo_dir):
    """Insert CMRF into the official CMX encoder (idempotent). Stages come from <repo>/cmrf_config.json."""
    src = open(osp.abspath(__file__)).read()
    open(osp.join(repo_dir, 'models', 'cmrf_v2.py'), 'w').write(src)
    p = osp.join(repo_dir, 'models', 'encoders', 'dual_segformer.py')
    s = open(p).read()
    if '[CMRFv2]' not in s:
        assert '[CMRF]' not in s, 'This repo copy has the old E2 patch: delete the repo folder and re-run the setup cell.'
        s = s.replace('import math\n', 'import math\nfrom ..cmrf_v2 import CMRF, load_cmrf_config  # [CMRFv2]\n', 1)
        anchor = 'norm_layer=norm_fuse)])\n\n        self.apply(self._init_weights)\n'
        assert s.count(anchor) == 1, 'unexpected dual_segformer.py layout'
        s = s.replace(anchor, (
            'norm_layer=norm_fuse)])\n\n'
            "        _cc = load_cmrf_config()  # [CMRFv2]\n"
            "        self.CMRFs = nn.ModuleDict({str(k): CMRF(embed_dims[k - 1], reduction=_cc.get('reduction', 4),\n"
            "                                                lam=_cc.get('lambda', 0.1), init_scale=_cc.get('init_scale', 0.1))\n"
            "                                    for k in _cc.get('stages', [])})\n"
            "        self.apply(self._init_weights)\n"), 1)
        for k in range(4):
            call = f'        x_rgb, x_e = self.FRMs[{k}](x_rgb, x_e)\n'
            assert s.count(call) == 1
            s = s.replace(call, (f"        if '{k + 1}' in self.CMRFs:  # [CMRFv2]\n"
                                 f"            x_rgb, x_e = self.CMRFs['{k + 1}'](x_rgb, x_e)\n" + call), 1)
        s = s.replace('    def init_weights(self, pretrained=None):\n', (
            "    def cmrf_aux_loss(self):  # [CMRFv2]\n"
            "        ls = [m.aux_loss for m in self.CMRFs.values() if m.aux_loss is not None]\n"
            "        return sum(ls) if ls else None\n\n"
            "    def init_weights(self, pretrained=None):\n"), 1)
        open(p, 'w').write(s)


def load_cmrf_config():
    p = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'cmrf_config.json')
    return json.load(open(p)) if osp.exists(p) else {'stages': []}


def write_cmrf_config(repo_dir, stages, reduction=4, lam=0.1, init_scale=0.1, **_):
    cfg = {'stages': list(stages), 'reduction': reduction, 'lambda': lam, 'init_scale': init_scale}
    json.dump(cfg, open(osp.join(repo_dir, 'cmrf_config.json'), 'w'), indent=2)
    return cfg


# ============================================================================
# 2. Model
# ============================================================================
def build_model(repo_dir, backbone='mit_b2', num_classes=5, pretrained=None, cmrf=None):
    """CMX EncoderDecoder from the official repo. cmrf: dict(stages=..., ...) written to cmrf_config.json first."""
    if cmrf is not None:
        write_cmrf_config(repo_dir, **cmrf)
    import sys
    from types import SimpleNamespace
    repo_dir = osp.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    for m in [k for k in list(sys.modules) if k == 'models' or k.startswith('models.')]:
        del sys.modules[m]                                    # pick up the current cmrf_config.json
    cwd = os.getcwd(); os.chdir(repo_dir)
    try:
        from models.builder import EncoderDecoder
        cfg = SimpleNamespace(backbone=backbone, pretrained_model=pretrained, decoder='MLPDecoder',
                              decoder_embed_dim=512, num_classes=num_classes, bn_eps=1e-3, bn_momentum=0.1)
        # a criterion must be passed: without one the repo skips init_weights (no ImageNet weights, no decoder init)
        model = EncoderDecoder(cfg=cfg, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
                               norm_layer=nn.BatchNorm2d)
    finally:
        os.chdir(cwd)
    if pretrained:   # the repo loads with strict=False, which hides failures: check it really loaded
        raw = torch.load(pretrained, map_location='cpu', weights_only=False)
        raw = raw.get('model', raw.get('state_dict', raw))
        k = 'patch_embed1.proj.weight'
        assert torch.equal(raw[k].float(), model.backbone.patch_embed1.proj.weight.detach().cpu().float()) and \
               torch.equal(raw[k].float(), model.backbone.extra_patch_embed1.proj.weight.detach().cpu().float()), \
               f'ImageNet weights were NOT loaded from {pretrained}'
        del raw
    return model


def param_groups(model, lr, weight_decay):
    """Same split as the CMX repo (decay on conv/linear weights), but EVERY parameter is included (fix 1)."""
    decay, no_decay, seen = [], [], set()
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            decay.append(m.weight); seen.add(id(m.weight))
            if m.bias is not None:
                no_decay.append(m.bias); seen.add(id(m.bias))
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for p in (m.weight, m.bias):
                if p is not None:
                    no_decay.append(p); seen.add(id(p))
    rest = [p for p in model.parameters() if id(p) not in seen]
    no_decay += rest
    return [dict(params=decay, lr=lr, weight_decay=weight_decay),
            dict(params=no_decay, lr=lr, weight_decay=0.0)], len(rest)


def load_weights(model, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck.get('model', ck))
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return ck


# ============================================================================
# 3. Validation split (held out from TRAIN, fixed forever, test set untouched)
# ============================================================================
def make_val_split(pst_root, out_json, frac=0.15, seed_tries=200):
    """Split the 597 training images by recording ('bagNN') so near-identical frames never leak."""
    if osp.exists(out_json):
        return json.load(open(out_json))
    import cv2
    names = sorted(osp.splitext(osp.basename(f))[0] for f in glob.glob(osp.join(pst_root, 'train', 'rgb', '*.png')))
    rec = lambda n: (re.search(r'bag\d+', n).group(0) if re.search(r'bag\d+', n) else n)
    present = {n: set(np.unique(cv2.imread(osp.join(pst_root, 'train', 'labels', n + '.png'), cv2.IMREAD_GRAYSCALE)).tolist())
               for n in names}
    groups = sorted(set(map(rec, names)))
    by_frame = len(groups) < 5
    for seed in range(seed_tries):
        rng = random.Random(seed)
        if by_frame:
            val = set(rng.sample(names, max(1, round(frac * len(names)))))
        else:
            g = groups[:]; rng.shuffle(g); val = set()
            for grp in g:
                if len(val) >= frac * len(names): break
                val |= {n for n in names if rec(n) == grp}
        if {1, 2, 3, 4} <= set().union(*(present[n] for n in val)) and 0.08 <= len(val) / len(names) <= 0.25:
            break
    split = {'val': sorted(val), 'train': [n for n in names if n not in val], 'seed': seed,
             'by': 'frame' if by_frame else 'recording',
             'val_recordings': [] if by_frame else sorted(set(map(rec, val)))}
    os.makedirs(osp.dirname(out_json), exist_ok=True)
    json.dump(split, open(out_json, 'w'), indent=1)
    return split


# ============================================================================
# 4. Training
# ============================================================================
def save_atomic(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)
    try:
        os.sync()          # push the file to Google Drive now, not later
    except Exception:
        pass


@torch.no_grad()
def validate(model, loader, device, size=(480, 640), use_amp=True):
    """Fast validation: whole image resized to 480x640 (labels resized with nearest)."""
    model.eval()
    tot, npx = 0.0, 0
    inter = np.zeros(5); union = np.zeros(5)
    for b in loader:
        rgb = F.interpolate(b['rgb'].to(device), size=size, mode='bilinear', align_corners=False)
        th = F.interpolate(b['thermal'].to(device), size=size, mode='bilinear', align_corners=False)
        y = F.interpolate(b['label'][:, None].float(), size=size, mode='nearest')[:, 0].long().to(device)
        with torch.autocast(device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16,
                            enabled=use_amp):
            out = model(rgb, th)
        out = out.float()
        loss = F.cross_entropy(out, y, ignore_index=255, reduction='sum')
        tot += loss.item(); npx += int((y != 255).sum())
        pred = out.argmax(1); keep = y != 255
        for c in range(5):
            p, g = (pred == c) & keep, (y == c) & keep
            inter[c] += int((p & g).sum()); union[c] += int((p | g).sum())
    model.train()
    iou = inter / np.maximum(union, 1)
    return tot / max(npx, 1), float(iou.mean() * 100)


def train(run):
    """run: dict loaded from run_config.json (written by the notebook)."""
    import sys
    sys.path.insert(0, run['toolkit_dir'])
    from rgbt_eval import PST900Dataset, PST900PairDataset

    torch.manual_seed(run['seed']); np.random.seed(run['seed']); random.seed(run['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = run['amp'] and device.type == 'cuda'
    ck_dir = run['ckpt_dir']; os.makedirs(ck_dir, exist_ok=True)
    last_path, best_path = osp.join(ck_dir, 'last.pth'), osp.join(ck_dir, 'best_model.pth')
    log_csv = osp.join(ck_dir, 'train_log.csv')
    if osp.exists(osp.join(ck_dir, 'DONE')):
        print(f"[{run['name']}] training already complete (DONE marker found)."); return

    # ---- data ----
    split = json.load(open(run['split_json']))
    beta = run.get('consistency', 0.0)            # E4: clean-to-corrupted consistency weight
    DS = PST900PairDataset if beta > 0 else PST900Dataset
    full_train = DS(run['pst_root'], 'train', is_training=True, crop_size=tuple(run['crop']),
                    train_corruption_p=0.0 if beta > 0 else run['corruption_p'], channel_order='BGR')
    stems = [osp.splitext(osp.basename(f))[0] for f in full_train.rgb_files]
    use_val = run['val_mode'] == 'holdout'
    tr_idx = [i for i, s in enumerate(stems) if (s in set(split['train']) or not use_val)]
    train_ds = Subset(full_train, tr_idx)
    if use_val:
        val_base = PST900Dataset(run['pst_root'], 'train', is_training=False, channel_order='BGR')
        val_ds = Subset(val_base, [i for i, s in enumerate(stems) if s in set(split['val'])])
    else:   # 'test' (not recommended: selects the model on the test set)
        val_ds = PST900Dataset(run['pst_root'], 'test', is_training=False, channel_order='BGR')
    g = torch.Generator(); g.manual_seed(run['seed'])
    train_loader = DataLoader(train_ds, batch_size=run['batch_size'], shuffle=True, drop_last=True, generator=g,
                              num_workers=run['num_workers'], pin_memory=device.type == 'cuda', persistent_workers=run['num_workers'] > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=min(2, run['num_workers']))
    ipe = len(train_loader)

    # ---- model / optimiser / schedule (CMX defaults: AdamW, poly 0.9, 10 warm-up epochs) ----
    model = build_model(run['repo'], backbone=run.get('backbone', 'mit_b2'), pretrained=run['pretrained'],
                        cmrf=run.get('cmrf')).to(device)
    groups, n_extra = param_groups(model, run['lr'], run['weight_decay'])
    opt = torch.optim.AdamW(groups, lr=run['lr'], betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    total_it, warm_it = run['epochs'] * ipe, run['warmup_epochs'] * ipe
    lr_at = lambda it: run['lr'] * (it / warm_it) if it < warm_it else run['lr'] * (1 - it / total_it) ** 0.9
    crit = nn.CrossEntropyLoss(ignore_index=255)

    start_ep, best_val, best_ep = 1, float('inf'), 0
    if osp.exists(last_path):
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        if ck.get('run_name') != run['name']:
            raise RuntimeError(f"last.pth belongs to run '{ck.get('run_name')}', not '{run['name']}'. Use another EXP name.")
        model.load_state_dict(ck['state_dict']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start_ep, best_val, best_ep = ck['epoch'] + 1, ck['best_val_loss'], ck['best_epoch']
        g.manual_seed(run['seed'] + ck['epoch'])               # different shuffle order after a resume
        print(f"[Resume] from {last_path}: continuing at epoch {start_ep} (best val loss {best_val:.4f} @ epoch {best_ep})")
        del ck
    else:
        print('[Fresh start] training from epoch 1')
    cmrf = getattr(model.backbone, 'CMRFs', {})
    print(f"[Run] {run['name']} | CMRF stages {list(cmrf.keys()) if len(cmrf) else 'none (plain CMX)'} | "
          f"corruption p={run['corruption_p']} | train {len(train_ds)} | val {len(val_ds)} ({run['val_mode']}) | "
          f"consistency {beta} | {ipe} it/epoch | AMP {use_amp} | extra params added to optimizer: {n_extra}")
    if start_ep > run['epochs']:
        print(f"Training already complete ({run['epochs']} epochs).")
        open(osp.join(ck_dir, 'DONE'), 'w').write(str(best_ep)); return

    def checkpoint(epoch_done, tr_loss=None, vl=None):
        save_atomic({'epoch': epoch_done, 'state_dict': model.state_dict(), 'optimizer': opt.state_dict(),
                     'scaler': scaler.state_dict(), 'best_val_loss': best_val, 'best_epoch': best_ep,
                     'run_name': run['name'], 'run': run}, last_path)

    model.train()
    ep = start_ep
    try:
        for ep in range(start_ep, run['epochs'] + 1):
            t0, tot, aux_tot = time.time(), 0.0, 0.0
            for i, b in enumerate(train_loader):
                it = (ep - 1) * ipe + i
                for gr in opt.param_groups: gr['lr'] = lr_at(it)
                rgb, th, y = b['rgb'].to(device, non_blocking=True), b['thermal'].to(device, non_blocking=True), b['label'].to(device, non_blocking=True)
                if beta > 0:   # E4: one forward pass over [clean ; corrupted] views of the same crops
                    rgb = torch.cat([rgb, b['rgb_c'].to(device, non_blocking=True)])
                    th = torch.cat([th, b['thermal_c'].to(device, non_blocking=True)])
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    out = model(rgb, th)
                    if beta > 0:
                        o_clean, o_corr = out.float().chunk(2)
                        loss = 0.5 * (crit(o_clean, y) + crit(o_corr, y))
                        keep = (y != 255).float()
                        kl = (F.softmax(o_clean.detach(), 1) * (F.log_softmax(o_clean.detach(), 1) - F.log_softmax(o_corr, 1))).sum(1)
                        cons = (kl * keep).sum() / keep.sum().clamp(min=1)
                    else:
                        loss = crit(out.float(), y)
                        cons = None
                    aux = model.backbone.cmrf_aux_loss() if hasattr(model.backbone, 'cmrf_aux_loss') else None
                    total = loss + (aux if aux is not None else 0) + (beta * cons if cons is not None else 0)
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward(); scaler.step(opt); scaler.update()
                tot += loss.item(); aux_tot += aux.item() if aux is not None else 0.0
                if i % 25 == 0 or i == ipe - 1:
                    print(f"\rEpoch {ep:3d}/{run['epochs']} it {i + 1:3d}/{ipe} | CE {tot / (i + 1):.4f} | aux {aux_tot / (i + 1):.4f} | lr {lr_at(it):.2e}", end='', flush=True)
            tr_loss = tot / ipe
            vl, vmiou = validate(model, val_loader, device, size=tuple(run.get('val_size', (480, 640))), use_amp=use_amp)
            is_best = vl < best_val
            if is_best:
                best_val, best_ep = vl, ep
                save_atomic({'epoch': ep, 'state_dict': model.state_dict(), 'val_loss': vl, 'val_mIoU': vmiou,
                             'run_name': run['name']}, best_path)
            if ep % run['snapshot_every'] == 0 or ep == run['epochs']:
                sd = model.state_dict()
                if run.get('snapshot_fp16', False):   # half the Drive space; load_weights casts back to float32
                    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
                save_atomic({'epoch': ep, 'state_dict': sd, 'val_loss': vl, 'run_name': run['name']},
                            osp.join(ck_dir, f"epoch_{ep}.pth"))
            checkpoint(ep)
            sc = {k: round(v.scale_t.item(), 4) for k, v in cmrf.items()} if len(cmrf) else {}
            row = {'epoch': ep, 'train_ce': tr_loss, 'aux': aux_tot / ipe, 'val_loss': vl, 'val_mIoU': vmiou,
                   'lr': lr_at(it), 'minutes': (time.time() - t0) / 60, 'best': is_best, 'scale_t': json.dumps(sc)}
            new = not osp.exists(log_csv)
            with open(log_csv, 'a') as f:
                if new: f.write(','.join(row) + '\n')
                f.write(','.join(str(v).replace(',', ';') for v in row.values()) + '\n')
            print(f"\n   train CE {tr_loss:.4f} | val loss {vl:.4f} | val mIoU {vmiou:.2f} | {row['minutes']:.1f} min"
                  + (' | [*] new best' if is_best else '') + (f' | scale_t {sc}' if sc else '')
                  + (f" | [+] epoch_{ep}.pth" if ep % run['snapshot_every'] == 0 else ''))
    except KeyboardInterrupt:
        # stopped by the user: keep everything up to the last COMPLETED epoch (last.pth is already saved),
        # nothing half-written is left behind because all saves are atomic
        print(f"\n[Stopped] Interrupted during epoch {ep}. Resume restarts epoch {ep} from last.pth (epoch {ep - 1} completed).")
        return
    open(osp.join(ck_dir, 'DONE'), 'w').write(str(best_ep))
    if not run.get('keep_last_when_done', True) and osp.exists(last_path):
        os.remove(last_path)                      # optimizer state is not needed once training is finished
    print(f"\nTRAINING COMPLETE | best val loss {best_val:.4f} at epoch {best_ep}")


# ============================================================================
# 5. Evaluation of a finished run (test set; same protocol as E1/E2)
# ============================================================================
def evaluate_run(run, ckpt='best_model.pth', robustness=True, force=False, show=False):
    """Clean + robustness benchmark, modality drop, cost, figures. Results in <ckpt_dir>/results/.
    Skips the work if results/summary.json already exists (force=True re-runs it)."""
    import sys
    sys.path.insert(0, run['toolkit_dir'])
    import matplotlib
    import matplotlib.pyplot as plt
    from rgbt_eval import (PST900Dataset, PST900_CLASSES, run_benchmark, model_cost, predict_probs, colorize)

    res = osp.join(run['ckpt_dir'], 'results'); os.makedirs(res, exist_ok=True)
    summ_path = osp.join(res, 'summary.json')
    if osp.exists(summ_path) and not force:
        s = json.load(open(summ_path))
        if s.get('robustness') or not robustness:
            print(f"[{run['name']}] already evaluated -> {summ_path}")
            return s
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(run['repo'], backbone=run.get('backbone', 'mit_b2'), cmrf=run.get('cmrf'))
    ck = load_weights(model, osp.join(run['ckpt_dir'], ckpt))
    model = model.to(device).eval()
    name, crop = run['name'], tuple(run['crop'])
    print(f"[{name}] evaluating {ckpt} (epoch {ck.get('epoch')})")

    bench = osp.join(res, 'benchmark.csv')
    if osp.exists(bench): os.remove(bench)
    rows = run_benchmark(model, run['pst_root'], corruptions=None if robustness else [], crop=crop, stride_rate=2 / 3,
                         flip=False, channel_order='BGR', out_csv=bench, tag=name)
    clean = rows[0]
    import pandas as pd
    pc = pd.DataFrame({m: [clean[f'{m}/{c}'] * 100 for c in PST900_CLASSES]
                       for m in ('IoU', 'Dice', 'Sensitivity', 'Specificity')}, index=PST900_CLASSES)
    pc.loc['mean (all 5)'] = [clean[k] * 100 for k in ('mIoU', 'mDice', 'mSensitivity', 'mSpecificity')]
    pc.loc['mean (4 objects)'] = [clean[k] * 100 for k in ('mIoU_obj', 'mDice_obj', 'mSensitivity_obj', 'mSpecificity_obj')]
    pc.to_csv(osp.join(res, 'per_class_clean.csv'))

    S = {'name': name, 'ckpt': ckpt, 'epoch': ck.get('epoch'), 'robustness': bool(robustness),
         'cmrf_stages': (run.get('cmrf') or {}).get('stages', []), 'corruption_p': run.get('corruption_p', 0.0),
         'consistency': run.get('consistency', 0.0)}
    for k in ('mIoU', 'mIoU_obj', 'mDice', 'mSensitivity', 'mSpecificity'):
        S[f'clean_{k}'] = clean[k] * 100
    for c in PST900_CLASSES:
        S[f'IoU_{c}'] = clean[f'IoU/{c}'] * 100
    if robustness:
        df = pd.DataFrame(rows)
        rob = df[df.condition != 'clean'].pivot(index='condition', columns='severity', values='mIoU').mul(100)
        rob['avg'] = rob.mean(1)
        rob.to_csv(osp.join(res, 'robustness_mIoU.csv'))
        robo = df[df.condition != 'clean'].pivot(index='condition', columns='severity', values='mIoU_obj').mul(100)
        robo['avg'] = robo.mean(1)
        robo.to_csv(osp.join(res, 'robustness_mIoU_objects.csv'))
        S['rgb_degraded_avg'] = rob[rob.index.str.startswith('rgb')]['avg'].mean()
        S['thermal_degraded_avg'] = rob[rob.index.str.startswith('thermal')]['avg'].mean()
        S['rgb_degraded_avg_obj'] = robo[robo.index.str.startswith('rgb')]['avg'].mean()
        S['thermal_degraded_avg_obj'] = robo[robo.index.str.startswith('thermal')]['avg'].mean()
        for cond in rob.index:
            S[f'{cond}_avg'] = rob.loc[cond, 'avg']
        S['rgb_low_light_s3'] = rob.loc['rgb_low_light', 3]
        S['rgb_low_light_s3_obj'] = robo.loc['rgb_low_light', 3]

    # modality drop: how much does the model rely on each camera?
    for tag, fwd in [('rgb_only', lambda m, r, t: m(r, torch.zeros_like(t))),
                     ('thermal_only', lambda m, r, t: m(torch.zeros_like(r), t))]:
        r0 = run_benchmark(model, run['pst_root'], forward_fn=fwd, corruptions=[], crop=crop, channel_order='BGR',
                           verbose=False)[0]
        S[f'{tag}_mIoU'], S[f'{tag}_mIoU_obj'] = r0['mIoU'] * 100, r0['mIoU_obj'] * 100
        S[f'{tag}_IoU_Survivor'] = r0['IoU/Survivor'] * 100

    cost = model_cost(model, input_size=crop)
    S.update({'params_M': cost.get('params_M'), 'GFLOPs': cost.get('GFLOPs'), 'latency_ms': cost.get('latency_ms')})

    # figures: predictions under degradation (+ CMRF residual / gate maps)
    n_test = len(PST900Dataset(run['pst_root'], 'test'))
    idxs = [min(i, n_test - 1) for i in (3, 100, 200)]
    SHOW = [('clean', 1), ('rgb_low_light', 3), ('rgb_occlusion', 3), ('thermal_occlusion', 3)]
    fig, ax = plt.subplots(len(idxs), 2 + len(SHOW), figsize=(4 * (2 + len(SHOW)), 3 * len(idxs)))
    for r, i in enumerate(idxs):
        for c, (cn, sev) in enumerate(SHOW):
            d = PST900Dataset(run['pst_root'], 'test', corruption=cn, severity=sev, channel_order='BGR')
            b = d[i]
            with torch.no_grad():
                p = predict_probs(model, b['rgb'][None].to(device), b['thermal'][None].to(device), crop=crop).argmax(1)[0].cpu().numpy()
            if c == 0:
                raw = d.load_raw(i)
                ax[r, 0].imshow(raw[0]); ax[r, 0].set_title(raw[3][:30])
                ax[r, 1].imshow(colorize(raw[2])); ax[r, 1].set_title('ground truth')
            ax[r, 2 + c].imshow(colorize(p)); ax[r, 2 + c].set_title(f'{cn} s{sev}')
    for a in ax.flat: a.axis('off')
    fig.suptitle(name); plt.tight_layout(); fig.savefig(osp.join(res, 'qualitative.png'), dpi=100)
    plt.show() if show else plt.close(fig)

    cms = getattr(model.backbone, 'CMRFs', {})
    if len(cms):
        cm = cms[str(S['cmrf_stages'][0])]
        S['scale_t'], S['scale_r'] = cm.scale_t.item(), cm.scale_r.item()
        stats = {'clean': [], 'rgb_low_light': []}
        fig, ax = plt.subplots(2, 5, figsize=(22, 6.5))
        for cn, sev in (('clean', 1), ('rgb_low_light', 3)):
            d = PST900Dataset(run['pst_root'], 'test', corruption=cn, severity=sev, channel_order='BGR')
            for i in sorted(set(range(0, n_test, max(1, n_test // 24))) | set(idxs[:2])):
                b = d[i]
                rgb = F.interpolate(b['rgb'][None], size=crop, mode='bilinear', align_corners=False).to(device)
                th = F.interpolate(b['thermal'][None], size=crop, mode='bilinear', align_corners=False).to(device)
                with torch.no_grad():
                    model(rgb, th)
                g = cm.last['gate_t'][0]
                stats[cn].append((g.mean().item(), g.std().item()))
                if i in idxs[:2]:
                    r = idxs.index(i); c = 0 if cn == 'clean' else 1
                    ax[r, 1 + 2 * c].imshow(cm.last['res_t'][0].cpu(), cmap='magma'); ax[r, 1 + 2 * c].set_title(f'thermal residual ({cn})')
                    ax[r, 2 + 2 * c].imshow(g.cpu(), cmap='viridis', vmin=0, vmax=1)
                    ax[r, 2 + 2 * c].set_title(f'gate ({cn}) mean {g.mean():.2f} std {g.std():.3f}')
                    if cn == 'clean':
                        ax[r, 0].imshow(d.load_raw(i)[0]); ax[r, 0].set_title('RGB (clean)')
        for a in ax.flat: a.axis('off')
        fig.suptitle(name); plt.tight_layout(); fig.savefig(osp.join(res, 'cmrf_maps.png'), dpi=100)
        plt.show() if show else plt.close(fig)
        for cn, v in stats.items():
            S[f'gate_mean_{cn}'] = float(np.mean([a for a, _ in v]))
            S[f'gate_spatial_std_{cn}'] = float(np.mean([b for _, b in v]))

    json.dump(S, open(summ_path, 'w'), indent=1)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return S


if __name__ == '__main__':
    import sys
    train(json.load(open(sys.argv[1])))
