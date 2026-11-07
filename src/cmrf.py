"""
CMRF: Cross-Modal Residual Filtering (E2 hypothesis module) for CMX.

At one encoder stage, before CMX's CM-FRM:
  1. Predict each modality's features from the other   T_hat = P_r2t(F_rgb),  R_hat = P_t2r(F_thm)
  2. Residual = what the other modality cannot explain  Res_t = F_thm - T_hat,  Res_r = F_rgb - R_hat
  3. Gate (per pixel, per channel), from |residual| and the receiving stream
  4. Inject the gated residual into the OTHER stream     F_rgb' = F_rgb + s_t * G_t * proj(Res_t)
                                                         F_thm' = F_thm + s_r * G_r * proj(Res_r)
Auxiliary loss (training only): normalised MSE between predictions and stop-gradient targets,
so the predictors learn real cross-modal correspondence and features cannot collapse.
"""
import json
import os.path as osp

import torch
import torch.nn as nn


def load_cmrf_config():
    """Reads <repo>/cmrf_config.json. Missing file -> CMRF disabled (plain CMX = E1)."""
    p = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'cmrf_config.json')
    if not osp.exists(p):
        return {'stages': []}
    with open(p) as f:
        return json.load(f)


class _Predictor(nn.Sequential):
    """Lightweight cross-modal predictor: 1x1 -> depthwise 3x3 -> 1x1."""
    def __init__(self, dim, hidden):
        super().__init__(
            nn.Conv2d(dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, dim, 1))


class _Gate(nn.Sequential):
    """Per-pixel, per-channel gate in [0, 1] from [|residual|, receiving features]."""
    def __init__(self, dim, hidden):
        super().__init__(
            nn.Conv2d(2 * dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, dim, 1), nn.Sigmoid())


class CMRF(nn.Module):
    def __init__(self, dim, reduction=4, lam=0.1, init_scale=0.1):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.pred_r2t = _Predictor(dim, hidden)   # RGB -> thermal
        self.pred_t2r = _Predictor(dim, hidden)   # thermal -> RGB
        self.gate_t = _Gate(dim, hidden)          # how much thermal-unique info enters the RGB stream
        self.gate_r = _Gate(dim, hidden)          # how much RGB-unique info enters the thermal stream
        self.proj_t = nn.Conv2d(dim, dim, 1)
        self.proj_r = nn.Conv2d(dim, dim, 1)
        self.scale_t = nn.Parameter(torch.full((1,), float(init_scale)))
        self.scale_r = nn.Parameter(torch.full((1,), float(init_scale)))
        self.lam = lam
        self.aux_loss = None
        self.last = {}                            # filled in eval mode, for visualisation

    @staticmethod
    def _nmse(pred, target):
        target = target.detach()                  # stop-gradient on the target
        return ((pred - target) ** 2).mean() / (target.pow(2).mean() + 1e-6)

    def forward(self, x_rgb, x_thm):
        t_hat = self.pred_r2t(x_rgb)
        r_hat = self.pred_t2r(x_thm)
        res_t = x_thm - t_hat                     # thermal information RGB cannot explain
        res_r = x_rgb - r_hat                     # RGB information thermal cannot explain
        g_t = self.gate_t(torch.cat([res_t.abs(), x_rgb], 1))
        g_r = self.gate_r(torch.cat([res_r.abs(), x_thm], 1))
        out_rgb = x_rgb + self.scale_t * g_t * self.proj_t(res_t)
        out_thm = x_thm + self.scale_r * g_r * self.proj_r(res_r)

        if self.training:
            self.aux_loss = self.lam * (self._nmse(t_hat, x_thm) + self._nmse(r_hat, x_rgb))
            self.last = {}
        else:
            self.aux_loss = None
            self.last = {'gate_t': g_t.mean(1).detach(), 'gate_r': g_r.mean(1).detach(),
                         'res_t': res_t.abs().mean(1).detach(), 'res_r': res_r.abs().mean(1).detach()}
        return out_rgb, out_thm


def patch_cmx_for_cmrf(repo_dir):
    """Adds CMRF to the official CMX code. Idempotent. Stages come from cmrf_config.json."""
    import shutil
    shutil.copy(osp.abspath(__file__), osp.join(repo_dir, 'models', 'cmrf.py'))

    p = osp.join(repo_dir, 'models', 'encoders', 'dual_segformer.py')
    s = open(p).read()
    if '[CMRF]' not in s:
        s = s.replace('import math\n', 'import math\nfrom ..cmrf import CMRF, load_cmrf_config  # [CMRF]\n', 1)
        anchor = ('norm_layer=norm_fuse)])\n\n'
                  '        self.apply(self._init_weights)\n')    # end of the FFM list in RGBXTransformer
        assert s.count(anchor) == 1, 'unexpected dual_segformer.py layout'
        s = s.replace(anchor, ('norm_layer=norm_fuse)])\n\n' +
            "        # [CMRF] cross-modal residual filtering before CM-FRM at the configured stages\n"
            "        _cc = load_cmrf_config()\n"
            "        self.cmrf_stages = [int(k) for k in _cc.get('stages', [])]\n"
            "        self.CMRFs = nn.ModuleDict({str(k): CMRF(embed_dims[k - 1], reduction=_cc.get('reduction', 4),\n"
            "                                                lam=_cc.get('lambda', 0.1), init_scale=_cc.get('init_scale', 0.1))\n"
            "                                    for k in self.cmrf_stages})\n"
            "        self.apply(self._init_weights)\n"), 1)
        for k in range(4):
            call = f'        x_rgb, x_e = self.FRMs[{k}](x_rgb, x_e)\n'
            assert s.count(call) == 1, f'FRM call for stage {k + 1} not found'
            s = s.replace(call, (f"        if '{k + 1}' in self.CMRFs:  # [CMRF]\n"
                                 f"            x_rgb, x_e = self.CMRFs['{k + 1}'](x_rgb, x_e)\n" + call), 1)
        s = s.replace('    def init_weights(self, pretrained=None):\n', (
            "    def cmrf_aux_loss(self):  # [CMRF]\n"
            "        ls = [m.aux_loss for m in self.CMRFs.values() if m.aux_loss is not None]\n"
            "        return sum(ls) if ls else None\n\n"
            "    def init_weights(self, pretrained=None):\n"), 1)
        open(p, 'w').write(s)

    p = osp.join(repo_dir, 'models', 'builder.py')
    s = open(p).read()
    if '[CMRF]' not in s:
        old = '            loss = self.criterion(out, label.long())\n'
        assert s.count(old) == 1, 'unexpected builder.py layout'
        s = s.replace(old, old + (
            "            _aux = self.backbone.cmrf_aux_loss() if hasattr(self.backbone, 'cmrf_aux_loss') else None  # [CMRF]\n"
            "            if _aux is not None:\n"
            "                loss = loss + _aux\n"), 1)
        open(p, 'w').write(s)
    print('CMX patched with CMRF (stages read from cmrf_config.json)')


def write_cmrf_config(repo_dir, stages=(3,), reduction=4, lam=0.1, init_scale=0.1):
    cfg = {'stages': list(stages), 'reduction': reduction, 'lambda': lam, 'init_scale': init_scale}
    with open(osp.join(repo_dir, 'cmrf_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print('cmrf_config.json:', cfg)
    return cfg
