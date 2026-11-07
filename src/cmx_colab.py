"""
cmx_colab.py -- helpers to run the OFFICIAL CMX repo (huaaaliu/RGBX_Semantic_Segmentation)
on PST900 in Google Colab (single GPU, resumable from Google Drive).
"""

import os
import re
import sys
import glob
import shutil

PST900_CLASSES = ["Background", "Fire-Extinguisher", "Backpack", "Hand-Drill", "Survivor"]


# -----------------------------------------------------------------------------
# 1. Patch the repo so it runs on modern PyTorch / single-GPU Colab.
#    These are bug fixes only; the model, loss, optimizer and LR schedule are untouched.
# -----------------------------------------------------------------------------
def patch_repo(repo_dir):
    def edit(rel, pairs):
        p = os.path.join(repo_dir, rel)
        src = open(p).read()
        for old, new in pairs:
            if new in src:
                continue                      # already patched
            if old not in src:
                raise RuntimeError(f"Patch target not found in {rel}: {old!r}")
            src = src.replace(old, new)
        open(p, "w").write(src)

    edit("train.py", [
        # Python-3 iterators have no .next()
        ("minibatch = dataloader.next()", "minibatch = next(dataloader)"),
        # `sum_loss += loss` keeps every iteration's autograd graph alive -> GPU memory leak
        ("                sum_loss += loss\n",
         "                sum_loss += loss.item()\n"),
        ("' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))",
         "' loss=%.4f total_loss=%.4f' % (loss.item(), (sum_loss / (idx + 1)))"),
    ])
    edit("engine/engine.py", [
        # resuming a NON-distributed run must not prefix keys with 'module.'
        ("load_model(self.state.model, tmp['model'], is_restore=True)",
         "load_model(self.state.model, tmp['model'], is_restore=self.distributed)"),
        ("tmp = torch.load(self.continue_state_object)\n",
         "tmp = torch.load(self.continue_state_object, map_location='cpu', weights_only=False)\n"),
    ])
    edit("engine/engine.py", [
        # each checkpoint is ~800 MB (model + AdamW state): keep only the newest 2 on Drive
        ("        self.save_checkpoint(current_epoch_checkpoint)\n",
         "        self.save_checkpoint(current_epoch_checkpoint)\n"
         "        import glob as _g, re as _re\n"
         "        _old = sorted(_g.glob(osp.join(checkpoint_dir, 'epoch-*.pth')),\n"
         "                      key=lambda p: int(_re.search(r'epoch-(\\d+)', p).group(1))\n"
         "                      if _re.search(r'epoch-(\\d+)\\.pth$', p) else -1)\n"
         "        for _p in [p for p in _old if _re.search(r'epoch-(\\d+)\\.pth$', p)][:-2]:\n"
         "            os.remove(_p)\n"),
    ])
    edit("utils/transforms.py", [
        # removed in Python 3.10
        ("isinstance(shape, collections.Iterable)", "isinstance(shape, collections.abc.Iterable)"),
        ("import collections\n", "import collections\nimport collections.abc\n"),
    ])
    print("Repo patched: next(), loss-graph leak, single-GPU resume, torch.load, collections.abc.")


# -----------------------------------------------------------------------------
# 2. Convert PST900 to the CMX folder layout (symlinks, no copying).
#    <dst>/RGB, <dst>/Thermal, <dst>/Label, train.txt, test.txt
#    Names are prefixed with the split so train/test files can never collide.
# -----------------------------------------------------------------------------
def prepare_pst900_for_cmx(src_root, dst):
    split_root = src_root
    if not os.path.isdir(os.path.join(split_root, "train")):
        split_root = os.path.join(src_root, "PST900_RGBT_Dataset")
    for sub in ("RGB", "Thermal", "Label"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    counts = {}
    for split in ("train", "test"):
        names = []
        rgb_files = sorted(glob.glob(os.path.join(split_root, split, "rgb", "*.png")))
        for f in rgb_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            new = f"{split}_{stem}"
            pairs = {"RGB": f,
                     "Thermal": os.path.join(split_root, split, "thermal", stem + ".png"),
                     "Label": os.path.join(split_root, split, "labels", stem + ".png")}
            for sub, srcf in pairs.items():
                if not os.path.exists(srcf):
                    raise FileNotFoundError(srcf)
                link = os.path.join(dst, sub, new + ".png")
                if not os.path.lexists(link):
                    try:
                        os.symlink(os.path.abspath(srcf), link)
                    except OSError:              # Windows without symlink rights: copy instead
                        shutil.copy(srcf, link)
            names.append(new)
        with open(os.path.join(dst, f"{split}.txt"), "w") as fh:
            fh.write("\n".join(names) + "\n")
        counts[split] = len(names)
    print(f"CMX-format dataset at {dst}: {counts}")
    return counts


# -----------------------------------------------------------------------------
# 3. Write config.py for PST900 (same structure/defaults as the repo's config.py).
# -----------------------------------------------------------------------------
CONFIG_TEMPLATE = '''import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = {seed}

C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

"""Dataset Path"""
C.dataset_name = 'PST900'
C.dataset_path = r'{dataset_path}'
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = False          # PST900: label 0 = Background is a VALID class (do not shift)
C.x_root_folder = osp.join(C.dataset_path, 'Thermal')
C.x_format = '.png'
C.x_is_single_channel = True    # thermal is single channel -> replicated to 3 channels
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = {num_train}
C.num_eval_imgs = {num_eval}
C.num_classes = 5
C.class_names = {class_names}

"""Image Config"""
C.background = 255
C.image_height = {crop_h}
C.image_width = {crop_w}
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

"""Network"""
C.backbone = '{backbone}'
C.pretrained_model = r'{pretrained}'
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = {lr}
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = {batch_size}
C.nepochs = {nepochs}
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = {num_workers}
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [{crop_h}, {crop_w}]

"""Store Config"""
C.checkpoint_start_epoch = {ckpt_every}
C.checkpoint_step = {ckpt_every}

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = r'{log_dir}'
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'
'''


def write_config(repo_dir, dataset_path, pretrained, log_dir, num_train, num_eval,
                 backbone="mit_b2", batch_size=8, nepochs=500, lr=6e-5, crop=(480, 640),
                 num_workers=2, ckpt_every=10, seed=12345):
    os.makedirs(log_dir, exist_ok=True)
    txt = CONFIG_TEMPLATE.format(
        seed=seed, dataset_path=dataset_path, num_train=num_train, num_eval=num_eval,
        class_names=PST900_CLASSES, crop_h=crop[0], crop_w=crop[1], backbone=backbone,
        pretrained=pretrained, lr=lr, batch_size=batch_size, nepochs=nepochs,
        num_workers=num_workers, ckpt_every=ckpt_every, log_dir=log_dir)
    path = os.path.join(repo_dir, "config.py")
    if os.path.exists(path) and not os.path.exists(path + ".orig"):
        shutil.copy(path, path + ".orig")
    open(path, "w").write(txt)
    print(f"Wrote {path}  (bs={batch_size}, epochs={nepochs}, lr={lr}, ckpt every {ckpt_every})")


# -----------------------------------------------------------------------------
# 4. Model helpers
# -----------------------------------------------------------------------------
def _import_repo(repo_dir):
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    for m in [k for k in list(sys.modules) if k == "config" or k.startswith(("models", "engine", "utils"))]:
        del sys.modules[m]                       # force re-import of the freshly written config
    cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        from config import config
        from models.builder import EncoderDecoder
    finally:
        os.chdir(cwd)
    return config, EncoderDecoder


def build_model(repo_dir, checkpoint=None, device="cuda", with_pretrained=False):
    """CMX EncoderDecoder from the repo; loads a training checkpoint if given."""
    import torch
    import torch.nn as nn
    config, EncoderDecoder = _import_repo(repo_dir)
    crit = nn.CrossEntropyLoss(ignore_index=255) if with_pretrained else None
    model = EncoderDecoder(cfg=config, criterion=crit, norm_layer=nn.BatchNorm2d)
    if checkpoint:
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        print(f"Loaded checkpoint {os.path.basename(checkpoint)}")
    return model.to(device).eval()


def verify_pretrained(repo_dir, pretrained_path):
    """The repo loads ImageNet weights with strict=False, which hides failures. Check it."""
    import torch
    raw = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    raw = raw.get("model", raw)
    model = build_model(repo_dir, None, device="cpu", with_pretrained=True)
    own = model.backbone.state_dict()
    hit = [k for k in raw if k in own and own[k].shape == raw[k].shape
           and torch.equal(own[k].cpu(), raw[k].cpu())]
    enc_keys = [k for k in raw if any(s in k for s in ("patch_embed", "block", "norm"))]
    print(f"Pretrained keys loaded into RGB stream: {len(hit)}/{len(enc_keys)}")
    if len(hit) < 0.9 * len(enc_keys):
        raise RuntimeError("Pretrained weights did NOT load -- check the file/backbone name.")
    return len(hit), len(enc_keys)


def latest_checkpoint(log_dir):
    ck = glob.glob(os.path.join(log_dir, "checkpoint", "epoch-*.pth"))
    ck = [c for c in ck if re.search(r"epoch-(\d+)\.pth$", c)]
    if not ck:
        return None
    return max(ck, key=lambda c: int(re.search(r"epoch-(\d+)\.pth$", c).group(1)))


# -----------------------------------------------------------------------------
# 5. Speed patches (results-equivalent data pipeline + optional mixed precision)
# -----------------------------------------------------------------------------

def patch_speed(repo_dir, amp=True):
    """
    Speed patches for the official CMX repo (results-equivalent):
      1. Data pipeline: crop BEFORE normalizing (the repo normalizes the whole up-to-1.75x
         scaled 720x1280 image in float64, then throws most of it away). Output is
         identical: padded pixels are still exactly 0 after normalization.
      2. Mixed precision (fp16 autocast + GradScaler) on the GPU: ~1.5-2x faster on T4
         and roughly half the activation memory. Set amp=False to switch it off.
    """
    def edit(rel, pairs):
        p = os.path.join(repo_dir, rel)
        src = open(p).read()
        for old, new in pairs:
            if new in src:
                continue
            if old not in src:
                raise RuntimeError(f"Patch target not found in {rel}: {old[:60]!r}")
            src = src.replace(old, new)
        open(p, "w").write(src)

    edit("dataloader/dataloader.py", [(
"""        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
""",
"""        # [speed patch] crop first, normalize only the crop (identical output)
        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
        valid, _ = random_crop_pad_to_shape(np.ones(rgb.shape[:2], np.float32), crop_pos, crop_size, 0)
        valid = valid[..., None]
        _m = np.asarray(self.norm_mean, np.float32); _s = np.asarray(self.norm_std, np.float32)
        p_rgb = ((p_rgb.astype(np.float32) / 255.0 - _m) / _s) * valid
        p_modal_x = ((p_modal_x.astype(np.float32) / 255.0 - _m) / _s) * valid
""")])

    if amp:
        edit("train.py", [
            ("    optimizer.zero_grad()\n    model.train()\n",
             "    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())  # [speed patch] AMP\n"
             "    optimizer.zero_grad()\n    model.train()\n"),
            ("            loss = model(imgs, modal_xs, gts)\n",
             "            with torch.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):\n"
             "                loss = model(imgs, modal_xs, gts)\n"),
            ("            loss.backward()\n            optimizer.step()\n",
             "            scaler.scale(loss).backward()\n            scaler.step(optimizer)\n            scaler.update()\n"),
        ])
    print("Speed patches applied: crop-then-normalize" + (" + mixed precision (AMP)" if amp else ""))
