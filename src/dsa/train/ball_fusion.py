"""Early-fusion ball experiment: RF-DETR Medium backbone + ball heatmap head, on TrackNet tennis.

Question: with frames t-1, t, t+1 stacked on channels, does the RF-DETR Medium
backbone find the ball as well as WASB? First test of the single-model design
(docs/training/MODEL.md).

Model
  backbone  RF-DETR Medium's DINOv2-S encoder + projector (COCO weights), the
            DETR decoder dropped. The patch embedding Conv2d(3, 384, 16, 16)
            becomes Conv2d(9, 384, 16, 16): the middle frame's slice keeps the
            pretrained weights, the neighbours start at zero, so step 0 is the
            single-frame model. `frames=1` keeps 3 channels (the ablation).
  input     1280 x 720, padded at the bottom to 736 (divisible by 16 px patches
            x 2 windows): 80 x 46 = 3,680 tokens, position embeddings
            interpolated by the encoder. Native resolution, so a 3-8 px ball
            is never shrunk below a few pixels.
  ball head dense heatmap at stride 4 (320 x 184): the stride-16 projector map
            upsampled twice, concatenated with a 3-layer conv stem on frame t
            only (sub-patch localisation, no motion cue, so any motion signal
            must come through the backbone), CenterNet focal loss on a Gaussian
            (sigma 1 cell). A visibility logit from pooled backbone features
            (BCE). Chosen over a single DETR "ball query" because a heatmap
            is dense supervision that converges in a few epochs on 14k frames,
            and is what TrackNet / WASB use; a query can come later.
  decode    arg-max cell, 3 x 3 weighted centroid, x = 4u + 1.5. The ball is
            called visible when the peak probability > 0.5 (WASB's rule);
            the visibility logit is scored too.

Data: TrackNet games 1-7 train, 8-10 test (WASB's split), read straight from
each clip's Label.csv. Optionally RacketVision tennis (`rv=True`): one 10 s
rally from each of 431 broadcast matches, 50 human-labelled frames per rally,
its own train / val / test split by match (only train is trained on, test is
scored alongside). `dsa.cloud.train_ball.prepare_rv` extracts each labelled
frame and its neighbours at 1280 x 720 into <root>/rv/<match>_<rally>/ with
an index.csv; the neighbours are +-1 frame at 25 / 30 fps and +-2 at 60 fps,
so the motion between the three frames is about TrackNet's (30 fps). Visibility 1, 2 = ball, 0 = none, 3 (occluded, 82
frames) is left out of the loss. At clip edges the missing neighbour repeats
frame t. Train augmentation, identical on the three frames: flip, scale 0.75-
1.25 with shift, brightness / contrast.

Metric (as dsa.astra.ball / WASB): a labelled ball (vis 1, 2) is a true
positive if predicted visible within 4 px (also 10 px) at 1280 x 720; any
other visible prediction is a false positive, and a missed ball a false
negative, so a far detection is both. Every test frame is scored.

    modal run src/dsa/cloud/train_ball.py                  # train + evaluate on Modal
    python -m dsa.train.ball_fusion wasb                   # WASB on the same full test set, locally
    python -m dsa.train.ball_fusion wasb --rv-test --root <dir holding rv/> --out ...   # RacketVision test
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

TRAIN_GAMES = tuple(f"game{i}" for i in range(1, 8))
TEST_GAMES = ("game8", "game9", "game10")
W, H, H_PAD, STRIDE = 1280, 720, 736, 4
RESUME_WARM = 200        # LR ramp after resuming from weights only (fresh AdamW moments)
WARM_STEPS = 30          # benchmark timing starts after these (loader workers and cudnn warm up)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- data

def load_items(root: Path, games: tuple[str, ...]) -> pd.DataFrame:
    """One row per labelled frame, with its neighbours (repeated at clip edges)."""
    rows = []
    for game in games:
        for clip in sorted((root / game).iterdir(), key=lambda p: p.name):
            if not (clip / "Label.csv").exists():
                continue
            lab = pd.read_csv(clip / "Label.csv")
            names = lab["file name"].tolist()
            n = len(names)
            for i, r in enumerate(lab.itertuples(index=False)):
                rows.append({"clip": f"{game}/{clip.name}", "prev": names[max(i - 1, 0)], "file": names[i],
                             "next": names[min(i + 1, n - 1)], "vis": int(r[1]),
                             "x": float(r[2]) if r[1] else np.nan, "y": float(r[3]) if r[1] else np.nan})
    return pd.DataFrame(rows)


def load_rv_items(root: Path, split: str) -> pd.DataFrame:
    """RacketVision rows (index.csv from prepare_rv), same columns as load_items; clip = rv/<match>_<rally>."""
    idx = pd.read_csv(root / "rv" / "index.csv", dtype={"prev": str, "file": str, "next": str})
    idx = idx[idx.split == split]
    return idx[["clip", "prev", "file", "next", "vis", "x", "y"]].reset_index(drop=True)


def heatmap_target(x: float, y: float, sigma: float = 1.0) -> np.ndarray:
    """Gaussian at the ball on the stride-4 grid; the nearest cell is exactly 1 (CenterNet)."""
    hs, ws = H_PAD // STRIDE, W // STRIDE
    u, v = (x - 1.5) / STRIDE, (y - 1.5) / STRIDE
    gx = np.exp(-((np.arange(ws) - u) ** 2) / (2 * sigma**2))
    gy = np.exp(-((np.arange(hs) - v) ** 2) / (2 * sigma**2))
    t = (gy[:, None] * gx[None, :]).astype(np.float32)
    iu, iv = int(round(u)), int(round(v))
    if 0 <= iu < ws and 0 <= iv < hs:
        t[iv, iu] = 1.0
    return t


class BallFrames(torch.utils.data.Dataset):
    def __init__(self, root: Path, items: pd.DataFrame, frames: int, train: bool):
        self.root, self.items, self.frames, self.train = Path(root), items.reset_index(drop=True), frames, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int):
        it = self.items.iloc[k]
        names = (it.prev, it.file, it.next) if self.frames == 3 else (it.file,)
        imgs = [cv2.cvtColor(cv2.imread(str(self.root / it["clip"] / n)), cv2.COLOR_BGR2RGB) for n in names]
        vis, x, y = int(it.vis), float(it.x), float(it.y)
        if self.train:
            imgs, x, y, vis = self._augment(imgs, x, y, vis)
        img = np.concatenate(imgs, axis=2)          # uint8, H x W x 3*frames; normalised on the GPU (to_input)
        ball = vis in (1, 2)
        hm = heatmap_target(x, y) if ball else np.zeros((H_PAD // STRIDE, W // STRIDE), np.float32)
        return (torch.from_numpy(img.transpose(2, 0, 1).copy()), torch.from_numpy(hm),
                torch.tensor(float(ball)), torch.tensor(float(vis != 3)), k)

    def _augment(self, imgs, x, y, vis):
        rng = np.random.default_rng()
        s = rng.uniform(0.75, 1.25) if rng.random() < 0.5 else 1.0
        tx, ty = rng.uniform(-0.1, 0.1) * W, rng.uniform(-0.1, 0.1) * H
        flip = rng.random() < 0.5
        M = np.array([[s, 0, (1 - s) * W / 2 + tx], [0, s, (1 - s) * H / 2 + ty]], np.float32)
        if flip:
            M[0] = -M[0]
            M[0, 2] += W - 1
        if not np.allclose(M, [[1, 0, 0], [0, 1, 0]]):
            imgs = [cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(124, 116, 104)) for im in imgs]
            if vis != 0:
                x, y = M @ np.array([x, y, 1.0])
                if not (0 <= x < W and 0 <= y < H):
                    vis = 0
        a, b = rng.uniform(0.7, 1.3), rng.uniform(-25, 25)
        imgs = [np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8) for im in imgs]
        return imgs, float(x), float(y), vis


def to_input(img: torch.Tensor) -> torch.Tensor:
    """uint8 B x 3F x H x W from the loader -> normalised float on the GPU, padded to H_PAD.

    Normalising here rather than in the workers moves a quarter of the bytes through the loader (a float32
    9-channel 1280 x 736 batch of 16 is 540 MB, which starved faster GPUs).
    """
    x = img.cuda(non_blocking=True).float().div_(255.0)
    f = x.shape[1] // 3
    mean = torch.tensor(MEAN, device=x.device).repeat(f)[None, :, None, None]
    std = torch.tensor(STD, device=x.device).repeat(f)[None, :, None, None]
    return F.pad((x - mean) / std, (0, 0, 0, H_PAD - H))


# ---------------------------------------------------------------- model

def _conv(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True))


class BallFusion(nn.Module):
    def __init__(self, frames: int = 3, weights: str | None = None):
        super().__init__()
        from rfdetr import RFDETRMedium

        bb = RFDETRMedium(pretrain_weights=weights).model.model.backbone[0]
        self.encoder, self.projector, self.frames = bb.encoder, bb.projector, frames
        if frames == 3:
            pe = self.encoder.encoder.embeddings.patch_embeddings
            old = pe.projection
            new = nn.Conv2d(9, old.out_channels, old.kernel_size, old.stride)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, 3:6] = old.weight
                new.bias.copy_(old.bias)
            pe.projection, pe.num_channels = new, 9
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"), _conv(256, 128),
                                nn.Upsample(scale_factor=2, mode="bilinear"), _conv(128, 64))
        self.stem = nn.Sequential(_conv(3, 32, s=2), _conv(32, 64, s=2), _conv(64, 64))
        self.fuse = nn.Sequential(_conv(128, 64), nn.Conv2d(64, 1, 1))
        nn.init.constant_(self.fuse[-1].bias, -4.6)
        self.vis = nn.Sequential(nn.Linear(512, 128), nn.SiLU(), nn.Linear(128, 1))

    def patch_embed_params(self):
        return list(self.encoder.encoder.embeddings.patch_embeddings.parameters())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.projector(self.encoder(x))[0]                     # B, 256, H/16, W/16
        mid = x[:, 3:6] if self.frames == 3 else x
        hm = self.fuse(torch.cat([self.up(f), self.stem(mid)], 1))[:, 0]
        vis = self.vis(torch.cat([f.amax((2, 3)), f.mean((2, 3))], 1))[:, 0]
        return hm, vis


def focal_loss(logits, target, mask):
    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos_l = torch.log(p) * (1 - p) ** 2 * pos
    neg_l = torch.log(1 - p) * p**2 * (1 - target) ** 4 * (1 - pos)
    m = mask[:, None, None]
    return -((pos_l + neg_l) * m).sum() / (pos * m).sum().clamp(min=1)


@torch.no_grad()
def decode(hm_logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peak probability and sub-cell position, in 1280 x 720 pixels."""
    p = hm_logits.float().sigmoid()
    B, hs, ws = p.shape
    idx = p.flatten(1).argmax(1)
    iy, ix = idx // ws, idx % ws
    score = p.flatten(1).gather(1, idx[:, None])[:, 0]
    pp = F.pad(p, (1, 1, 1, 1))
    xs, ys = [], []
    for b in range(B):
        win = pp[b, iy[b]:iy[b] + 3, ix[b]:ix[b] + 3]
        d = torch.arange(-1, 2, device=p.device, dtype=p.dtype)
        xs.append(ix[b] + (win.sum(0) * d).sum() / win.sum())
        ys.append(iy[b] + (win.sum(1) * d).sum() / win.sum())
    x = torch.stack(xs) * STRIDE + 1.5
    y = torch.stack(ys) * STRIDE + 1.5
    return score.cpu().numpy(), x.cpu().numpy(), y.cpu().numpy()


# ---------------------------------------------------------------- metrics

def score_frames(df: pd.DataFrame, visible: pd.Series) -> dict:
    """WASB-style precision / recall / F1 at 4 and 10 px, plus empty-frame behaviour."""
    labelled = df.vis.isin([1, 2])
    dist = np.hypot(df.px - df.x, df.py - df.y).where(labelled & visible)
    out = {"frames": int(len(df)), "labelled": int(labelled.sum()), "empty": int((df.vis == 0).sum())}
    for px in (4, 10):
        tp = int((dist <= px).sum())
        fp = int(visible.sum()) - tp
        fn = int(labelled.sum()) - tp
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out |= {f"precision@{px}": p, f"recall@{px}": r, f"f1@{px}": 2 * p * r / max(p + r, 1e-9),
                f"tp@{px}": tp, f"fp@{px}": fp, f"fn@{px}": fn}
    out["median_dist_px"] = float(dist.median())
    out["said_visible_when_none"] = int((visible & (df.vis == 0)).sum())
    out["said_none_when_visible"] = int((~visible & labelled).sum())
    return out


def summarize(df: pd.DataFrame) -> dict:
    """Headline (peak > 0.5), the visibility-logit rule, and a threshold sweep (oracle, for reference)."""
    s = {"peak>0.5": score_frames(df, df.score > 0.5)}
    if "vis_prob" in df:
        s["vis_logit>0.5"] = score_frames(df, df.vis_prob > 0.5)
    sweep = {f"{t:.2f}": score_frames(df, df.score > t)["f1@4"] for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    s["f1@4_by_peak_threshold"] = sweep
    s["by_game"] = {g: score_frames(d, d.score > 0.5) for g, d in df.groupby(df["clip"].str.split("/").str[0])}
    return s


def visualize(df: pd.DataFrame, root: Path, out: Path, n: int = 16, prefix: str = "") -> None:
    """Crops around the label (green) and prediction (red): worst misses, false alarms, random."""
    def crop(r, size=160):
        im = cv2.imread(str(root / r.clip / r.file))
        cx, cy = (r.x, r.y) if r.vis in (1, 2) else (r.px, r.py)
        if r.vis in (1, 2):
            cv2.circle(im, (int(r.x), int(r.y)), 7, (0, 255, 0), 1)
        if r.score > 0.5:
            cv2.drawMarker(im, (int(r.px), int(r.py)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)
        x0 = int(np.clip(cx - size / 2, 0, W - size))
        y0 = int(np.clip(cy - size / 2, 0, H - size))
        c = cv2.resize(im[y0:y0 + size, x0:x0 + size], (320, 320), interpolation=cv2.INTER_NEAREST)
        d = "" if np.isnan(r.dist) else f" d={r.dist:.0f}"
        cv2.putText(c, f"{r.clip} {r.file} v{r.vis} p={r.score:.2f}{d}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
        # context thumbnail: whole frame, bottom-right
        th = cv2.resize(im, (128, 72))
        c[-72:, -128:] = th
        return c

    def grid(rows, name):
        tiles = [crop(r) for r in rows.itertuples()]
        if not tiles:
            return
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / (prefix + name)), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))

    lab = df[df.vis.isin([1, 2])].copy()
    lab["miss"] = np.where(lab.score > 0.5, lab.dist.fillna(1e4), 1e4 + (1 - lab.score))
    grid(lab.sort_values("miss", ascending=False).head(n), "worst_misses.jpg")
    grid(lab[(lab.score > 0.5) & (lab.dist > 4)].sort_values("dist").head(n), "near_misses_4_to_10px.jpg")
    grid(df[(df.vis == 0) & (df.score > 0.5)].sort_values("score", ascending=False).head(n), "false_alarms_empty.jpg")
    grid(df.sample(min(n, len(df)), random_state=0), "random.jpg")


# ---------------------------------------------------------------- train / evaluate

@torch.no_grad()
def predict(model, root: Path, items: pd.DataFrame, frames: int, batch: int, workers: int) -> pd.DataFrame:
    model.eval()
    dl = torch.utils.data.DataLoader(BallFrames(root, items, frames, train=False), batch, num_workers=workers,
                                     pin_memory=True)
    out = []
    for img, *_, k in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hm, vis = model(to_input(img))
        s, x, y = decode(hm)
        out.append(pd.DataFrame({"k": k.numpy(), "score": s, "px": x, "py": y,
                                 "vis_prob": vis.float().sigmoid().cpu().numpy()}))
    df = items.reset_index(drop=True).join(pd.concat(out).set_index("k"))
    df["dist"] = np.hypot(df.px - df.x, df.py - df.y).where(df.vis.isin([1, 2]) & (df.score > 0.5))
    return df


def train(root: Path, out: Path, frames: int = 3, epochs: int = 12, batch: int = 16, lr: float = 1e-4,
          workers: int = 12, weights: str | None = None, limit: int | None = None, rv: bool = False,
          rv_train: bool = True, max_steps: int | None = None, resume: bool = False, log=print,
          on_checkpoint=None) -> dict:
    """Train, then score the final checkpoint on TrackNet games 8-10 (and RacketVision test when `rv`).

    `max_steps` stops early and returns only timing (the GPU benchmark): seconds per step over the steps
    after the first WARM_STEPS, and the share of it spent waiting on the data loader. `rv` scores
    RacketVision test too; `rv_train` (with `rv`) also trains on RacketVision train.

    Each epoch writes last.pt (model weights, as before) and resume.pt (optimizer, epoch, step). `resume`
    continues from <out>/last.pt after the last finished epoch, on the same LR schedule. Without resume.pt
    (runs from before it existed) the epoch comes from history.json and AdamW restarts from zero moments,
    re-warmed with a short linear ramp (RESUME_WARM steps), which the log states.
    """
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tr, te = load_items(root, TRAIN_GAMES), load_items(root, TEST_GAMES)
    te_rv = load_rv_items(root, "test") if rv else None
    if rv and rv_train:
        tr = pd.concat([tr, load_rv_items(root, "train")], ignore_index=True)
    if limit:
        tr, te = tr.sample(limit, random_state=0), te.sample(min(limit, len(te)), random_state=0)
        te_rv = te_rv.sample(min(limit, len(te_rv)), random_state=0) if rv else None
    n_rv = int(tr["clip"].str.startswith("rv/").sum())
    log(f"train {len(tr)} frames ({n_rv} RacketVision), test {len(te)} frames"
        f"{f' + {len(te_rv)} RacketVision' if rv else ''}, frames={frames}")
    model = BallFusion(frames, weights).cuda()
    pe = {id(p) for p in model.patch_embed_params()}
    backbone = [p for n, p in model.named_parameters() if n.startswith(("encoder", "projector")) and id(p) not in pe]
    head = [p for n, p in model.named_parameters() if not n.startswith(("encoder", "projector"))]
    opt = torch.optim.AdamW([{"params": backbone, "lr": lr}, {"params": model.patch_embed_params(), "lr": lr * 10},
                             {"params": head, "lr": lr * 10}], weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(BallFrames(root, tr, frames, train=True), batch, shuffle=True, drop_last=True,
                                     num_workers=workers, pin_memory=True, persistent_workers=True)
    total, warm = epochs * len(dl), min(500, epochs * len(dl) // 10)
    base = [g["lr"] for g in opt.param_groups]
    step, t0, history, start, rewarm = 0, time.time(), [], 0, 0
    if resume and (out / "last.pt").exists():
        model.load_state_dict(torch.load(out / "last.pt", map_location="cuda"))
        if (out / "resume.pt").exists():
            st = torch.load(out / "resume.pt", map_location="cuda", weights_only=False)
            opt.load_state_dict(st["opt"])
            start, step, history = st["epoch"] + 1, st["step"], st["history"]
            log(f"resumed after epoch {start - 1} (step {step}) with optimizer state")
        else:
            history = json.loads((out / "history.json").read_text())
            start = history[-1]["epoch"] + 1
            step, rewarm = start * len(dl), RESUME_WARM
            log(f"resumed after epoch {start - 1} (step {step}) from weights only: fresh AdamW, "
                f"{rewarm}-step re-warm")
        t0 -= history[-1]["minutes"] * 60 if history else 0
    resume_step = step
    wait = 0.0          # seconds blocked on the data loader (after the first WARM_STEPS)
    for ep in range(start, epochs):
        model.train()
        run_hm = run_vis = 0.0
        t_fetch = time.time()
        for i, (img, hm_t, vis_t, mask, _) in enumerate(dl):
            if step == WARM_STEPS:
                torch.cuda.synchronize()
                t10, wait = time.time(), 0.0
            wait += time.time() - t_fetch
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            if step - resume_step < rewarm:
                f *= (step - resume_step + 1) / rewarm
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            img = to_input(img)
            hm_t, vis_t, mask = (t.cuda(non_blocking=True) for t in (hm_t, vis_t, mask))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hm, vis = model(img)
            l_hm = focal_loss(hm, hm_t, mask)
            l_vis = (F.binary_cross_entropy_with_logits(vis.float(), vis_t, reduction="none") * mask).mean()
            loss = l_hm + l_vis
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            run_hm += l_hm.item()
            run_vis += l_vis.item()
            if max_steps and step >= max_steps:
                torch.cuda.synchronize()
                sec = (time.time() - t10) / (step - WARM_STEPS)
                bench = {"gpu": torch.cuda.get_device_name(), "steps": step - WARM_STEPS, "s_per_step": sec,
                         "loader_wait_share": wait / (time.time() - t10), "workers": workers, "batch": batch,
                         "steps_per_epoch": len(dl), "hm_loss": run_hm / (i + 1)}
                log(json.dumps(bench))
                return bench
            if (i + 1) % 100 == 0:
                log(f"ep {ep} it {i + 1}/{len(dl)} hm {run_hm / (i + 1):.3f} vis {run_vis / (i + 1):.3f} "
                    f"{(time.time() - t0) / step:.2f}s/it")
            t_fetch = time.time()
        history.append({"epoch": ep, "hm_loss": run_hm / len(dl), "vis_loss": run_vis / len(dl),
                        "minutes": (time.time() - t0) / 60})
        torch.save(model.state_dict(), out / "last.pt")
        torch.save({"opt": opt.state_dict(), "epoch": ep, "step": step, "history": history}, out / "resume.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if on_checkpoint:
            on_checkpoint()          # e.g. commit the Modal volume, so a killed container can resume
        log(json.dumps(history[-1]))
        if ep in (0, epochs // 4, epochs // 2, 3 * epochs // 4):   # monitoring only, on fixed 600-frame test
            # subsets; the final checkpoint is what is reported
            for name, t in [("tracknet", te)] + ([("racketvision", te_rv)] if rv else []):
                m = summarize(predict(model, root, t.sample(min(600, len(t)), random_state=1), frames, batch,
                                      workers))
                log(f"monitor ep {ep} {name}: " + json.dumps({k: round(v, 3) for k, v in m["peak>0.5"].items()
                                                              if k.startswith(("f1", "median"))}))
    train_min = (time.time() - t0) / 60
    summary = {"frames": frames, "epochs": epochs, "batch": batch, "lr": lr, "train_frames": len(tr),
               "train_frames_racketvision": n_rv, "steps": step, "train_minutes": train_min,
               "gpu": torch.cuda.get_device_name(), **evaluate(model, root, out, frames, batch, workers, te, te_rv)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def evaluate(model, root: Path, out: Path, frames: int, batch: int, workers: int, te: pd.DataFrame,
             te_rv: pd.DataFrame | None) -> dict:
    """TrackNet games 8-10 (per_frame.parquet, headline) and RacketVision test (per_frame_rv_test.parquet)."""
    df = predict(model, root, te, frames, batch, workers)
    df.to_parquet(out / "per_frame.parquet", index=False)
    visualize(df, root, out)
    s = summarize(df)
    if te_rv is not None:
        d = predict(model, root, te_rv, frames, batch, workers)
        d.to_parquet(out / "per_frame_rv_test.parquet", index=False)
        visualize(d, root, out, prefix="rv_test_")
        s["racketvision_test"] = {k: v for k, v in summarize(d).items() if k != "by_game"}
    return s


def evaluate_checkpoint(root: Path, ckpt: Path, out: Path, frames: int = 3, batch: int = 16, workers: int = 12,
                        rv: bool = True, weights: str | None = None) -> dict:
    """Score a saved checkpoint (e.g. the TrackNet-only run) on both test sets."""
    out.mkdir(parents=True, exist_ok=True)
    model = BallFusion(frames, weights).cuda()
    model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    s = evaluate(model, root, out, frames, batch, workers, load_items(root, TEST_GAMES),
                 load_rv_items(root, "test") if rv else None)
    (out / "summary.json").write_text(json.dumps({"checkpoint": str(ckpt), **s}, indent=2, default=float))
    return s


# ---------------------------------------------------------------- WASB on the same frames

def wasb_eval(root: Path, out: Path, device: str, rv_test: bool = False) -> dict:
    """WASB (dsa.astra.ball.Wasb, no tracker) on every test frame, scored identically.

    `rv_test`: RacketVision's test split instead (root must hold rv/, as prepare_rv writes it).
    """
    from dsa.astra.ball import Wasb

    out.mkdir(parents=True, exist_ok=True)
    te = load_rv_items(root, "test") if rv_test else load_items(root, TEST_GAMES)
    wasb, rows, t0 = Wasb(device), [], time.time()
    for i, it in enumerate(te.itertuples()):
        frames = [cv2.imread(str(root / it.clip / n)) for n in (it.prev, it.file, it.next)]
        vis, x, y, score = wasb.locate(frames)
        rows.append({"score": 1.0 if vis else 0.0, "px": x, "py": y, "blob": score})
        if i % 500 == 0:
            print(f"{i}/{len(te)} {time.time() - t0:.0f}s", flush=True)
    df = te.join(pd.DataFrame(rows))
    df["dist"] = np.hypot(df.px - df.x, df.py - df.y).where(df.vis.isin([1, 2]) & (df.score > 0.5))
    df.to_parquet(out / "per_frame.parquet", index=False)
    s = {"peak>0.5": score_frames(df, df.score > 0.5),
         "by_game": {g: score_frames(d, d.score > 0.5) for g, d in df.groupby(df["clip"].str.split("/").str[0])}}
    (out / "summary.json").write_text(json.dumps(s, indent=2, default=float))
    visualize(df, root, out)
    return s


def main() -> None:
    from dsa.data.paths import SOURCES

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["wasb"])
    p.add_argument("--root", default=str(SOURCES / "tracknet/TrackNet/Dataset"))
    p.add_argument("--out", default="output/train/ball_fusion/wasb_full_test")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--rv-test", action="store_true", help="RacketVision test split (root must hold rv/)")
    a = p.parse_args()
    print(json.dumps(wasb_eval(Path(a.root), Path(a.out), a.device, a.rv_test)["peak>0.5"], indent=2, default=float))


if __name__ == "__main__":
    main()
