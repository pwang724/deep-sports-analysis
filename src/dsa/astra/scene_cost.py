"""Cheaper scene calls: variants of the prefill scene call (dsa.label.prefill) scored per field.

Variants (prefill.VARIANTS): v0 as clips runs it (keyframe, diagram, 640 px
sheet, boxed keyframe); v1 slim (diagram, 320 px sheet, keyframe with thin
corner brackets, trimmed prompt); v2 = v1 for 4 frames per call.

Test sets, each scored only on the fields it has ground truth for:

  court    the first 40 of dsa.astra.court's 100 TennisCourtDetector val
           frames; a still, so the sheet repeats the frame. Within 7 / 15 px,
           median px, as dsa.astra.court.
  view     dsa.astra.view_play samples (10 per class per task): view on USO
  in play  highlights, in play on the phone recordings, real 0.5 s sheets.
  players  40 TennisSegmentation frames (stills). RF-DETR boxes as prefill;
           ground truth = detections matching a labelled player at IoU > 0.5.
           Exact-set accuracy, precision and recall of the chosen boxes.

RF-DETR detections are computed once and saved, so the images (and so the
cache keys) do not change between reruns. Cost is read from each call's usage
and from Codex's own meter: the used_percent / credits of the rate_limits in
~/.codex/sessions, before and after each variant (run variants one at a time).

    python -m dsa.astra.scene_cost --prepare           # local only: detections and images
    python -m dsa.astra.scene_cost --variant v0        # then v1, v2
    python -m dsa.astra.scene_cost --score
"""
from __future__ import annotations

import argparse
import json
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from dsa.astra import court as court_eval
from dsa.astra import view_play
from dsa.astra.codex import EFFORT, MODEL, usage
from dsa.data.paths import MODELS, SOURCES
from dsa.label import prefill
from dsa.label.prefill import BATCH, VARIANTS, draw_boxes, sheet_image

OUT = Path("output/astra_eval/scene_cost")
N = 40
OFFSETS = (-0.75, -0.25, 0.25, 0.75)
SESSIONS = Path.home() / ".codex" / "sessions"


def court_items() -> list[dict]:
    out = []
    for it in court_eval.sample(100)[:N]:
        bgr = cv2.imread(str(court_eval.ROOT / "images" / f"{it['id']}.png"))
        out.append({"set": "court", "id": f"court_{it['id']}", "bgr": bgr, "sheet": [bgr] * 4, "gt": it})
    return out


def view_play_items() -> list[dict]:
    rng = np.random.default_rng(0)
    samples = view_play.view_samples(N // 4, rng) + view_play.play_samples(N // 4, rng)
    out = []
    for s in samples:
        video = view_play.USO_VIDEO if s["task"] == "view" else f"{view_play.PHONE}/{s['source']}/analysis.mp4"
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        for f in [s["frame"]] + [s["frame"] + round(dt * fps) for dt in OFFSETS]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, bgr = cap.read()
            assert ok, (video, f)
            frames.append(bgr)
        cap.release()
        out.append({"set": s["task"], "id": f"{s['task']}_{s['source']}_{s['frame']}", "bgr": frames[0],
                    "sheet": frames[1:], "gt": s})
    return out


def players_items() -> list[dict]:
    from dsa.data.tennis_segmentation import load

    samples = [s for s in load(SOURCES / "tennis_segmentation") if s.boxes]
    rng = np.random.default_rng(0)
    out = []
    for i in sorted(rng.choice(len(samples), N, replace=False)):
        s = samples[i]
        bgr = cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR)
        out.append({"set": "players", "id": f"players_{s.index}", "bgr": bgr, "sheet": [bgr] * 4,
                    "gt": {"boxes": [b.tolist() for b in s.boxes.values()]}})
    return out


def prepare(out: Path) -> list[dict]:
    """Items with RF-DETR people and every variant's images on disk; saved once and reused."""
    saved = out / "items.pkl"
    if saved.exists():
        return pickle.loads(saved.read_bytes())
    import torch

    from dsa.pose.backends import load_models

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    det = load_models("rfdetr", 1152, MODELS / "vitpose-plus-huge", device).detector
    items = court_items() + view_play_items() + players_items()
    img = out / "images"
    img.mkdir(parents=True, exist_ok=True)
    court_eval.diagram(out / "diagram.png")
    for k, it in enumerate(items):
        xyxy, conf = det.detect(cv2.cvtColor(it["bgr"], cv2.COLOR_BGR2RGB))
        it["people"] = [{"box": b} for b in xyxy[conf >= 0.4]]
        it["frame_path"] = img / f"{it['id']}.jpg"
        cv2.imwrite(str(it["frame_path"]), it["bgr"])
        for v in VARIANTS:
            cv2.imwrite(str(img / f"{it['id']}_{v}_boxes.jpg"), draw_boxes(it["bgr"], it["people"], v))
            if v != "v2":
                cv2.imwrite(str(img / f"{it['id']}_{v}_sheet.jpg"), sheet_image(it["sheet"], v))
        del it["sheet"]
        print(f"prepared {k + 1}/{len(items)}", flush=True)
    # v2 batches: 4 items of the same test set in shuffled order (so a batch mixes classes); the sheet
    # carries its moment number.
    by_set: dict[str, list] = {}
    for it in items:
        by_set.setdefault(it["set"], []).append(it)
    batches, rng = [], np.random.default_rng(0)
    for group in by_set.values():
        group = [group[i] for i in rng.permutation(len(group))]
        for b in range(0, len(group), BATCH):
            batches.append([it["id"] for it in group[b:b + BATCH]])
    for batch in batches:
        for k, iid in enumerate(batch):
            it = next(x for x in items if x["id"] == iid)
            frames = view_play_sheet(it) if it["set"] in ("view", "play") else [it["bgr"]] * 4
            cv2.imwrite(str(img / f"{iid}_v2_sheet.jpg"), sheet_image(frames, "v2", label=f"moment {k + 1}"))
    for it in items:
        it.pop("bgr")
    saved.write_bytes(pickle.dumps({"items": items, "batches": batches}))
    return pickle.loads(saved.read_bytes())


def view_play_sheet(it: dict) -> list[np.ndarray]:
    s = it["gt"]
    video = view_play.USO_VIDEO if s["task"] == "view" else f"{view_play.PHONE}/{s['source']}/analysis.mp4"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for dt in OFFSETS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame"] + round(dt * fps))
        frames.append(cap.read()[1])
    cap.release()
    return frames


def call_items(data: dict, out: Path, v: str) -> list[dict]:
    img = out / "images"
    items = []
    for it in data["items"]:
        items.append({**it, "bgr": cv2.imread(str(it["frame_path"])),
                      "sheet_path": img / f"{it['id']}_{v}_sheet.jpg", "boxes_path": img / f"{it['id']}_{v}_boxes.jpg"})
    return items


def meter() -> dict:
    """Codex's rate-limit state in the newest session log."""
    files = sorted((p for p in SESSIONS.glob("*/*/*/rollout-*.jsonl")), key=lambda p: p.stat().st_mtime)
    for f in reversed(files[-50:]):
        for line in reversed(f.read_text().splitlines()):
            if '"rate_limits"' in line:
                rl = json.loads(line)["payload"]["rate_limits"]
                return {"file": f.name, "mtime": f.stat().st_mtime, "primary": rl.get("primary"),
                        "credits": rl.get("credits"), "reached": rl.get("rate_limit_reached_type"),
                        "limit_id": rl.get("limit_id")}
    return {}


def meter_trace(since: float) -> list[dict]:
    """(time, used_percent, has_credits, reached) of every session written since `since`, in time order."""
    rows = []
    for f in SESSIONS.glob("*/*/*/rollout-*.jsonl"):
        if f.stat().st_mtime < since:
            continue
        for line in f.read_text().splitlines():
            if '"rate_limits"' in line:
                rl = json.loads(line)["payload"]["rate_limits"]
                p = rl.get("primary") or {}
                rows.append({"file": f.name, "mtime": f.stat().st_mtime, "used_percent": p.get("used_percent"),
                             "has_credits": (rl.get("credits") or {}).get("has_credits"),
                             "reached": rl.get("rate_limit_reached_type")})
    return sorted(rows, key=lambda r: r["mtime"])


def run(v: str, data: dict, out: Path, workers: int, limit: int | None) -> None:
    items = call_items(data, out, v)
    diagram = out / "diagram.png"
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    had = {q.stem for q in cache.glob("*.json") if not q.name.startswith("._")}
    before, t0 = meter(), time.time()
    by_id = {it["id"]: it for it in items}
    stop = threading.Event()   # first out-of-credits error stops every later call

    def work(b: list[str]) -> list[tuple]:
        if stop.is_set():
            raise RuntimeError("skipped: stopped after an out-of-credits error")
        try:
            if v == "v2":
                return list(zip(b, prefill.astra_scenes([by_id[i] for i in b], diagram, cache)))
            return [(b[0], prefill.astra_scene(by_id[b[0]], diagram, cache, v))]
        except RuntimeError as e:
            if "credits" in str(e) or "usage limit" in str(e):
                stop.set()
            raise

    batches = data["batches"] if v == "v2" else [[it["id"]] for it in items]
    batches = batches[:limit]
    answers, errors = {}, []
    with ThreadPoolExecutor(workers) as pool:
        for f in [pool.submit(work, b) for b in batches]:
            try:
                answers.update(dict(f.result()))
            except RuntimeError as e:
                errors.append(str(e)[-300:])
    time.sleep(2)
    new = {q.name.removesuffix(".usage.json") for q in cache.glob("*.usage.json") if not q.name.startswith("._")}
    res = {"variant": v, "calls": len(batches), "new_keys": sorted(new - had), "answered": len(answers),
           "errors": errors,
           "wall_sec": time.time() - t0, "meter_before": before, "meter_after": meter(),
           "trace": meter_trace(t0 - 1)}
    (out / f"run_{v}.json").write_text(json.dumps(res, indent=2, default=str))
    (out / f"answers_{v}.pkl").write_bytes(pickle.dumps(answers))
    print(json.dumps({k: res[k] for k in ("variant", "calls", "answered", "wall_sec")}), len(errors), "errors")
    if errors:
        print(errors[0])
    if stop.is_set():
        raise SystemExit("OUT OF CREDITS")


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def score(data: dict, out: Path) -> dict:
    items = {it["id"]: it for it in data["items"]}
    summary = {}
    for v in VARIANTS:
        path = out / f"answers_{v}.pkl"
        if not path.exists():
            continue
        ans = pickle.loads(path.read_bytes())
        r: dict = {}
        # court
        errs, rows = [], []
        for iid, it in items.items():
            if it["set"] != "court" or iid not in ans:
                continue
            c = ans[iid]["court"]
            xy = np.full((14, 2), np.nan) if c is None else c[:, :2]
            row = court_eval.score(v, xy, it["gt"])
            rows.append(row)
            errs += [row[f"err{k}"] for k in range(14) if f"err{k}" in row]
        errs = np.array(errs)
        r["court"] = {"frames": len(rows), "within7": float((errs <= 7).mean()), "within15": float((errs <= 15).mean()),
                      "median_px": float(np.median(errs)),
                      "frame_within15": [float(x["within15"]) for x in rows]}
        for task, key in (("view", "view"), ("play", "in_play")):
            g = [(it["gt"]["truth"], ans[iid][key]) for iid, it in items.items() if it["set"] == task and iid in ans]
            t, p = np.array(g).T.astype(bool) if g else (np.array([]), np.array([]))
            r[task] = {"n": len(g), "acc": float((t == p).mean()), "false_yes": float(p[~t].mean()),
                       "false_no": float((~p[t]).mean()), "correct": (t == p).astype(int).tolist()}
        exact, tp, fp, fn = [], 0, 0, 0
        for iid, it in items.items():
            if it["set"] != "players" or iid not in ans:
                continue
            gt = set()
            for g in it["gt"]["boxes"]:
                ious = [iou(p["box"], g) for p in it["people"]]
                if ious and max(ious) > 0.5:
                    gt.add(int(np.argmax(ious)))
            ch = set(ans[iid]["players"])
            exact.append(ch == gt)
            tp, fp, fn = tp + len(ch & gt), fp + len(ch - gt), fn + len(gt - ch)
        r["players"] = {"n": len(exact), "exact": float(np.mean(exact)), "precision": tp / max(tp + fp, 1),
                        "recall": tp / max(tp + fn, 1), "correct": [int(e) for e in exact]}
        run_info = json.loads((out / f"run_{v}.json").read_text())
        keys = {k for q in out.glob(f"run_{v}*.json") if not q.name.startswith("._")
                for k in json.loads(q.read_text())["new_keys"]}   # the run plus any pilot run
        r["usage"] = usage(out / "cache", MODEL, EFFORT, keys)
        r["run"] = {k: run_info[k] for k in ("calls", "answered", "wall_sec", "meter_before", "meter_after")}
        summary[v] = r
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--score", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="first N calls only")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = prepare(out)
    if a.variant:
        run(a.variant, data, out, a.workers, a.limit)
    if a.score:
        s = score(data, out)
        (out / "summary.json").write_text(json.dumps(s, indent=2, default=str))
        print(json.dumps({v: {k: {kk: vv for kk, vv in x.items() if kk not in ("correct", "frame_within15")}
                              if isinstance(x, dict) else x for k, x in r.items() if k != "run"}
                          for v, r in s.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
