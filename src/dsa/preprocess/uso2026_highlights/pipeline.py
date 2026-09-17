"""Cleanup for the USO 2026 highlights clip; not a general video preprocessor."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..media import contact_sheet, extract_frames, make_proxy, source_info
from ..manifest import validate_manifest

MODEL = "facebook/dinov2-small"
PROFILE = Path(__file__).with_name("profile.json")

def sample_frames(start: int, end: int) -> list[int]:
    if end <= start:
        raise ValueError("Shot must contain at least one frame")
    return sorted({start + int((end - start - 1) * f) for f in (0.15, 0.5, 0.85)})


def classify(samples: np.ndarray, refs: np.ndarray, labels: list[str], margin: float = 0.0) -> dict:
    """Nearest reference of each class, then unanimous vote across shot samples.

    Scores are cosine similarities, not calibrated probabilities. Disagreement
    or a small score gap means review, never an automatic keep.
    """
    if set(labels) != {"keep", "discard"} or len(labels) != len(refs):
        raise ValueError("Provide at least one reference for each of keep and discard")
    if margin < 0 or len(samples) == 0:
        raise ValueError("Need samples and a nonnegative margin")
    samples = samples / np.maximum(np.linalg.norm(samples, axis=1, keepdims=True), 1e-12)
    refs = refs / np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), 1e-12)
    scores = samples @ refs.T
    keep = scores[:, np.array(labels) == "keep"].max(axis=1)
    discard = scores[:, np.array(labels) == "discard"].max(axis=1)
    votes = ["keep" if k - d > margin else "discard" if d - k > margin else "review"
             for k, d in zip(keep, discard)]
    label = votes[0] if len(set(votes)) == 1 else "review"
    return {"label": label, "sample_labels": votes, "keep_scores": keep.tolist(),
            "discard_scores": discard.tolist(), "nearest_reference": scores.argmax(axis=1).tolist()}


def detect_shots(proxy: Path, frame_count: int, settings: dict) -> list[tuple[int, int]]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector

    video = open_video(str(proxy), backend="opencv")
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector(**settings))
    manager.detect_scenes(video, show_progress=True)
    shots = [(a.get_frames(), b.get_frames()) for a, b in manager.get_scene_list(start_in_scene=True)]
    if not shots:
        shots = [(0, frame_count)]
    if shots[0][0] != 0 or shots[-1][1] != frame_count:
        raise ValueError("Detected shots do not cover the source video")
    return shots


class ImageEncoder:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).to(self.device).eval()
        self.revision = self.model.config._commit_hash

    def encode(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        import torch

        features = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                # Preserve the entire camera composition. The default center
                # crop + CLS token can match a close-up court to a wide court.
                resized = [im.resize((224, 224), Image.Resampling.BICUBIC)
                           for im in images[start:start + batch_size]]
                inputs = self.processor(images=resized, do_resize=False, do_center_crop=False,
                                        return_tensors="pt").to(self.device)
                patches = self.model(**inputs).last_hidden_state[:, 1:]
                grid = patches.transpose(1, 2).reshape(len(resized), -1, 16, 16)
                embeddings = torch.nn.functional.adaptive_avg_pool2d(grid, (4, 4)).flatten(1)
                features.append(torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy())
                print(f"embedded {min(start + batch_size, len(images))}/{len(images)} images", flush=True)
        return np.concatenate(features)


def load_profile(path: Path, video: Path, info: dict, margin: float | None = None) -> dict:
    """Reject accidental reuse of a calibrated profile on a different source."""
    config = json.loads(path.read_text())
    source = config.get("source", {})
    if source.get("filename") and source["filename"] != video.name:
        raise ValueError("Profile belongs to a different video; create a profile for this source")
    if source.get("sha256"):
        with video.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != source["sha256"]:
            raise ValueError("Profile source hash does not match this video")
    margin = config.get("margin", 0.0) if margin is None else margin
    sample_fps = config.get("sample_fps", 2.0)
    cuts = {"adaptive_threshold": 2.0, "min_content_val": 5.0, **config.get("cuts", {})}
    if (not math.isfinite(margin) or margin < 0 or not math.isfinite(sample_fps)
            or not 0 < sample_fps <= info["fps"]):
        raise ValueError("Need a finite nonnegative margin and sample_fps within source fps")
    if set(cuts) != {"adaptive_threshold", "min_content_val"} or any(
            not math.isfinite(v) or v <= 0 for v in cuts.values()):
        raise ValueError("Cut thresholds must be finite and positive")
    refs = config.get("references", [])
    if {r["label"] for r in refs} != {"keep", "discard"}:
        raise ValueError("References must include both keep and discard examples")
    for ref in refs:
        if ("time" in ref) == ("image" in ref):
            raise ValueError("Each reference needs exactly one of time or image")
        if "time" in ref and (not math.isfinite(ref["time"]) or
                not 0 <= round(ref["time"] * info["fps"]) < info["frames"]):
            raise ValueError("Reference timestamp is outside the source video")
    return {**config, "margin": margin, "sample_fps": sample_fps, "cuts": cuts}


def load_references(config: dict, profile: Path, proxy: Path, fps: float, out: Path) -> list[Image.Image]:
    refs = config["references"]
    ref_images = []
    for ref in refs:
        if "image" in ref:
            with Image.open(profile.parent / ref["image"]) as image:
                ref_images.append(image.convert("RGB"))
        else:
            index = round(ref["time"] * fps)
            ref_images.extend(extract_frames(proxy, [index]))
    ref_dir = out / "references"
    ref_dir.mkdir(exist_ok=True)
    for label in ("keep", "discard"):
        for previous in ref_dir.glob(f"[0-9][0-9]_{label}.jpg"):
            previous.unlink()
    for i, (ref, image) in enumerate(zip(refs, ref_images)):
        image.save(ref_dir / f"{i:02d}_{ref['label']}.jpg")
    contact_sheet([(image, f"{i}: {ref['label']} {ref.get('note', '')}")
                   for i, (ref, image) in enumerate(zip(refs, ref_images))], out / "references.jpg")
    return ref_images


def refine_changes(sampled: list[int], frame_label, ensure_frames) -> list[tuple[int, int]]:
    """Bisect observed label changes; sampling/encoding are provided by the caller."""
    intervals = [(a, b) for a, b in zip(sampled, sampled[1:]) if frame_label(a) != frame_label(b)]
    while any(b - a > 1 for a, b in intervals):
        ensure_frames(sorted({(a + b) // 2 for a, b in intervals if b - a > 1}))
        refined = []
        for a, b in intervals:
            if b - a <= 1:
                refined.append((a, b))
            else:
                mid = (a + b) // 2
                if frame_label(a) != frame_label(mid):
                    refined.append((a, mid))
                if frame_label(mid) != frame_label(b):
                    refined.append((mid, b))
        intervals = refined
    return intervals


def run(video: Path, out: Path, device: str | None = None) -> dict:
    """Remove unwanted views from this clip using its bundled recipe."""
    references = PROFILE
    info = source_info(video)
    config = load_profile(references, video, info)
    refs, margin = config["references"], config["margin"]
    labels = [r["label"] for r in refs]
    out.mkdir(parents=True, exist_ok=True)
    proxy = make_proxy(video, out, info)
    ref_images = load_references(config, references, proxy, info["fps"], out)
    reference_images_sha256 = [hashlib.sha256(image.tobytes()).hexdigest() for image in ref_images]

    camera_shots = detect_shots(proxy, info["frames"], config["cuts"])
    # Sample throughout shots as well: dissolves and gradual camera changes can
    # escape a hard-cut detector. This is cheap local inference, not VLM calls.
    sampled = sorted({int(f) for f in np.arange(0, info["frames"], info["fps"] / config["sample_fps"])} |
                     {f for a, b in camera_shots for f in [a, b - 1, *sample_frames(a, b)]})
    images = extract_frames(proxy, sampled)
    encoder = ImageEncoder(device)
    descriptor = "full-frame 224x224; DINOv2 patch features pooled to 4x4 grid"
    old_manifest = out / "shots.json"
    old_vectors = out / "embeddings.npz"
    cache = {}
    ref_vectors = None
    if old_manifest.exists() and old_vectors.exists():
        old = json.loads(old_manifest.read_text())
        source_signature = json.loads((out / "proxy-source.json").read_text())
        if (old.get("source_signature") == source_signature and old.get("references") == refs
                and old.get("reference_images_sha256") == reference_images_sha256
                and old.get("encoder_revision") == encoder.revision and old.get("descriptor") == descriptor):
            with np.load(old_vectors) as saved:
                ref_vectors = saved["references"]
                cache = dict(zip(saved["sample_frames"].tolist(), saved["samples"]))
    if ref_vectors is None:
        ref_vectors = encoder.encode(ref_images)
    missing = [i for i, f in enumerate(sampled) if f not in cache]
    if missing:
        cache.update(zip([sampled[i] for i in missing], encoder.encode([images[i] for i in missing])))
    vectors = np.array([cache[f] for f in sampled])
    lookup = dict(zip(sampled, vectors))
    image_lookup = dict(zip(sampled, images))

    def frame_label(frame):
        return classify(lookup[frame][None], ref_vectors, labels, margin)["label"]

    # Refine observed class changes to adjacent source frames in batched rounds.
    # Unlike cut detection this also separates views within a continuous zoom.
    def ensure_frames(frames):
        middle = sorted(set(frames) - lookup.keys())
        if middle:
            middle_images = extract_frames(proxy, middle)
            missing = [i for i, f in enumerate(middle) if f not in cache]
            if missing:
                cache.update(zip([middle[i] for i in missing], encoder.encode([middle_images[i] for i in missing])))
            lookup.update((f, cache[f]) for f in middle)
            image_lookup.update(zip(middle, middle_images))
    intervals = refine_changes(sampled, frame_label, ensure_frames)
    edges = sorted({0, info["frames"]} | {a for a, _ in camera_shots} | {b for _, b in intervals})
    boundaries = list(zip(edges, edges[1:]))
    sampled = sorted(lookup)
    # Every resulting interval contains a classified sample, including one-frame
    # transitions. Keep the complete partition; never merge across camera cuts.
    indices = [[f for f in sampled if a <= f < b] for a, b in boundaries]
    np.savez_compressed(out / "embeddings.npz", references=ref_vectors,
                        samples=np.array([lookup[f] for f in sampled]), sample_frames=np.array(sampled))
    shots = []
    for i, ((start, end), frames) in enumerate(zip(boundaries, indices)):
        decision = classify(np.array([lookup[f] for f in frames]), ref_vectors, labels, margin)
        shots.append({"id": i, "start_frame": start, "end_frame": end,
                      "start": start / info["fps"], "end": end / info["fps"],
                      "sample_frames": frames, **decision})
    result = {"version": 1, "pipeline": "uso2026_highlights", "video": str(video.resolve()), **info, "encoder": MODEL,
              "encoder_revision": encoder.revision, "margin": margin,
              "descriptor": descriptor,
              "source_signature": json.loads((out / "proxy-source.json").read_text()),
              "view_sample_fps": config["sample_fps"], "camera_shots": camera_shots,
              "cut_detector": {"name": "AdaptiveDetector", **config["cuts"]},
              "references": refs, "reference_config_sha256": hashlib.sha256(references.read_bytes()).hexdigest(),
              "reference_images_sha256": reference_images_sha256,
              "shots": shots}
    result["summary"] = {label: {"shots": sum(s["label"] == label for s in shots),
                                  "seconds": sum(s["end"] - s["start"] for s in shots if s["label"] == label)}
                         for label in ("keep", "discard", "review")}
    validate_manifest(result)
    (out / "shots.json").write_text(json.dumps(result, indent=2))
    for label in ("keep", "discard", "review"):
        for old_sheet in out.glob(f"{label}_[0-9][0-9].jpg"):
            old_sheet.unlink()
        items = []
        for shot in shots:
            n = len(shot["sample_frames"])
            if shot["label"] == label:
                frame = shot["sample_frames"][n // 2]
                items.append((image_lookup[frame], f"#{shot['id']} {shot['start']:.2f}-{shot['end']:.2f}s {label}"))
        for page in range(0, len(items), 32):
            contact_sheet(items[page:page + 32], out / f"{label}_{page // 32 + 1:02d}.jpg")
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result
