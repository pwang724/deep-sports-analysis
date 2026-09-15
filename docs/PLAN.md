# Plan

Tennis video analysis from unfixed, handheld or broadcast footage. Three
separate vision problems feed one shared coordinate system, and everything
analytical is computed downstream in plain NumPy. Model choices are backed by
[RESEARCH.md](RESEARCH.md).

```
video frame
    |
    +--> COURT     14-pt keypoint model  -> homography (pixels -> meters)
    |              + corner tracking        re-detect on cut or drift
    |
    +--> PLAYERS   pose model            -> 17 joints per player per frame
    |
    +--> BALL      multi-frame heatmap   -> (x, y) per frame -> smoothed track
    |              tracker
    v
everything in court coordinates over time
    -> speed, shot type, footwork, contact point, bounce location
```

Reference for the pipeline shape and the analysis-heavy style:
[jeremyipark/vision-demos](https://github.com/jeremyipark/vision-demos).

## Models

| Problem | Pick | Why |
|---|---|---|
| Players | ViTPose-Plus-Large via the VLM Run gateway | Highest COCO accuracy family (ViTPose-G 81.1 AP, nothing has beaten it). No weights, no GPU, cached responses. |
| Players, if feet matter | RTMW-l via `rtmlib` | 133 keypoints incl. 6 foot points, Apache, runs on CPU. |
| Players, real-time local | RF-DETR Keypoint | Best single-stage: 71.8 AP at 9.7 ms on T4, Apache. Still preview. |
| Court | TennisCourtDetector dataset (8,841 frames, 14 pts) fine-tuned into either a heatmap CNN or RF-DETR Keypoint with a 14-point skeleton | Field standard, 3.8 px error. Camera moves so corners are re-found per frame. Add perspective augmentation and a few hundred self-labelled handheld frames. |
| Ball | WASB (NTT, MIT, tennis weights) | Multi-frame heatmap trackers beat box detectors in every controlled comparison (F1 95.6 vs 47 for single-frame on tennis). Only open tracker trained on footage with camera motion. |
| Ball, fallback | RF-DETR-S/M detection at 1024 px with tiling | If the ball is large in frame or WASB fails on handheld. Best small-object AP among real-time detectors. |
| 3D pose, later | TRAM + WHAM (MIT) or Fast SAM 3D Body | CalTennis benchmark says joint angles are reliable but monocular depth and foot contact are not. Get court position from the homography, never from 3D pose. |

Rejected: TrackNetV5 (best numbers, no weights, proprietary). TrackNetV3/V4
for handheld (assume a static camera). Manual corner clicks and Hough lines
(assume a fixed camera). YOLO family (AGPL).

## Datasets

- **RacketVision** (HF, MIT): 150k tennis broadcast frames with ball and racket
  keypoints. Primary ball training data.
- **TennisCourtDetector** (Google Drive): 8,841 frames, 14 court points.
- **TrackNet tennis**: 20k frames with ball and hit/bounce events.
- Own handheld clips: self-label a few thousand frames. Label the ball at the
  blur-streak center (BlurBall convention). No public handheld tennis dataset
  exists.

## Order of work

1. **Player pose.** Clone the vision-demos structure. Get ViTPose running on a
   clip through the gateway, render the overlay, dump joints to JSON.
2. **Court detection + homography.** Fine-tune on TennisCourtDetector with
   heavy perspective augmentation. Per frame: detect points, drop frames with
   fewer than 4 confident points, track corners with Lucas-Kanade between
   detections, refit the homography when reprojection error grows or a cut is
   detected. Map player feet onto a top-down court. This already gives footwork
   and positioning.
3. **Ball.** Stabilize frames with the tracked homography, run WASB, smooth,
   fill gaps, fit parabolas between bounces. Fine-tune on RacketVision plus
   own labels. Fall back to RF-DETR if recall is poor.
4. **Analytics.** Speed, bounce location, contact point, shot classification,
   all in court coordinates.

## Constraints

- The camera moves, so every downstream point is mapped through *that frame's*
  homography, never a global one. Court detection runs first and everything
  depends on it.
- 2D only at first. Joint angles are projected, not anatomical. Ball height is
  not recoverable from one view.
- Pose is per-frame with no memory. Left/right can swap when legs cross.
  Smooth signals before computing anything from them.
- Broadcast datasets have open labels but broadcaster-owned footage. Fine for
  training, not for redistribution.
