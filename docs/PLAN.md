# Plan

Tennis video analysis from unfixed, handheld or broadcast footage. Three
separate vision problems feed one shared coordinate system, and everything
analytical is computed downstream in plain NumPy. Model choices are backed by
[RESEARCH.md](RESEARCH.md). Data in [DATASETS.md](DATASETS.md). This modular
pipeline is also the labeler for the single trained model in
[training/](training/README.md).

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
| Ball, first attempt | RF-DETR-S/M detection at 1024 px, fine-tuned on RacketVision | One tool, one training loop. Best small-object AP among real-time detectors. Decide by measurement, not literature. |
| Ball, if recall is poor | RF-DETR with 3 stacked frames (9 input channels), then WASB (NTT, MIT, tennis weights) | The heatmap trackers win because they see motion across 3 frames, not because of architecture. Try giving RF-DETR the same input first. WASB is the only open tracker trained on footage with camera motion. |
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

1. **Player pose.** Done: RF-DETR Medium boxes, ByteTrack ids, ViTPose joints,
   on Modal per segment. See [DECISIONS.md](DECISIONS.md) 1-3.
2. **Who is who.** One VLM call per batch of new tracks names the players and
   drops everyone else; ViTPose then runs on players only. Decided, not built:
   [DECISIONS.md](DECISIONS.md) 4. Needed before any per-player analytics.
3. **Court detection + homography.** Fine-tune on TennisCourtDetector with
   heavy perspective augmentation. Per frame: detect points, drop frames with
   fewer than 4 confident points, track corners with Lucas-Kanade between
   detections, refit the homography when reprojection error grows or a cut is
   detected. Map player feet onto a top-down court. This already gives footwork
   and positioning.
4. **Ball.** Fine-tune RF-DETR on RacketVision tennis at 1024 px. Measure
   ball recall on the held-out split against WASB's published F1 of 95.6. If
   the gap is under ~10 points, keep RF-DETR. If larger, try stacked-frame
   input, then WASB. Either way: stabilize frames with the tracked homography,
   smooth, fill gaps, fit parabolas between bounces.
5. **Analytics.** Speed, bounce location, contact point, shot classification,
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
