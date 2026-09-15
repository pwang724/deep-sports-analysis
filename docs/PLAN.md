# Plan

Tennis video analysis from unfixed, handheld or broadcast footage. Three
separate vision problems feed one shared coordinate system, and everything
analytical is computed downstream in plain NumPy.

```
video frame
    |
    +--> PLAYERS   pose model       -> 17 joints per player per frame
    |
    +--> COURT     keypoint model   -> 14 line intersections -> homography
    |                                  (pixels -> meters on the court)
    |
    +--> BALL      small-object     -> (x, y) per frame -> smoothed track
    |              tracker
    v
everything in court coordinates over time
    -> speed, shot type, footwork, contact point, bounce location
```

Reference for the pipeline shape and the analysis-heavy style:
[jeremyipark/vision-demos](https://github.com/jeremyipark/vision-demos).

## Models

| Problem | Model | Why |
|---|---|---|
| Players | ViTPose-Plus-Large via the VLM Run gateway | Most accurate 17-joint pose on recorded clips. No weights, no GPU, cached responses. |
| Court | RF-DETR Keypoint, fine-tuned on 14 court points | Camera moves, so corners must be re-found every frame across varied footage. Custom skeleton is exactly what RF-DETR is built for. Apache 2.0. |
| Ball | RF-DETR detection at high resolution first | One tool we already know. Expected to struggle on a 5-10 px blurry ball. |
| Ball, fallback | TrackNet | Purpose-built for tennis and badminton. Takes 3 consecutive frames, uses motion to find the ball. Switch here if RF-DETR misses too often. |

Rejected for court: manual corner clicks and Hough lines. Both assume a fixed
or nearly fixed camera, which is not our footage.

## Order of work

1. **Player pose.** Clone the vision-demos structure. Get ViTPose running on a
   clip through the gateway, render the overlay, dump joints to JSON.
2. **Court detection + homography.** Fine-tune RF-DETR Keypoint on a public
   tennis-court keypoint dataset (14 points, several thousand broadcast frames
   exist on Roboflow Universe). Add a few hundred of our own frames only if our
   angles look different. Per frame: detect points, drop frames with fewer than
   4 confident points, smooth over time, fit a homography. Map player feet onto
   a top-down court diagram. This already gives footwork and positioning.
3. **Ball.** Fine-tune RF-DETR detection on a public tennis-ball set. Smooth,
   fill gaps, fit parabolas between bounces. If recall is poor, move to
   TrackNet.
4. **Analytics.** Speed, bounce location, contact point, shot classification,
   all in court coordinates.

## Constraints

- The camera moves, so every downstream point is mapped through *that frame's*
  homography, never a global one. Court detection runs first and everything
  depends on it.
- 2D only. Joint angles are projected, not anatomical. Ball height is not
  recoverable from one view.
- Pose is per-frame with no memory. Left/right can swap when legs cross.
  Smooth signals before computing anything from them.
