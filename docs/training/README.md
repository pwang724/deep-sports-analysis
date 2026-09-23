# Training plan: one model, one pass

Goal: replace per-video preprocessing and the chain of separate models with a
single network that reads raw tennis video (and its audio) once and returns,
per frame or per event:

- whether the frame is a usable court view, and whether a point is in play
- player boxes, track identity and 17 joints
- the 14 court keypoints (hence the homography)
- hit and bounce events, the hitter, and the stroke type at each hit
- the ball position, from early-fused neighbouring frames
- the contact point (hit frame + wrist/ball position, mapped to the court)

WASB supplies the ball labels and stays as the fallback if the ball head
falls short. Naming players and ball height are out of scope.

The current modular pipeline is not thrown away. It becomes the **labeler**:
the best available model for each task runs over large amounts of public and
own footage, and the single model is trained on the merged, partial labels.

| File | Contents |
|---|---|
| [MODEL.md](MODEL.md) | architecture, early fusion, backbone choice, heads, losses |
| [DATA.md](DATA.md) | ground truth by task, labelers (incl. Astra via Codex), label format, gold set |
| [PROGRESS.md](PROGRESS.md) | ordered checklist |

## Phases

Each phase ends with a measurement, not a feeling. Numbers go in a
`RESULTS-training.md` next to [RESULTS-pose.md](../RESULTS-pose.md).

0. **Choose a labeler per task on public ground truth.** For every label type
   (joints, person boxes, court, ball, events, stroke, view,
   in play), score Astra and the specialist labeler (ViTPose, WASB, court
   model, event spotter) on a 100-200 item sample of an already-labelled
   dataset. Stop early where a labeler is clearly failing. Each task's
   result, a score table plus the worst misses, goes to Peter to confirm
   which labeler (or agreement of both) is used. Joints, person boxes,
   coarse stroke type, view and in play can run on data already local.
1. **Transfer check on our domain.** A small hand-checked gold set that never
   enters training: two of our own sessions and two or three YouTube
   recordings from different courts and camera heights. It confirms the
   phase-0 choices hold on self-recorded footage (public sets are mostly
   broadcast and may be in Astra's training data), and scores the model later.
2. **Footage.** Collect self-recorded tennis from YouTube and similar, 100
   hours to start. Filter out non-court footage automatically.
3. **Label factory.** Run every labeler over all footage into one label format
   with per-label confidence ([DATA.md](DATA.md)). Spot-check samples by hand.
4. **Model v0, per frame.** Trunk + view, in-play, player, court heads, no
   temporal module. Compare head by head against the labelers on the gold set.
5. **Model v1, temporal.** Add the temporal module, audio, event and stroke
   heads, and contact point.
6. **Self-training.** The model relabels the footage; Astra and hand review
   look only where the model and labelers disagree or confidence is low;
   retrain. Repeat while gold-set numbers improve.

## Done when

On the gold set, with one forward pass per window and no per-video settings:

- view and in-play agree with hand labels at least as well as the current
  recipes ([PREPROCESSING.md](../PREPROCESSING.md)) do on their own clips
- player recall, far player included, matches RF-DETR Medium; joint OKS is
  within 0.02 of ViTPose-Plus-Huge (0.824)
- court keypoint error is at or below the court labeler's
- hit events within 2 frames and stroke type accuracy at or above the
  labelers', which set the ceiling until self-training beats them

Until then the modular pipeline stays the production path.
