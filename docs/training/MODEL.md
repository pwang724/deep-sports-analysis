# Model

Tennis-only first version. A sport-general design is recorded at the end
and deliberately deferred.

## Shape

```
for each frame t (30 per window):
  frames t-1, t, t+1 ──> early fusion: 9 channels, 16 px patches ──> 3,600 tokens
      ──> transformer 1 (RF-DETR Medium backbone, pretrained) ──┬─> court head: 14 points + visibility
                                                                ├─> DETR decoder: player tokens -> box, 30 keypoints
                                                                ├─> ball head: x, y, visible -> ball token
                                                                └─> shrink: 256 scene tokens

across the window:
  player + ball + scene tokens (30 frames) + audio tokens (1 per frame)
      ──> transformer 2 (attends across frames) ──> event heads:
            view ok, in play, hit / bounce / serve, hitter, stroke type, track links
```

Transformer 1 sees one frame at full detail (plus its two neighbours through
fusion). Transformer 2 sees all 30 frames, but only as summaries: about 260
tokens per frame, 7,800 per window, instead of 108,000. Per-frame features are
computed once and reused by every overlapping window.

Contact point is computed, not predicted: the hit frame from the event head,
the hitter's racket head (or ball) at that frame, mapped to court coordinates
through that frame's homography.

## Early fusion

The only backbone weight that changes shape is the patch embedding,
`Conv2d(3, 384, 16, 16)` in RF-DETR Medium. It becomes `Conv2d(9, 384, 16,
16)`: frames t-1, t, t+1 stacked per pixel. The middle frame's slice keeps
the pretrained weights; the neighbours start at zero, so step 0 is exactly
the pretrained model. Token count and every later layer are unchanged. The
learned weights can act as frame differencing, which is what finds a small,
blurred, moving ball (TrackNet's trick).

Requirements that follow:

- Frames resampled to a fixed 30 fps; t±1 must mean the same time step
  everywhere.
- At clip edges and camera cuts, repeat frame t instead of crossing the cut.
- Identical augmentation (crop, flip, scale, colour) on all three frames.
- Single-image datasets (COCO, court sets) feed the same frame three times.
- Pixel mean / std repeated to 9 channels; higher learning rate on the new
  layer than on the pretrained backbone.
- RF-DETR assumes 3 channels in input checks, preprocessing and export: wrap
  or patch those rather than using the stock training entry point.

## Backbone

RF-DETR is a DINOv2 backbone (ViT-S width 384, 16 px patches in Medium) with
a deformable DETR decoder, trained end to end on COCO. Start there: it is
measured on our footage (100% recall on both players, DECISIONS 1), the
detection decoder comes with it, and it is Apache.

First ablation after v1 works: a DINOv3 ViT-B backbone with the same fusion,
decoder and heads. DINOv3 has better high-resolution dense features (court
lines, far player, ball) but no COCO-trained decoder at this size and a custom
license. Keep whichever scores better on the gold set at acceptable speed.

Rejected: video encoders (V-JEPA 2, VideoMAE) lose the ball and far player at
their 224-384 px inputs and cost too much at 720p; SAM 3 lost the far player
and switched identities (DECISIONS 2); a VLM emitting coordinates is too slow
and imprecise and is used as a labeler instead.

## Known risk: joints from a single stage

On our labelled set, single-stage RF-DETR Keypoint scored 0.773 OKS against
0.824 for ViTPose top-down; the gap is in wrists and elbows (RESULTS-pose).
Mitigation: a refinement head that crops each player's region from the
backbone's features (RoIAlign) and predicts joint heatmaps there. Still one
forward pass. Train at a higher resolution than COCO defaults.

## Keypoints per person

30 points on each person token: the 17 COCO body joints, neck, head top, six
foot points (big toe, small toe, heel, each side) and five racket points.
The racket is part of its holder's skeleton, as in Hawk-Eye's SkeleTRACK, so
"whose racket" needs no matching step and the hitter's racket at contact is
read off the token the event head names. Each point's loss counts only where
that point is labelled: COCO and Tennis Player Actions supply body joints,
Halpe / COCO-WholeBody supply head and feet, RacketVision supplies rackets.
New points get their own OKS tolerances, looser than wrists for the racket.
Hands (21 points each) are left out: too small on the far player.

## Heads, losses, masking

| Head | Output | Loss |
|---|---|---|
| people | box, 30 keypoints + visibility (every person; players = hitters) | DETR set loss; OKS / heatmap loss on keypoints |
| court | 14 heatmaps + visibility | focal heatmap loss |
| ball | heatmap + visibility, per frame | focal heatmap loss |
| view, in play | per-frame probability | BCE |
| events | per-frame hit / bounce / serve, hitter side | focal loss with a few-frame tolerance (E2E-Spot style) |
| stroke | class at each hit (forehand, backhand, serve, volley, slice, overhead, ...) | cross-entropy |
| identity | player token links across frames | contrastive / matching loss (MOTR style) |

Every sample carries only the labels it has; a head's loss counts only where
its label exists, weighted by label confidence ([DATA.md](DATA.md)). Mix
batches across sources and balance by head, not frame count. A window of one
frame trains only the per-frame heads.

Augment for handheld footage: perspective warps, rotation, crop and scale
jitter, blur, compression, colour and exposure shifts.

The ball head is judged against WASB, which also supplies its labels. If it
falls short, WASB's track replaces the ball token at inference.

## Deferred: sport-general version

Early fusion and the fixed pooled summary bake in tennis assumptions: a
still-ish camera, one motion scale at 30 fps, objects big enough to survive
pooling. A version meant to transfer across sports would instead use a slow
heavy pathway plus a fast light one (SlowFast), temporal attention with
learned offsets rather than stacked pixels, and persistent object tokens
rather than a pooled grid, with field templates and action names given as
inputs. Not built now: harder to train, and tennis comes first.
