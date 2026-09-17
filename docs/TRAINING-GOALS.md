# Future training goals

## Joint view selection and player detection

Train one shared visual encoder with two outputs: player bounding boxes and
the probability that the frame shows the desired aerial court view. Start
from RF-DETR, retaining its detection output and adding a frame classification
head. This is a proposed extension, not a capability of the current checkpoint.

Supervision consists of aerial/other frame labels and boxes for actual players;
ball kids, officials, and spectators are background for the player detector.
VLM-assisted annotation can bootstrap a reviewed training set. Evaluate on
held-out matches and broadcasts, not adjacent frames from the training shots.
Measure player recall (especially the far player), false player detections,
view-selection precision/recall, and end-to-end latency. Compare against the
reference-based preprocessing baseline before replacing it.

The intended benefit is learned scene and role distinctions with one expensive
encoder pass and no routine VLM calls. It does not solve A/B identity, camera
cut detection, or tracker identity switches. It also performs detection on
rejected views, so an end-to-end speed improvement is not guaranteed.

For now, use a small keep/discard reference set and a frozen image encoder.
Preserve original timestamps and camera-shot boundaries for subsequent tracking.
