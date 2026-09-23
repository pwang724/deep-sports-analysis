# Preprocessing

The job is to cut away unwanted footage before analysis. Each video owns its
preprocessing implementation. A different video can use a different model,
manual time ranges, or any other method that produces the required intervals.

```
src/dsa/preprocess/
  manifest.py             shared output contract
  media.py                shared video I/O and exports
  uso2026_highlights/      this video's complete cleanup recipe
    pipeline.py
    profile.json
    __main__.py
    README.md
```

There is no global selection model or pipeline registry. For another video,
create another sibling directory with its own code, configuration and command.
The current implementation is described in the
[USO highlights recipe](../src/dsa/preprocess/uso2026_highlights/README.md).

The [desktop tennis recipe](../src/dsa/preprocess/desktop_tennis/README.md) is
separate: it normalizes Peter's Richard/Dylan phone recordings and exports
unscored activity from source-matched cut lists, independently of point labels.
Its audio-assisted proposals require visual review. `play_shots.json` supplies
the resulting activity windows to tracking. A separate optional workflow adds
eleven/match scoring later; its point annotations remain incomplete drafts.

## Output consumed by analysis

`shots.json` identifies the original `video`, `fps`, `frames`, and an ordered
list of intervals with integer `start_frame`, integer `end_frame`, and `label`.
Frame bounds are half-open: `[start_frame, end_frame)`. The intervals cover the
source exactly once, with no gaps or overlap. Only `keep` intervals are passed
to tracking; `discard` and `review` are excluded. Extra recipe metadata is allowed.
Use `validate_manifest` to check the format. Include `fps_fraction` when known
for exact-rate exports.

Preserve camera-cut boundaries even between neighboring kept intervals, so
tracking can reset there. A recipe must choose useful footage and boundaries;
format validation alone cannot establish that those choices are correct.

Optional `kept.mp4` and `unkept.mp4` are concatenated review videos. Analysis
reads the original video with the interval manifest, preserving source times.
Replacing a video's cleanup method does not require changes to the tracker.
