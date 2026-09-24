"""Where everything under data/ lives. Import paths from here; never hard-code them.

    data/                       symlink to /Volumes/PW_SSD/deep-sports-analysis/data
      sources/<name>/           public datasets exactly as released (see dsa.data.sources)
      code/<name>/              third-party repos we run (WASB, RacketVision, TennisCourtDetector)
      videos/<collection>/      footage: e2e_spot/, f3set/ (whole matches behind their events),
                                broadcast/ (three hand-picked matches), youtube/
      raw/                      our own recordings; preprocess outputs store these paths, so they stay here
      labels/<name>/            every source converted to the one label format (dsa.data.schema)
      gold/                     human-reviewed labels on sampled frames
      scratch/                  throwaway images
"""
from pathlib import Path

DATA = Path("data")
SOURCES = DATA / "sources"
CODE = DATA / "code"
VIDEOS = DATA / "videos"
RAW = DATA / "raw"
LABELS = DATA / "labels"
GOLD = DATA / "gold"
SCRATCH = DATA / "scratch"
MODELS = Path("models")
