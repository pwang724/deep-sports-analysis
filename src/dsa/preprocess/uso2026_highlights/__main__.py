"""Cut unwanted footage from the USO 2026 highlights source."""
import argparse
import json
from pathlib import Path

from dsa.preprocess.manifest import validate_manifest
from dsa.preprocess.media import export_reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/preprocess/uso2026_highlights"))
    parser.add_argument("--device")
    parser.add_argument("--export", action="store_true", help="Write kept/unkept preview videos")
    parser.add_argument("--export-only", action="store_true", help="Export a saved shots.json without inference")
    args = parser.parse_args()
    if args.export_only:
        manifest = json.loads((args.out / "shots.json").read_text())
    else:
        from .pipeline import run
        manifest = run(args.video, args.out, args.device)
    validate_manifest(manifest, args.video)
    if args.export or args.export_only:
        export_reviews(manifest, args.out)


if __name__ == "__main__":
    main()
