"""Prepare phone footage, review point windows, and export scored point-only videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotations import load, write_outputs
from .media import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Normalize source and initialize its editable point log")
    prepare.add_argument("video", type=Path)
    prepare.add_argument("--opponent", required=True, choices=("richard", "dylan"))
    prepare.add_argument("--out", type=Path, required=True)
    play = commands.add_parser("export-play", help="Export unscored activity from cuts.json; no point labels required")
    play.add_argument("--out", type=Path, required=True)
    play.add_argument("--destination", type=Path)
    detect = commands.add_parser("detect-play", help="Propose unscored activity windows for visual review")
    detect.add_argument("--out", type=Path, required=True)
    detect.add_argument("--start", type=float, default=0)
    detect.add_argument("--end", type=float)
    for name in ("build", "review", "export"):
        sub = commands.add_parser(name)
        sub.add_argument("--out", type=Path, required=True)
        if name == "review":
            sub.add_argument("--port", type=int, default=8765)
        elif name == "export":
            sub.add_argument("--segment", help="Export one segment ID; otherwise export all")
            sub.add_argument("--draft", action="store_true", help="Visible DRAFT badge; unknown scores remain ?")
    args = parser.parse_args()
    if args.command == "detect-play":
        from .activity import propose_play
        propose_play(args.out, args.start, args.end)
    elif args.command == "export-play":
        from .cuts import export_play
        export_play(args.out, args.destination)
    elif args.command == "prepare":
        source = normalize(args.video, args.out)
        path = args.out / "annotations.json"
        if not path.exists():
            log = {"version": 1, "source_sha256": source["original_sha256"],
                   "players": ["Peter", args.opponent.title()],
                   "identities": {"Richard": "Shirtless during later play; verify earlier clothing by continuity."
                                  } if args.opponent == "richard" else {"Dylan": "Black shirt and cap"},
                   "rules": {"eleven": "first to 11, win by two", "match": "standard advantage"},
                   "editing": "Retain only point windows, including missed first serves. "
                              "Exclude aborted tosses, dead time, and ball collection using separate clips within a point.",
                   "segments": []}
            profile_path = Path(__file__).with_name("profiles") / f"{args.opponent}.json"
            if profile_path.exists():
                profile = json.loads(profile_path.read_text())
                if (profile["source_filename"] == args.video.name
                        and profile["source_sha256"] == source["original_sha256"]):
                    log = profile["log"]
            path.write_text(json.dumps(log, indent=2))
        log, source = load(args.out)
        write_outputs(log, source, args.out)
    elif args.command == "review":
        from .review import serve
        serve(args.out, args.port)
    else:
        log, source = load(args.out)
        write_outputs(log, source, args.out)
        if args.command == "export":
            from .render import render_segment
            selected = [s for s in log["segments"] if args.segment is None or s["id"] == args.segment]
            if not selected:
                parser.error("No matching segments; add point annotations first")
            for segment in selected:
                render_segment(log, source, segment, args.out, draft=args.draft)


if __name__ == "__main__":
    main()
