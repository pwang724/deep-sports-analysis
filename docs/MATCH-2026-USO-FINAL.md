# Target match: 2026 US Open final, Zverev d. Shelton

Zverev d. Shelton 6-3, 7-6(2), 5-7, 6-2. Sunday Sep 13 2026, Arthur Ashe,
3h33m. Zverev now 6-0 head to head.

Headline numbers from reports: Shelton 47 winners / 47 unforced errors, 10
aces, 4 double faults (two on break points in set 1). Zverev 9 aces, 6 double
faults, 84% first-serve points won, 59% second, 4/6 break points. Shelton made
11 unforced errors in set 1 to Zverev's 3. Zverev won 12 of the last 14
points. Zverev won nearly all rallies over 9 shots. Turning points named by
reporters: the set-2 tiebreak (Zverev 5-0 up), and the break to open set 4 on
a 19-ball rally.

## Goal

All analysis derived from the broadcast video. Where each player's errors
come from on the court, which strokes lose rallies, serve and return
positions by score state, movement and recovery, and how it shifted over the
match. No Hawk-Eye, no charting project. Video in, tactical table out.

## The one input: video

- **ESPN Unlimited** has the full-match on-demand replay. The only complete
  source. Screen-record for personal analysis. Fubo free trial with DVR is
  the alternative.
- **usopen.org / US Open YouTube** post 10-15 min extended highlights, free.
  Good for pipeline development, not for match-level stats.
- Do not redistribute frames.

What the broadcast gives: ~3.5 h, ~380k frames at 30 fps, of which 15-20% is
live play. Main camera is fixed, elevated, end-on behind the baseline, the
easy case and what every public dataset was built on. Replays, crowd shots,
close-ups are camera cuts to detect and drop. The scoreboard graphic is on
screen throughout and is free ground truth.

## Derivation pipeline

```
broadcast video
    |
    +--> cut detection + rally segmentation     keep only end-on live play
    |
    +--> scoreboard OCR                         point winner, score state, server
    |                                           (free labels, validates segmentation)
    +--> court keypoints -> homography          pixels -> court meters, per rally
    |
    +--> players: box + track + pose            court position every frame,
    |                                           stroke type from pose
    +--> ball track                             contact frames, bounce xy,
    |                                           rally length, who hit last
    v
per-shot table:
  player | shot # | stroke | contact (x,y m) | bounce (x,y m)
  | rally length | point outcome | score state
```

## Questions that table answers

- Error origin by court zone and stroke: contact position of each
  point-ending shot, split forehand / backhand / slice / volley.
- Which strokes each player loses rallies on, by rally-length bucket.
- Serve placement and return contact position by score state.
- Distance covered, recovery position after each shot, how far a player was
  pushed wide before the error.
- All of the above before vs after the set-2 tiebreak.

## Build order

1. Video. Extract frames, detect cuts, segment rallies. No model needed.
2. Scoreboard OCR. Point outcomes and server for free.
3. Court keypoints + homography. Broadcast angle, existing 8,841-frame model
   works nearly as is.
4. Players: detect, track, pose. Court position and stroke type. This alone
   yields contact positions and stroke-level error maps, most of the
   tactical story, timing the swing from pose.
5. Ball. Adds bounce locations, rally length, exact contact frames. Hardest
   piece, and 1-4 already deliver "which shot from where" without it.

Scaffold candidate: HarshTomar1234/Tennis-Vision (MIT) does roughly 1, 3, 4,
5 on broadcast footage.

## Reference numbers to sanity-check against

From reports: Shelton 47 W / 47 UE, 10 aces, 4 DF. Zverev 9 aces, 6 DF, 84%
first-serve points won, 4/6 BP. Zverev won nearly all rallies over 9 shots.
Shelton won 22 of his last 24 service points. If the pipeline's counts land
near these, the segmentation and outcome labelling are right.

## Other data that exists but is not the goal

- Point-by-point score: usopen.org IBM feed, Sofascore, Sackmann
  tennis_slam_pointbypoint. Useful only to cross-check OCR.
- Match Charting Project shot-by-shot: not charted yet (last commit May
  2026). Would be a validation set for the pipeline's stroke labels once it
  lands.
- Hawk-Eye tracking: closed, licensed to IBM / ATP Tennis IQ / TDI.
