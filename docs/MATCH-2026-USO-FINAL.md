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

## Data layers, from cheapest to hardest

```
layer                       what it gives                 where            status
1  aggregate box score      aces, serve %, BP, W/UE       ATP, tennis.com  free, now
2  point-by-point score     every point, who served,      usopen.org IBM,  free, now
                            score state, momentum curve   Sofascore
3  shot-by-shot log         each shot: type, direction,   Match Charting   volunteer, not yet
                            depth, rally length, outcome  Project (MCP)    charted (as of May commit)
4  tracking data            ball + player xy per frame,   Hawk-Eye via     CLOSED. Not public.
                            speeds, contact points        ATP Tennis IQ,
                                                          IBM, TDI
5  video                    the broadcast                 ESPN / ESPN      subscription
                                                          Unlimited replay
```

Momentum and tactical insight needs layers 2 and 3. Layer 4 is what this
repo's CV pipeline exists to reconstruct from layer 5.

## Layer 1: aggregate stats (have it)

- ATP report: https://www.atptour.com/en/news/zverev-shelton-us-open-2026-final-report
- tennis.com match page (serve / return / BP table): https://www.tennis.com/tournaments/us-open/matches/a-zverev-vs-b-shelton-2026-09-13
- usopen.org recap: https://www.usopen.org/en_US/news/articles/2026-09-13/alexander_zverev_defeats_ben_shelton_to_win_2026_us_open.html

Enough for a summary, useless for momentum.

## Layer 2: point-by-point score (get this first)

Every point in order with server, score before the point, and outcome. From
this you compute: momentum curves (rolling point win %), win probability per
point, leverage of each point, break-point conversion timeline, serve
dominance by set, runs of consecutive points, tiebreak sequence.

Sources:
- **usopen.org IBM match page.** The official site keeps a per-match page
  with the IBM point-by-point feed and "Match Insights" (win-probability
  swings, key points). Find it from the draws page for the men's final. Data
  loads from a JSON endpoint behind the page; capture it from the browser
  network tab.
- **Sofascore / Flashscore.** Both show point-by-point for Slam matches
  including serve speed on some points. Sofascore has an unofficial JSON API
  (`api.sofascore.com/api/v1/event/<id>/point-by-point`). Scrapeable, not
  licensed.
- **Sackmann tennis_slam_pointbypoint.** Jeff Sackmann republishes Slam
  point-by-point CSVs (score, server, rally length, serve speed where IBM
  exposed it). Updated in batches, usually after the tournament. Check
  https://github.com/JeffSackmann/tennis_slam_pointbypoint for a
  `2026-usopen-points.csv`. CC BY-NC-SA.

## Layer 3: shot-by-shot (the real tactical layer)

The Match Charting Project format records every shot: serve direction, return
type and depth, each rally shot's stroke and direction, how the point ended.
With it you get: serve placement patterns by score state, return depth,
backhand-cross vs down-the-line ratios, net approaches, who controlled
rallies of each length, and how any of those shifted after the set-2 tiebreak.

- Repo: https://github.com/JeffSackmann/tennis_MatchChartingProject
  (CC BY-NC-SA). Last commit May 25 2026. **The final is not charted yet.**
  Slam finals are almost always charted by volunteers within a few weeks.
  Watch `charting-m-matches.csv` for a `20260913-M-US_Open-F` row.
- Tennis Abstract renders charted matches at
  `tennisabstract.com/charting/20260913-M-US_Open-F-Alexander_Zverev-Ben_Shelton.html`
  once it lands. Currently 404.
- **Or chart it yourself.** The MCP instructions doc defines the notation.
  A 4-set match is ~4 hours of work with the video. Submitting it back is how
  the project grows.

## Layer 4: tracking data (closed)

Hawk-Eye ball and player positions at every frame, serve speed and placement,
contact heights, player distance covered. This exists for the match. It is
owned by the USTA / Sony and licensed to IBM (fan features), ATP Tennis IQ
(players and coaches only), and TDI / Sportradar (betting and media). None of
it is downloadable. IBM's site surfaces fragments: serve speed per point,
"Match Insights" summaries, sometimes rally length.

The only way to get positional data independently is to run this repo's
pipeline on the broadcast. That is the project.

## Layer 5: video

- **Full match:** ESPN / ESPN Unlimited on-demand replay (US subscription).
  The broadcast is a fixed elevated end-on camera for most points, which is
  the easy case for court detection and matches every public dataset.
- **Highlights:** usopen.org and the US Open YouTube channel post 10-15 minute
  extended highlights within a day, free.
- Recording the replay for personal analysis is a rights question. Do not
  redistribute frames.

## Suggested first analysis, no CV needed

1. Pull the point-by-point from usopen.org or Sofascore.
2. Compute a win-probability curve using a standard tennis Markov model
   with each player's in-match serve-win percentages.
3. Find the top 10 leverage points (largest win-probability swing). The
   set-2 tiebreak and the set-4 opening break should dominate.
4. Rolling 20-point serve-win % per player to show Shelton's collapse in set
   1 and recovery in set 3 (22 of last 24 service points).
5. When MCP charting lands, layer on rally-length buckets and shot direction
   by score state to explain why the long rallies went to Zverev.

Then the CV pipeline adds what none of the above has: where each player
stood, contact points, movement, and court coverage over time.
