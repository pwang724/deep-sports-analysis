"""Deterministic scoring from point winners; no score inference from camera motion."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Score:
    kind: str
    points: list[int] = field(default_factory=lambda: [0, 0])
    games: list[int] = field(default_factory=lambda: [0, 0])
    sets: list[int] = field(default_factory=lambda: [0, 0])
    completed_sets: list[list[int]] = field(default_factory=list)
    winner: int | None = None
    known: bool = True

    def __post_init__(self):
        if self.kind not in {"eleven", "match"}:
            raise ValueError("Scoring kind must be eleven or match")
        for pair in (self.points, self.games, self.sets):
            if len(pair) != 2 or any(type(x) is not int or x < 0 for x in pair):
                raise ValueError("Scores must be pairs of nonnegative integers")

    @property
    def tiebreak(self) -> bool:
        return self.kind == "match" and self.games == [6, 6]

    def award(self, player: int | None) -> None:
        if player is not None and (type(player) is not int or player not in (0, 1)):
            raise ValueError("Point winner must be player 0, player 1, or null")
        if self.winner is not None:
            raise ValueError("An eleven is already finished; start a separate segment")
        if player is None:
            self.known = False
        if not self.known:
            return
        other = 1 - player
        self.points[player] += 1
        target = 11 if self.kind == "eleven" else 7 if self.tiebreak else 4
        if self.points[player] < target or self.points[player] - self.points[other] < 2:
            return
        if self.kind == "eleven":
            self.winner = player
            return
        self.games[player] += 1
        self.points = [0, 0]
        if self.games[player] >= 6 and (self.games[player] - self.games[other] >= 2
                                        or self.games[player] == 7):
            self.completed_sets.append(self.games.copy())
            self.sets[player] += 1
            self.games = [0, 0]

    def display(self) -> dict:
        if not self.known:
            return {"points": ["?", "?"], "games": ["?", "?"], "sets": ["?", "?"],
                    "status": "Score unverified", "completed_sets": self.completed_sets}
        if self.kind == "eleven" or self.tiebreak:
            points = [str(p) for p in self.points]
        elif min(self.points) >= 3:
            points = ["40", "40"] if self.points[0] == self.points[1] else [
                "AD" if p > self.points[1 - i] else "40" for i, p in enumerate(self.points)]
        else:
            points = [str((0, 15, 30, 40)[min(p, 3)]) for p in self.points]
        return {"points": points, "games": self.games.copy(), "sets": self.sets.copy(),
                "status": "Final" if self.winner is not None else "Tiebreak" if self.tiebreak else "",
                "completed_sets": [s.copy() for s in self.completed_sets]}
