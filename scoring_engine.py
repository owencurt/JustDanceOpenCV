# scoring_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Any
import json

# Types
Angles = Dict[str, float]
Embedding = Dict[str, Optional[float]]
SimilarityFn = Callable[[Embedding, Embedding], Tuple[float, Dict[str, Any]]]

# Default tier thresholds on your 0..100 similarity scale (higher is better).
DEFAULT_THRESHOLDS = {
    "perfect": 88.0,
    "great":   75.0,
    "good":    60.0,
    "ok":      40.0,
    # miss: anything below ok
}

@dataclass
class MoveSpec:
    name: str
    start_ms: int
    ref_emb: Embedding           # we’ll store pose.angles directly as the "embedding"
    mirror: bool = False         # present in JSON; unused per your spec
    raw: dict = None             # full raw move dict (optional)

@dataclass
class MoveResult:
    name: str
    start_ms: int
    best_score: float
    best_ts_ms: Optional[int]    # None if never got a valid frame
    tier: str

def tier_of(score: float, thresholds: Dict[str, float] = DEFAULT_THRESHOLDS) -> str:
    if score >= thresholds.get("perfect", 88.0): return "perfect"
    if score >= thresholds.get("great",   75.0): return "great"
    if score >= thresholds.get("good",    60.0): return "good"
    if score >= thresholds.get("ok",      40.0): return "ok"
    return "miss"

def load_choreography(json_path: str) -> List[MoveSpec]:
    with open(json_path, "r") as f:
        data = json.load(f)

    moves = []
    for m in data.get("moves", []):
        # Use the angles dict directly as the reference "embedding"
        angles: Angles = (m.get("pose", {}) or {}).get("angles", {}) or {}
        # Normalize into the embedding shape your similarity expects: dict[str -> float or None]
        ref_emb: Embedding = {
            "l_elbow":    angles.get("l_elbow"),
            "r_elbow":    angles.get("r_elbow"),
            "l_shoulder": angles.get("l_shoulder"),
            "r_shoulder": angles.get("r_shoulder"),
            "l_hip":      angles.get("l_hip"),
            "r_hip":      angles.get("r_hip"),
            # knees/ankles omitted (your similarity gives them 0 weight anyway)
        }
        moves.append(MoveSpec(
            name=str(m.get("name", "")),
            start_ms=int(m.get("start_ms", 0)),
            ref_emb=ref_emb,
            mirror=bool(m.get("mirror", False)),
            raw=m,
        ))

    # Ensure chronological order (robustness)
    moves.sort(key=lambda x: x.start_ms)
    return moves

class ScoringEngine:
    """
    Feed it frames in timestamp order via update(ts_ms, live_emb, similarity_fn).
    It will evaluate each move within [start_ms - window_half_ms, start_ms + window_half_ms],
    pick the best score for that move, and emit a MoveResult once the window is past.
    """
    def __init__(
        self,
        moves: List[MoveSpec],
        window_half_ms: int = 250,
        thresholds: Dict[str, float] = None,
        tie_breaker: str = "closest",   # "closest" to start_ms or "earliest"
    ):
        self.moves = moves
        self.window_half_ms = int(window_half_ms)
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.tie_breaker = tie_breaker

        self._i = 0  # index of current move
        self._best_score: Optional[float] = None
        self._best_ts: Optional[int] = None

        # cache start/end for current move
        self._cur_start: Optional[int] = None
        self._cur_lo: Optional[int] = None
        self._cur_hi: Optional[int] = None

        if self.moves:
            self._prime_current()

    def _prime_current(self):
        m = self.moves[self._i]
        self._cur_start = m.start_ms
        self._cur_lo = m.start_ms - self.window_half_ms
        self._cur_hi = m.start_ms + self.window_half_ms
        self._best_score = None
        self._best_ts = None

    def _finalize_current(self) -> MoveResult:
        m = self.moves[self._i]
        best = self._best_score if self._best_score is not None else 0.0
        tier = tier_of(best, self.thresholds)
        result = MoveResult(
            name=m.name,
            start_ms=m.start_ms,
            best_score=best,
            best_ts_ms=self._best_ts,
            tier=tier,
        )
        self._i += 1
        if self._i < len(self.moves):
            self._prime_current()
        else:
            # Exhausted
            self._cur_start = self._cur_lo = self._cur_hi = None
        return result

    def update(
        self,
        ts_ms: int,
        live_emb: Embedding,
        similarity_fn: SimilarityFn,
    ) -> List[MoveResult]:
        """
        Call every frame in increasing ts_ms order.
        Returns 0 or more MoveResults (e.g., if frames jump ahead and we skip past multiple windows).
        """
        emitted: List[MoveResult] = []
        if not self.moves or self._cur_start is None:
            return emitted

        # Handle skipped moves if time jumps beyond their windows
        while self._cur_hi is not None and ts_ms > self._cur_hi and self._i < len(self.moves):
            # We have passed beyond the current window → finalize it
            emitted.append(self._finalize_current())
            if self._cur_start is None:
                return emitted  # no more moves

            # In case time jumped way forward, keep finalizing until current window includes ts or we run out
            if ts_ms <= self._cur_hi:
                break

        # If current frame falls inside the active move window, evaluate
        if self._cur_lo is not None and self._cur_hi is not None and self._cur_lo <= ts_ms <= self._cur_hi:
            m = self.moves[self._i]
            score, _meta = similarity_fn(live_emb, m.ref_emb)

            # Accept frames with non-zero score, or keep 0 if it's truly the best so far
            if self._best_score is None or score > self._best_score:
                self._best_score = score
                self._best_ts = ts_ms
            elif self._best_score is not None and score == self._best_score:
                # tie-breaker
                if self.tie_breaker == "closest" and self._best_ts is not None:
                    cur_dist = abs(ts_ms - m.start_ms)
                    best_dist = abs(self._best_ts - m.start_ms)
                    if cur_dist < best_dist:
                        self._best_ts = ts_ms
                elif self.tie_breaker == "earliest":
                    # keep the earlier timestamp as the winner (do nothing since _best_ts is earlier)
                    pass

        return emitted

    def finalize_all(self) -> List[MoveResult]:
        """Call once at the very end if you want to flush any remaining move(s)."""
        out: List[MoveResult] = []
        while self._cur_start is not None and self._i < len(self.moves):
            out.append(self._finalize_current())
        return out
