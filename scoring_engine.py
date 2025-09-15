# scoring_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Any
import json
import math

# Types
Embedding = Dict[str, Optional[float]]
SimilarityMeta = Dict[str, Any]
SimilarityFn = Callable[[Embedding, Embedding], Tuple[float, SimilarityMeta]]

# -------------------------
# Difficulty / Scoring knobs
# -------------------------

# Stricter tier bands (hard but fair)
DEFAULT_THRESHOLDS = {
    "perfect": 94.0,
    "great":   85.0,
    "good":    72.0,
    "ok":      55.0,
    # miss: anything below ok
}

# Window aggregation
GAUSS_SIGMA_MS_HOLD = 150      # time weighting width for hold moves
GAUSS_SIGMA_MS_HIT  = 120      # time weighting width for hit moves
PERCENTILE = 0.85              # weighted percentile to take in the window

# Timing penalty (subtract after aggregation)
TIMING_PENALTY_PER_MS = 0.06   # points per ms from start (60 pts per second)
TIMING_PENALTY_MAX = 10.0      # cap penalty so it’s not brutal

# Stability requirements (consecutive frames at/above tier threshold)
CONSEC_FRAMES_HOLD = 4         # ~133 ms at 30 fps
CONSEC_FRAMES_HIT  = 2         # ~66 ms at 30 fps

# Form gate: cap max tier if any critical joint is way off (post-tolerance error)
CRITICAL_JOINTS = ("l_elbow", "r_elbow", "l_shoulder", "r_shoulder")
CRIT_JOINT_ERR_CAP_DEG = 30.0  # if exceeded on the peak frame, cap tier to "good"

# Hit/Hold classification
HIT_GAP_MS = 450               # gap to next move < this => classify as HIT
HIT_WINDOW_PRE_MS  = 200       # asymmetric window for hits (pre, post)
HIT_WINDOW_POST_MS = 100

# -------------------------

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
    best_score: float           # final numeric score after penalty/aggregation
    best_ts_ms: Optional[int]   # representative time (peak) used for penalty
    tier: str

def tier_of(score: float, thresholds: Dict[str, float] = DEFAULT_THRESHOLDS) -> str:
    if score >= thresholds.get("perfect", 94.0): return "perfect"
    if score >= thresholds.get("great",   85.0): return "great"
    if score >= thresholds.get("good",    72.0): return "good"
    if score >= thresholds.get("ok",      55.0): return "ok"
    return "miss"

def load_choreography(json_path: str) -> List[MoveSpec]:
    with open(json_path, "r") as f:
        data = json.load(f)

    moves = []
    for m in data.get("moves", []):
        angles = (m.get("pose", {}) or {}).get("angles", {}) or {}
        ref_emb: Embedding = {
            "l_elbow":    angles.get("l_elbow"),
            "r_elbow":    angles.get("r_elbow"),
            "l_shoulder": angles.get("l_shoulder"),
            "r_shoulder": angles.get("r_shoulder"),
            "l_hip":      angles.get("l_hip"),
            "r_hip":      angles.get("r_hip"),
        }
        moves.append(MoveSpec(
            name=str(m.get("name", "")),
            start_ms=int(m.get("start_ms", 0)),
            ref_emb=ref_emb,
            mirror=bool(m.get("mirror", False)),
            raw=m,
        ))
    moves.sort(key=lambda x: x.start_ms)
    return moves

class ScoringEngine:
    """
    Feed frames (timestamped by GAME CLOCK) via update(ts_ms, live_emb, similarity_fn).
    For each move, keep all frames inside a per-move window (HIT vs HOLD):
      - HIT: [start - 200ms, start + 100ms], sigma=120, consecutive N=2
      - HOLD: [start - W,    start + W   ], sigma=150, consecutive N=4
    Aggregate by weighted 85th percentile, subtract timing penalty, enforce stability,
    apply the critical-joint cap, then emit a result.
    """
    def __init__(
        self,
        moves: List[MoveSpec],
        window_half_ms: int = 250,
        thresholds: Dict[str, float] = None,
        tie_breaker: str = "closest",   # kept for compatibility (unused in new scheme)
    ):
        self.moves = moves
        self.window_half_ms = int(window_half_ms)
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.tie_breaker = tie_breaker

        # Per-move parameters (computed from gaps)
        self._params: List[Dict[str, Any]] = self._classify_moves()

        self._i = 0
        self._cur_start: Optional[int] = None
        self._cur_lo: Optional[int] = None
        self._cur_hi: Optional[int] = None
        self._cur_sigma: int = GAUSS_SIGMA_MS_HOLD
        self._cur_consec_N: int = CONSEC_FRAMES_HOLD

        # Buffer of frames within current window: list of dicts with ts, score, meta
        self._frames: List[Dict[str, Any]] = []

        if self.moves:
            self._prime_current()

    def _classify_moves(self) -> List[Dict[str, Any]]:
        """
        Build per-move parameters based on gap to next move.
        """
        params: List[Dict[str, Any]] = []
        for idx, m in enumerate(self.moves):
            # Compute gap to the next move (if any)
            if idx + 1 < len(self.moves):
                gap = self.moves[idx + 1].start_ms - m.start_ms
            else:
                gap = 10_000_000  # huge gap for last move

            is_hit = gap < HIT_GAP_MS
            if is_hit:
                pre = HIT_WINDOW_PRE_MS
                post = HIT_WINDOW_POST_MS
                sigma = GAUSS_SIGMA_MS_HIT
                consec = CONSEC_FRAMES_HIT
                mtype = "hit"
            else:
                pre = self.window_half_ms
                post = self.window_half_ms
                sigma = GAUSS_SIGMA_MS_HOLD
                consec = CONSEC_FRAMES_HOLD
                mtype = "hold"

            params.append({
                "type": mtype,
                "pre": int(pre),
                "post": int(post),
                "sigma": int(sigma),
                "consec": int(consec),
            })
        return params

    def _prime_current(self):
        m = self.moves[self._i]
        p = self._params[self._i]
        self._cur_start = m.start_ms
        self._cur_lo = m.start_ms - p["pre"]
        self._cur_hi = m.start_ms + p["post"]
        self._cur_sigma = p["sigma"]
        self._cur_consec_N = p["consec"]
        self._frames = []

    def _finalize_current(self) -> MoveResult:
        m = self.moves[self._i]
        sigma = self._cur_sigma
        consec_N = self._cur_consec_N

        if not self._frames:
            # No usable frames in the window → MISS
            result = MoveResult(name=m.name, start_ms=m.start_ms, best_score=0.0, best_ts_ms=None, tier="miss")
            self._advance()
            return result

        # Sort frames by time for stability checks
        self._frames.sort(key=lambda d: d["ts"])

        # --- Time weights (Gaussian centered on start_ms) ---
        weights = []
        for fr in self._frames:
            dt = fr["ts"] - m.start_ms
            w = math.exp(- (dt / sigma) ** 2)
            weights.append(w)
        w_sum = sum(weights) if weights else 1.0
        weights = [w / w_sum for w in weights]

        # --- Weighted percentile (e.g., 85th) ---
        # Build pairs sorted by score ascending
        pairs = sorted(zip([fr["score"] for fr in self._frames], weights, range(len(self._frames))), key=lambda x: x[0])
        cdf = 0.0
        agg_score = pairs[-1][0]  # fallback to max
        agg_index = pairs[-1][2]
        for s, w, idx in pairs:
            cdf += w
            if cdf >= PERCENTILE:
                agg_score = s
                agg_index = idx
                break

        # --- Peak time for timing penalty: frame with max (w_i * score_i) ---
        peak_idx = max(range(len(self._frames)), key=lambda i: self._frames[i]["score"] * weights[i])
        peak_ts = self._frames[peak_idx]["ts"]
        # Timing penalty
        pen = min(TIMING_PENALTY_MAX, TIMING_PENALTY_PER_MS * abs(peak_ts - m.start_ms))
        score_after_pen = max(0.0, agg_score - pen)

        # --- Consecutive-frames stability (check against tier thresholds) ---
        def longest_consec_at_or_above(thresh: float) -> int:
            longest = cur = 0
            for fr in self._frames:
                if fr["score"] >= thresh:
                    cur += 1
                    if cur > longest:
                        longest = cur
                else:
                    cur = 0
            return longest

        tiers_desc = sorted(
            [(name, val) for name, val in self.thresholds.items()],
            key=lambda x: x[1],
            reverse=True
        )
        provisional_tier = "miss"
        for t_name, t_thresh in tiers_desc:
            if score_after_pen >= t_thresh and longest_consec_at_or_above(t_thresh) >= consec_N:
                provisional_tier = t_name
                break

        # --- Critical-joint cap on the peak frame ---
        peak_meta = self._frames[peak_idx]["meta"] or {}
        per_joint = peak_meta.get("per_joint_err", {})
        cap_to_good = False
        for j in CRITICAL_JOINTS:
            if j in per_joint:
                post_tol_err_deg = per_joint[j][0]  # (diff_after_tol, weight)
                if post_tol_err_deg is not None and post_tol_err_deg > CRIT_JOINT_ERR_CAP_DEG:
                    cap_to_good = True
                    break

        final_tier = provisional_tier
        if cap_to_good:
            order = ["miss", "ok", "good", "great", "perfect"]
            idx = min(order.index("good"), order.index(final_tier))
            final_tier = order[idx]

        result = MoveResult(
            name=m.name,
            start_ms=m.start_ms,
            best_score=score_after_pen,
            best_ts_ms=peak_ts,
            tier=final_tier,
        )

        self._advance()
        return result

    def _advance(self):
        self._i += 1
        if self._i < len(self.moves):
            self._prime_current()
        else:
            self._cur_start = self._cur_lo = self._cur_hi = None
            self._frames = []

    def update(
        self,
        ts_ms: int,
        live_emb: Embedding,
        similarity_fn: SimilarityFn,
    ) -> List[MoveResult]:
        """
        Call every frame in increasing GAME ts_ms order.
        Returns 0 or more MoveResults (if time leaps beyond windows).
        """
        emitted: List[MoveResult] = []
        if not self.moves or self._cur_start is None:
            return emitted

        # If we jumped past the current window end, finalize (and possibly more than one)
        while self._cur_hi is not None and ts_ms > self._cur_hi and self._i < len(self.moves):
            emitted.append(self._finalize_current())
            if self._cur_start is None:
                return emitted
            if ts_ms <= self._cur_hi:
                break

        # If inside current window, evaluate this frame
        if self._cur_lo is not None and self._cur_hi is not None and self._cur_lo <= ts_ms <= self._cur_hi:
            m = self.moves[self._i]
            score, meta = similarity_fn(live_emb, m.ref_emb)
            self._frames.append({"ts": ts_ms, "score": float(score), "meta": meta})

        return emitted

    def finalize_all(self) -> List[MoveResult]:
        """Flush any remaining move(s) at the end."""
        out: List[MoveResult] = []
        while self._cur_start is not None and self._i < len(self.moves):
            out.append(self._finalize_current())
        return out
