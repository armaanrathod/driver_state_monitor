"""
temporal.py
-----------
Logic layer — temporal aggregation of frame-level eye metrics.

Responsibilities:
  - Maintain a time-bounded sliding window of EAR observations
  - Calculate PERCLOS (Percentage of Eye Closure) over the window
  - Detect micro-sleep events (sustained closure beyond a safe threshold)
  - Produce a composite DrowsinessScore (0.0–1.0)

Does NOT:
  - Import or use OpenCV, MediaPipe, or Camera
  - Read from any hardware device
  - Draw or visualize anything
  - Compute EAR itself (that is eyes.py's responsibility)

Architecture note
-----------------
This module sits between the feature extraction layer (eyes.py) and the
future decision/alerting layer. It converts per-frame snapshots into
time-aware signals that a decision layer can act on.

    Camera → FaceMesh → Eyes (EyeMetrics) → TemporalAggregator (TemporalMetrics)
                                                         ↓
                                               (future) Decision / Alert layer

References
----------
- Wierwille & Ellsworth (1994): PERCLOS as a drowsiness measure.
  P80 definition: proportion of time eyelids cover ≥ 80% of the pupil
  (approximated here as EAR ≤ ear_closed_threshold).
- Dinges et al. (1998): PERCLOS > 0.15 indicates moderate drowsiness;
  > 0.30 indicates severe drowsiness.
- Normal blink rate: 12–20 blinks/min at rest;
  deviations in either direction can signal fatigue.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from driver_awareness.perception.eyes import EyeMetrics


# ---------------------------------------------------------------------------
# Internal sliding-window record
# ---------------------------------------------------------------------------

class _Frame(NamedTuple):
    """One observation stored in the sliding window."""
    timestamp: float   # time.monotonic() at ingestion
    ear: float         # mean EAR for this frame
    is_closed: bool    # confirmed closed state from BlinkDetector
    blinked: bool      # True on the frame a blink completed


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalMetrics:
    """
    Immutable snapshot of temporal / time-aware drowsiness signals.

    Attributes
    ----------
    perclos : float
        Proportion of frames within the sliding window where eyes were
        confirmed closed. Range [0.0, 1.0].
        Thresholds (Dinges 1998):
          < 0.15  → alert
          0.15–0.30 → moderately drowsy
          > 0.30  → severely drowsy

    blink_rate_per_min : float
        Number of blinks per minute calculated over the current window.
        Normal range is ~12–20 bpm; deviation indicates fatigue.

    microsleep_detected : bool
        True if eyes are *currently* in a sustained closure that has
        exceeded ``microsleep_threshold_s``.

    microsleep_duration_s : float
        Duration (seconds) of the *current* sustained closure, or 0.0
        if the eyes are not currently closed.

    longest_closure_s : float
        Duration (seconds) of the longest single closure event recorded
        within the current window.

    drowsiness_score : float
        Composite fatigue index in [0.0, 1.0].
        0.0 = fully alert; 1.0 = critically impaired.
        Weighted from PERCLOS (60%), blink-rate deviation (25%),
        and micro-sleep penalty (15%).

    window_duration_s : float
        Actual time span covered by the current window (≤ ``window_s``).

    frames_in_window : int
        Number of observations currently held in the window.
    """

    perclos: float
    blink_rate_per_min: float
    microsleep_detected: bool
    microsleep_duration_s: float
    longest_closure_s: float
    drowsiness_score: float
    window_duration_s: float
    frames_in_window: int


# ---------------------------------------------------------------------------
# Temporal aggregator
# ---------------------------------------------------------------------------

@dataclass
class TemporalAggregator:
    """
    Converts a stream of per-frame ``EyeMetrics`` into time-aware signals.

    The aggregator maintains a sliding window (bounded by wall-clock time)
    and derives PERCLOS, blink rate, micro-sleep detection, and a composite
    drowsiness score from that window.

    Parameters
    ----------
    window_s : float
        Length of the sliding window in seconds (default 60 s).
        Frames older than this are evicted on each ``update()`` call.
    ear_closed_threshold : float
        EAR value at or below which an eye is considered closed for
        PERCLOS counting (default 0.20). Should match BlinkDetector's
        ``ear_threshold`` for consistency.
    microsleep_threshold_s : float
        Sustained closure duration (seconds) above which a micro-sleep
        event is flagged (default 1.5 s).
    normal_blink_rate_min : float
        Lower bound of normal blink rate (blinks/min) used for scoring
        (default 12.0).
    normal_blink_rate_max : float
        Upper bound of normal blink rate (blinks/min) used for scoring
        (default 20.0).

    Usage
    -----
        aggregator = TemporalAggregator()

        # Inside your frame loop:
        temporal = aggregator.update(eye_metrics)
        print(temporal.drowsiness_score)

    Design note
    -----------
    All timing is driven by ``time.monotonic()`` by default. For unit
    tests or simulation, pass a manual ``timestamp`` to ``update()``.
    This keeps the module pure and fully testable without real time.
    """

    window_s: float = 60.0
    ear_closed_threshold: float = 0.20
    microsleep_threshold_s: float = 1.5
    normal_blink_rate_min: float = 12.0
    normal_blink_rate_max: float = 20.0

    # ------------------------------------------------------------------
    # Private state — not part of public API
    # ------------------------------------------------------------------

    # Sliding window: fixed-size deque (maxlen acts as a hard memory cap
    # beyond the time-based eviction, set generously at 10 min × 60 fps).
    _window: deque[_Frame] = field(
        default_factory=lambda: deque(maxlen=36_000),
        init=False,
        repr=False,
    )

    # Sustained-closure tracking
    _current_closure_start: float | None = field(
        default=None, init=False, repr=False
    )

    # Longest closure event seen within the active window
    _longest_closure_s: float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        metrics: EyeMetrics,
        timestamp: float | None = None,
    ) -> TemporalMetrics:
        """
        Ingest one frame of EyeMetrics and return updated temporal signals.

        Parameters
        ----------
        metrics : EyeMetrics
            Output from ``eyes.process_eyes()`` for the current frame.
        timestamp : float | None
            Wall-clock time for this frame (``time.monotonic()`` units).
            If None, the current ``time.monotonic()`` value is used.
            Pass an explicit value in tests to decouple from real time.

        Returns
        -------
        TemporalMetrics
            Immutable snapshot of all temporal signals for this frame.
        """
        ts = timestamp if timestamp is not None else time.monotonic()

        # 1. Store frame in window
        frame = _Frame(
            timestamp=ts,
            ear=metrics.mean_ear,
            is_closed=metrics.is_closed,
            blinked=metrics.blink_detected,
        )
        self._window.append(frame)

        # 2. Evict frames older than the window duration
        cutoff = ts - self.window_s
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()

        # 3. Update sustained-closure state
        self._update_closure_tracking(metrics.is_closed, ts)

        # 4. Derive signals from window
        perclos = self._compute_perclos()
        blink_rate = self._compute_blink_rate(ts)
        microsleep_duration = self._current_closure_duration(ts)
        microsleep_detected = microsleep_duration >= self.microsleep_threshold_s

        # 5. Composite drowsiness score
        drowsiness = self._compute_drowsiness_score(
            perclos=perclos,
            blink_rate=blink_rate,
            microsleep_detected=microsleep_detected,
        )

        # 6. Window metadata
        if len(self._window) >= 2:
            window_duration = self._window[-1].timestamp - self._window[0].timestamp
        else:
            window_duration = 0.0

        return TemporalMetrics(
            perclos=perclos,
            blink_rate_per_min=blink_rate,
            microsleep_detected=microsleep_detected,
            microsleep_duration_s=microsleep_duration,
            longest_closure_s=self._longest_closure_s,
            drowsiness_score=drowsiness,
            window_duration_s=window_duration,
            frames_in_window=len(self._window),
        )

    def reset(self) -> None:
        """
        Clear all internal state.

        Call after a long gap with no face detected to prevent stale data
        from polluting subsequent scores.
        """
        self._window.clear()
        self._current_closure_start = None
        self._longest_closure_s = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_closure_tracking(self, is_closed: bool, ts: float) -> None:
        """Maintain the start-time of the current sustained closure."""
        if is_closed:
            if self._current_closure_start is None:
                # Eyes just closed — record the onset
                self._current_closure_start = ts
            else:
                # Eyes still closed — update longest-closure record
                duration = ts - self._current_closure_start
                if duration > self._longest_closure_s:
                    self._longest_closure_s = duration
        else:
            # Eyes just opened (or were already open)
            if self._current_closure_start is not None:
                # Closure event ended — record its duration
                duration = ts - self._current_closure_start
                if duration > self._longest_closure_s:
                    self._longest_closure_s = duration
            self._current_closure_start = None

    def _current_closure_duration(self, ts: float) -> float:
        """Return how long eyes have been continuously closed, or 0.0."""
        if self._current_closure_start is None:
            return 0.0
        return ts - self._current_closure_start

    def _compute_perclos(self) -> float:
        """
        PERCLOS = fraction of frames in the window where eyes were closed.

        Uses the ``is_closed`` flag from EyeMetrics (which already applies
        the BlinkDetector's debounce), so transient noise is filtered out
        before it reaches this layer.
        """
        if not self._window:
            return 0.0

        closed_flags = np.array([f.is_closed for f in self._window], dtype=np.float32)
        return float(closed_flags.mean())

    def _compute_blink_rate(self, current_ts: float) -> float:
        """
        Blink rate in blinks per minute over the active window.

        Only frames within the window are counted; the rate is scaled to
        60 seconds so it's comparable regardless of window fill level.
        """
        if not self._window:
            return 0.0

        blink_count = sum(1 for f in self._window if f.blinked)
        if len(self._window) < 2:
            return 0.0

        window_span_s = current_ts - self._window[0].timestamp
        if window_span_s < 1.0:
            # Not enough data yet — avoid division by near-zero
            return 0.0

        return float(blink_count / window_span_s * 60.0)

    def _compute_drowsiness_score(
        self,
        perclos: float,
        blink_rate: float,
        microsleep_detected: bool,
    ) -> float:
        """
        Composite drowsiness index in [0.0, 1.0].

        Component weights
        -----------------
        PERCLOS (60%)
            Primary signal. Scaled so that 0.30 (severe threshold per
            Dinges 1998) maps to 1.0.

        Blink-rate deviation (25%)
            Peaks when blink rate falls outside the normal range.
            Both abnormally high (> normal_max × 1.5) and abnormally
            low (< normal_min × 0.5) rates contribute, but low rate
            (potential micro-staring) is weighted more heavily.

        Micro-sleep penalty (15%)
            Binary: 1.0 if a micro-sleep is currently active, 0.0
            otherwise. Acts as a hard boost when the most dangerous
            condition is present.
        """
        # --- PERCLOS component ---
        # 0.30 = severe drowsiness threshold (maps to score 1.0)
        perclos_score = min(perclos / 0.30, 1.0)

        # --- Blink-rate deviation component ---
        blink_score = self._blink_rate_score(blink_rate)

        # --- Micro-sleep component ---
        microsleep_score = 1.0 if microsleep_detected else 0.0

        # --- Weighted sum ---
        raw = (
            0.60 * perclos_score
            + 0.25 * blink_score
            + 0.15 * microsleep_score
        )
        return float(min(max(raw, 0.0), 1.0))

    def _blink_rate_score(self, blink_rate: float) -> float:
        """
        Map blink rate to a [0, 1] fatigue sub-score.

        Score is 0.0 when rate is inside the normal band and rises toward
        1.0 as rate deviates further in either direction.

        Low rate (<  normal_min × 0.5) → score rises to 1.0 (micro-staring)
        High rate (> normal_max × 1.5) → score rises to 0.5 (agitation)
        """
        if blink_rate <= 0.0:
            # No data or no blinks yet — treat as unknown, no contribution
            return 0.0

        low_threshold = self.normal_blink_rate_min * 0.5   # e.g. 6 bpm
        high_threshold = self.normal_blink_rate_max * 1.5  # e.g. 30 bpm

        if low_threshold <= blink_rate <= high_threshold:
            # Inside the acceptable band
            return 0.0

        if blink_rate < low_threshold:
            # Abnormally low — micro-staring risk
            # Linear ramp: score = 1.0 at 0 bpm, 0.0 at low_threshold
            score = 1.0 - (blink_rate / low_threshold)
            return float(min(score, 1.0))

        # Abnormally high — fatigue agitation; cap contribution at 0.5
        # Linear ramp: 0.0 at high_threshold, 0.5 at high_threshold × 2
        excess = blink_rate - high_threshold
        score = (excess / high_threshold) * 0.5
        return float(min(score, 0.5))