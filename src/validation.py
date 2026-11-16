"""Shared feature validation contract (API + dashboard).

Single source of truth so both entries enforce the same ranges.
Raises ValueError with a human-readable message; the API maps it to 422,
the dashboard shows it inline.
"""
import math

# Biologically plausible ranges (from training data, with margin).
# ffreq: bee buzz fundamental ~50-1000 Hz. duration: positive, < 5 s.
# MFCC stats: unbounded in theory; reject absurd values to catch unit errors.
FIELD_CONSTRAINTS = {"ffreq": (50.0, 1000.0), "duration.s": (0.0, 5.0)}
MFCC_ABS_MAX = 500.0


def validate_features(d: dict) -> None:
    for f, (lo, hi) in FIELD_CONSTRAINTS.items():
        v = d.get(f)
        if v is not None and not (lo <= v <= hi):
            raise ValueError(
                f"{f}={v} out of plausible range [{lo}, {hi}]. Check units (Hz, seconds).")
    for f in d:
        if f.startswith(("mean.", "sd.", "skew.", "kurt.", "med.")):
            v = d.get(f)
            if v is not None and (not math.isfinite(v) or abs(v) > MFCC_ABS_MAX):
                raise ValueError(
                    f"{f}={v} implausible (non-finite or |v| > {MFCC_ABS_MAX}). Check feature extraction.")
