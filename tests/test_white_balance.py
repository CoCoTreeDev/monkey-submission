"""A broken white balance blinds the detector; correct only the broken frames.

2026-08-15 06:17, 場域甲: five macaques walking down the road in plain view, and
the production model found ONE box across 59 frames. Not resolution — it was
just as blind at 1600, 1920, 2048 and 2560. The camera's white balance had
failed at the day/night transition: B/R 1.92 against a normal 1.07. Rebalancing
the channels took champion from 6 boxes to 96 on the same frames, peak 1 to 5.

Correcting EVERY frame is tempting — it lifts crowd-set recall 0.426 -> 0.540 —
but it costs the existing benchmark (test F1 0.8429 -> 0.8283, past the 0.01
ratchet tolerance), so it is gated instead. Above a 1.5 cast ratio the
correction is invisible to everything healthy: 0 of 285 sampled harvest frames
trip it, and benchmark test F1 is unchanged to four decimals.

A brightness-only control changed nothing (recall 0.452 vs 0.450), so the effect
is the channel balance and not the exposure.

No model, no camera, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402

from config import (  # noqa: E402
    WHITE_BALANCE_CAST_THRESHOLD,
    colour_cast,
    correct_white_balance,
)

FAILS: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILS.append(label)


def frame(b: int, g: int, r: int, h: int = 240, w: int = 320) -> np.ndarray:
    """A BGR frame with the given per-channel means (OpenCV channel order)."""
    f = np.zeros((h, w, 3), np.uint8)
    f[:, :, 0], f[:, :, 1], f[:, :, 2] = b, g, r
    return f


print("Measuring the cast")
check(abs(colour_cast(frame(120, 120, 120)) - 1.0) < 1e-6,
      "a neutral frame measures 1.0")
check(abs(colour_cast(frame(89, 89, 83)) - 89 / 83) < 0.01,
      "a healthy daylight frame (B89 G89 R83) is ~1.07, well under the threshold")
dawn = colour_cast(frame(166, 109, 86))
check(abs(dawn - 166 / 86) < 0.01,
      "the 2026-08-15 dawn failure (B166 G109 R86) measures %.2f" % dawn)
check(dawn > WHITE_BALANCE_CAST_THRESHOLD,
      "...which is above the %.1f threshold, so it gets corrected" % WHITE_BALANCE_CAST_THRESHOLD)
check(colour_cast(frame(89, 89, 83)) < WHITE_BALANCE_CAST_THRESHOLD,
      "...while the healthy frame is below it, so it is left alone")

print("\nThe threshold sits between the two populations")
# The crowd set's 195 hand-labelled frames span 1.02-1.45; the dawn frames 1.9+.
check(colour_cast(frame(130, 120, 100)) < WHITE_BALANCE_CAST_THRESHOLD,
      "the worst NORMAL frame measured in the crowd set (~1.45) stays untouched")
check(colour_cast(frame(180, 110, 90)) > WHITE_BALANCE_CAST_THRESHOLD,
      "a genuinely broken frame (~2.0) is corrected")

print("\nCorrecting")
fixed = correct_white_balance(frame(166, 109, 86))
check(abs(colour_cast(fixed) - 1.0) < 0.02, "correction lands on a neutral cast")
check(fixed.dtype == np.uint8, "output stays uint8 — it is fed straight to the model")
check(fixed.shape == (240, 320, 3), "shape is preserved")

bright = correct_white_balance(frame(250, 250, 250))
check(bright.max() <= 255, "a near-white frame does not overflow")
dark = correct_white_balance(frame(2, 1, 1))
check(dark.min() >= 0 and dark.dtype == np.uint8, "a near-black frame does not underflow")

neutral_in = frame(120, 120, 120)
neutral_out = correct_white_balance(neutral_in)
check(int(np.abs(neutral_out.astype(int) - neutral_in.astype(int)).max()) <= 1,
      "correcting an already-neutral frame is a no-op")

print("\nSampling shortcut is sound")
rng = np.random.default_rng(0)
noisy = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
full = [float(noisy[:, :, i].mean()) for i in range(3)]
full_ratio = max(full) / min(full)
check(abs(colour_cast(noisy) - full_ratio) < 0.02,
      "every-8th-pixel sampling matches the full-frame ratio (%.4f vs %.4f)"
      % (colour_cast(noisy), full_ratio))

print("\nDisabling")
check(colour_cast(frame(166, 109, 86)) > 0,
      "cast is always computable; setting the threshold to 0 is what disables the fix")

print("\n" + ("ALL PASS" if not FAILS else "FAILED: %d\n  - %s" % (len(FAILS), "\n  - ".join(FAILS))))
sys.exit(1 if FAILS else 0)
