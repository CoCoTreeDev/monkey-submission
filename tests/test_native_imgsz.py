"""Never downscale the frame.

INFERENCE_IMGSZ used to be a fixed size, chosen when every camera delivered
1280x720. 場域甲 was later reconfigured to 1920x1080 and nothing noticed, so the
only feed that has ever seen a macaque was being shrunk by 0.833x before
inference — a median 67x71 px animal reaching the network as 56x59.

Measured on 195 hand-labelled 1920x1080 production frames (2026-08-13):
recall 0.349 at a fixed 1600 versus 0.426 inferred at native 1920.

The fix is deliberately a RULE, not a new number, because a fixed 1920 is not
the answer either: on the 2048x1094 must_sight fixture, shrinking to 1920 drops
box[0] from IoU 0.452 to 0.000 — the animal vanishes entirely at a scale factor
of 0.94 and comes back at 2048. Rescaling is what hurts. So: floor, never
shrink, and let each camera's own resolution decide.

No model, no camera, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "tests"))

from config import INFERENCE_IMGSZ, imgsz_for_frame  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILS.append(label)


print("The rule (floor = %d)" % INFERENCE_IMGSZ)

check(imgsz_for_frame(720, 1280) == INFERENCE_IMGSZ,
      "a 1280x720 camera still infers at the floor — unchanged and free")
check(imgsz_for_frame(1080, 1920) == 1920,
      "a 1920x1080 camera infers at 1920, not squeezed into %d" % INFERENCE_IMGSZ)
check(imgsz_for_frame(1094, 2048) == 2048,
      "the 2048x1094 fixture infers at its own size — the one 1920 loses a box on")
check(imgsz_for_frame(2160, 3840) == 3840,
      "a 4K feed would infer at 4K rather than being shrunk 0.42x")

print("\nIt is a floor, and it never shrinks")
for h, w in [(480, 640), (720, 1280), (600, 800), (1, 1)]:
    check(imgsz_for_frame(h, w) == INFERENCE_IMGSZ,
          "%dx%d is below the floor, so it is raised to it, not left small" % (w, h))
for h, w in [(1080, 1920), (1094, 2048), (1440, 2560)]:
    long_side = max(h, w)
    check(imgsz_for_frame(h, w) >= long_side,
          "%dx%d is never inferred smaller than itself" % (w, h))

print("\nOrientation and stride")
check(imgsz_for_frame(1920, 1080) == imgsz_for_frame(1080, 1920),
      "portrait and landscape of the same frame get the same size (long side)")
check(imgsz_for_frame(1344, 1008) == INFERENCE_IMGSZ,
      "a 1008x1344 phone photo has a 1344 long side — BELOW the floor, so it is "
      "raised to %d rather than inferred at its own size" % INFERENCE_IMGSZ)
for h, w in [(1090, 2050), (1000, 1601), (1000, 1633)]:
    v = imgsz_for_frame(h, w)
    check(v % 32 == 0, "%dx%d rounds to a multiple of 32 — got %d" % (w, h, v))
    check(v >= max(h, w), "%dx%d rounds UP, never down — %d >= %d" % (w, h, v, max(h, w)))

print("\nThe gate can express the same rule")
from eval_ratchet import _imgsz_for  # noqa: E402


class _Args:
    def __init__(self, imgsz, native):
        self.imgsz, self.native = imgsz, native


fixed, native = _Args(1600, False), _Args(1600, True)
check(_imgsz_for(fixed, 1920, 1080) == 1600,
      "without --native the gate keeps the old fixed behaviour")
check(_imgsz_for(native, 1920, 1080) == 1920,
      "with --native the gate infers a 1920x1080 image at 1920")
check(_imgsz_for(native, 1280, 720) == 1600,
      "with --native a 1280x720 image still gets the floor")
check(_imgsz_for(native, 1920, 1080) == imgsz_for_frame(1080, 1920),
      "the gate and the detector compute the SAME size — a gate that measured a "
      "different rule than production runs could not police it")

print("\n" + ("ALL PASS" if not FAILS else "FAILED: %d\n  - %s" % (len(FAILS), "\n  - ".join(FAILS))))
sys.exit(1 if FAILS else 0)
