"""Eval-set drift guard: is the pinned ndjson still the platform's test/val?

The ratchet's F1 numbers are only comparable on a FIXED eval set, so the
ndjson is a pinned snapshot, deliberately not auto-synced. This script is the
cap that makes the pin safe:

  --check    compare the pinned test+val (image content hashes + label
             fingerprints) against the platform's CURRENT state.
             exit 0 = identical, exit 3 = drifted (details printed).
             Wired into model-promote.yml (hard gate) and
             model-evaluate.yml (warning).
  --refresh  pull a fresh export into ml/datasets/ and print the re-anchor
             steps (new baseline must be recorded on the same commit).

Train-split drift is reported but never fails the check — growing train data
is the whole point of the flywheel; only test/val changes invalidate baselines.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_platform as fp  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATASET_NAME = "monkey cocotree v2"  # resolved to {owner}/{url-name} at call time:
# Dataset references resolve through the configured account and dataset name.
#
# This was pinned to the pre-v2 `monkey cocotree` (test 91 / train 827) until
# 2026-08-22, months after everything else moved to v2. The drift check
# therefore compared the local v2 export against a different dataset, reported
# "test: 本地 292 / 平台 91" every single time, and exited 3 — which killed the
# 上板 button in ~28s before the ratchet gate ever ran. Three dispatches failed
# that way (08-20, 08-21, 08-22) and both promotions in between were made by
# hand.
GUARDED_SPLITS = ("test", "val")

# The basis every baseline in tests/ was recorded under. eval_ratchet refuses to
# gate on a mismatch, so a re-anchor run that omits these does not re-anchor —
# it exits 1 on REFUSING TO GATE.
EVAL_BASIS = ("--native", "--imgsz", "1600",
              "--gray-world", "--gray-world-above", "1.5",
              "--half", "--conf", "0.5")


def _box_fingerprint(boxes) -> str:
    """Order-independent label fingerprint, tolerant to float noise."""
    rounded = sorted(
        tuple(round(float(v), 4) for v in b) for b in (boxes or [])
    )
    return json.dumps(rounded)


def local_state(ndjson: Path) -> dict[str, dict[str, str]]:
    """{split: {content_hash: label_fingerprint}} from the pinned export."""
    state: dict[str, dict[str, str]] = {}
    for line in ndjson.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("type") != "image":
            continue
        m = re.search(r"/i/([0-9a-f]{32})\.", r.get("url", ""))
        if not m:
            continue
        boxes = (r.get("annotations") or {}).get("boxes") or []
        state.setdefault(r.get("split", "?"), {})[m.group(1)] = _box_fingerprint(boxes)
    return state


def _ds_ref() -> tuple[str, ...]:
    """(owner, url-name) for the pinned dataset — the SDK's positional args."""
    if not _ds_ref_cache:
        _ds_ref_cache.append(fp._ref(fp._resolve("datasets", DATASET_NAME), "datasets"))
    return _ds_ref_cache[0]


_ds_ref_cache: list[tuple[str, ...]] = []


def platform_state() -> dict[str, dict[str, str]]:
    """Same shape, from the live platform (paged, includeLabels)."""
    state: dict[str, dict[str, str]] = {}
    offset = 0
    while True:
        page = fp.api().datasets.list_images(
            *_ds_ref(), limit=500, offset=offset, include_labels="true"
        )
        imgs = page.get("images", [])
        for i in imgs:
            boxes = [l.get("bbox", []) + [l.get("classId", 0)] for l in (i.get("labels") or [])]
            # keep fingerprint comparable with the ndjson (cls first there)
            boxes = [[b[-1], *b[:-1]] for b in boxes]
            state.setdefault(i.get("split", "?"), {})[i.get("hash")] = _box_fingerprint(boxes)
        offset += len(imgs)
        if not imgs or offset >= page.get("total", 0):
            break
    return state


def check() -> int:
    candidates = sorted((REPO / "ml" / "datasets").glob("*.ndjson"))
    if not candidates:
        print("找不到 ml/datasets/*.ndjson — 先跑 --refresh")
        return 3
    pinned = candidates[-1]
    loc, plat = local_state(pinned), platform_state()
    print(f"考卷: {pinned.name}")
    # Additive drift is NOT a broken exam. If every archived image is still on
    # the platform with the same labels, the frozen exam is intact and its
    # baseline stays comparable — the platform simply has extra material, which
    # per the project's benchmark discipline becomes a NEW exam rather than
    # being folded into this one. Only images that vanished or were relabelled
    # actually invalidate a recorded F1.
    #
    # Treating "新增" as drift is what made this exit 3 on a healthy repo: on
    # 2026-08-22 the platform had 50 extra test images and nothing removed, and
    # the old code called that incomparable and blocked promotion.
    drifted = False   # destructive — the archived exam itself changed
    grew = False      # additive — exam intact, new material alongside it
    for split in GUARDED_SPLITS + ("train",):
        l, p = loc.get(split, {}), plat.get(split, {})
        added = set(p) - set(l)
        removed = set(l) - set(p)
        relabeled = {h for h in set(l) & set(p) if l[h] != p[h]}
        line = f"  {split}: 本地 {len(l)} / 平台 {len(p)}"
        if added or removed or relabeled:
            line += f" — 新增 {len(added)}, 移除 {len(removed)}, 改標 {len(relabeled)}"
            if split in GUARDED_SPLITS:
                if removed or relabeled:
                    drifted = True
                    line += "  ⚠ 考卷變了，歷史 F1 不可比"
                else:
                    grew = True
                    line += "  ＋ 純新增，封存考卷每一題都還在"
        else:
            line += " ✓"
        print(line)
    if drifted:
        print("\nDRIFT: 平台的 test/val 已與封存考卷不同 —")
        print("  歷史 F1 與新評測不可比。重定錨流程:")
        print("  1. uv run --project backend python tests/dataset_sync.py --refresh")
        print("  2. uv run --project backend python tests/eval_ratchet.py --device mps "
              + " ".join(EVAL_BASIS) + " --update-baseline --rebase-basis")
        print("  3. 新 ndjson + 新 f1_baseline.json 同一個 commit 提交")
        return 3
    if grew:
        print("\n平台多了題目，但封存考卷的每一題都還在、答案也沒改 —")
        print("  基準仍然有效，棘輪照常評分。")
        print("  紀律：既有考卷不增刪；要考新素材就另開一份新考卷與新基準。")
        return 0
    print("考卷未變 — 基準線有效")
    return 0


def refresh() -> None:
    ds = fp.api().datasets.retrieve_export(*_ds_ref())
    # Overwrite the exam that check() pins, not a hardcoded name — this wrote
    # monkey-cocotree-july.ndjson long after the pinned exam became v2, which
    # would have left TWO exports in ml/datasets/ and made find_ndjson ambiguous.
    existing = sorted((REPO / "ml" / "datasets").glob("*.ndjson"))
    dest = existing[-1] if existing else (
        REPO / "ml" / "datasets" / f"{DATASET_NAME.replace(' ', '-')}.ndjson")
    fp._download(ds["downloadUrl"], dest)
    print("已更新考卷。接著在 Apple Silicon 機器上重定錨:")
    print("  uv run --project backend python tests/eval_ratchet.py --device mps "
          + " ".join(EVAL_BASIS) + " --update-baseline --rebase-basis")
    print("  （新 ndjson 與新 baseline 必須同一個 commit）")


def auto_reanchor() -> int:
    """Self-healing cap: on eval-set drift, re-pin and re-anchor automatically.

    The ratchet's invariant is not "the exam never changes" — it is "champion
    and challenger sit the SAME exam". So on drift the right move is: pull the
    new exam, let the CURRENT production model set the bar on it, commit exam
    + bar together, and only then let candidates be judged. Runs on the mini
    (needs MPS + push credentials); at most one re-anchor commit per dataset
    change, no-op when nothing drifted.
    """
    import os
    import subprocess

    if check() == 0:
        return 0

    # Establish HOW we will push BEFORE changing anything. The mini's ssh deploy
    # key is read-only, so a plain `git push origin main` can never succeed there
    # — it committed, failed to push, returned 3, and the next deploy's
    # `git reset --hard origin/main` silently discarded the re-anchor. Observed
    # live 2026-08-13: the node sat one unpushed commit ahead. model-promote.yml
    # already solves this by pushing over https with the workflow token; do the
    # same, and refuse to start rather than strand a commit.
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
    slug = os.getenv("GITHUB_REPOSITORY") or ""
    if token and slug:
        push_remote = f"https://x-access-token:{token}@github.com/{slug}.git"
    else:
        probe = subprocess.run(
            ["git", "push", "--dry-run", "origin", "HEAD:main"],
            cwd=REPO, capture_output=True, text=True,
        )
        if probe.returncode != 0:
            print("\n無法推送 — 這台機器的 deploy key 是唯讀，且環境沒有 "
                  "GH_TOKEN/GITHUB_REPOSITORY。")
            print("不重定錨，也不留下推不出去的 commit（下次部署的 reset --hard 會把它丟掉）。")
            print(f"git 回報：{(probe.stderr or probe.stdout).strip()[:200]}")
            return 3
        push_remote = "origin"

    print("\n→ 自動重定錨開始")
    refresh()
    # Both guarded splits, because each has its own baseline file now. Leaving
    # val un-anchored would make the val gate REFUSE on the next promotion
    # (its recorded test_images would not match the refreshed export).
    for split in GUARDED_SPLITS:
        print(f"\n→ 重定錨 {split} split")
        # --rebase-basis, not just --update-baseline: a refreshed exam changes
        # test_images, which is a basis mismatch, and --update-baseline alone
        # deliberately refuses to re-record on one. Without it every re-anchor
        # exits 1 on REFUSING TO GATE and this function can never succeed.
        r = subprocess.run(
            [sys.executable, str(REPO / "tests" / "eval_ratchet.py"),
             "--device", "mps", *EVAL_BASIS,
             "--split", split, "--update-baseline", "--rebase-basis"],
            cwd=REPO,
        )
        if r.returncode != 0:
            print(f"重定錨評測失敗（{split}）— 不提交，人工介入")
            return 3
    for cmd in (
        ["git", "add", "ml/datasets", "tests/f1_baseline.json", "tests/f1_baseline_val.json"],
        ["git", "commit", "-m", "chore: re-anchor ratchet baseline to refreshed eval set"],
        ["git", "push", push_remote, "HEAD:main"],
    ):
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            safe = " ".join("<remote>" if c is push_remote else c for c in cmd)
            print(f"git 步驟失敗: {safe}")
            return 3
    print("✓ 新考卷 + 新基準已同 commit 推上 — 候選模型現在可公平評測")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--refresh", action="store_true")
    g.add_argument("--auto-reanchor", action="store_true",
                   help="check; on test/val drift: pull fresh ndjson, re-anchor "
                        "the baseline with the CURRENT production model, commit "
                        "both — then exit 0 so the caller can evaluate candidates "
                        "against the fresh, comparable baseline")
    args = ap.parse_args()
    if args.check:
        sys.exit(check())
    if args.refresh:
        refresh()
        sys.exit(0)
    sys.exit(auto_reanchor())
