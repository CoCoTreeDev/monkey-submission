"""End-to-end model cycle: upload → train → ratchet gate → ship.

  uv run --project backend python tests/model_cycle.py \
      --upload ~/new-backgrounds/ --name exp-3 --yes

Stages (resumable — state in ml/cycle_state.json, pass --resume):
  1. upload   images (mostly production false-positive backgrounds) into the
              dataset — no-label images act as negatives
  2. train    POST /api/training/start (RTX 6000 by default, employee
              credits); polls every 5 minutes until the job finishes
  3. gate     download best.pt → eval_ratchet at the DEPLOYMENT BASIS (see
              GATE_BASIS) on BOTH test and val splits
  4. ship     swap backend/best.pt, refresh the baseline, header-only commit,
              push (CI deploys cloud + edge)

`--force-promote` ships a model the gate rejected. It does NOT skip the
measurement — both splits are still scored and the verdict is printed and
recorded, because a promotion nobody measured cannot be written up in
ml/PROMOTIONS.md, and every forced promotion this project has made needed that
scorecard afterwards. What it skips is the *abort*.

A forced promotion never touches tests/f1_baseline*.json. Lowering the floor to
fit the model is how the bar dropped twice in 2026-08 (see PROMOTIONS.md); the
floor stays where the project actually reached, so eval_ratchet keeps reporting
FAIL for a below-floor model in production instead of hiding it.

Money/irreversibility guards: `--yes` is required to start training (burns
GPU credits) and to push. Without it the script stops before each of those
and prints the exact command it would run.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "ml" / "cycle_state.json"
FETCH = [sys.executable, str(REPO / "tests" / "fetch_platform.py")]

sys.path.insert(0, str(REPO / "tests"))
import fetch_platform as fp  # noqa: E402  (reuse _get/_post/_resolve/_id)

RATCHET = [sys.executable, str(REPO / "tests" / "eval_ratchet.py")]

# The deployment basis: what the edge node actually runs, and the only settings
# tests/f1_baseline*.json were ever recorded under ("native>=1600+grayworld>1.5",
# half, conf 0.5). eval_ratchet compares this basis key-by-key and *refuses to
# gate* on any mismatch, so these flags are not decoration — the gate used to
# pass plain `--imgsz 1600 --half`, which mismatches the recorded basis, exits 1
# on "REFUSING TO GATE", and therefore could never pass for any model at all.
GATE_BASIS = ("--native", "--imgsz", "1600",
              "--gray-world", "--gray-world-above", "1.5",
              "--half", "--conf", "0.5")


def sh(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=1))


def stage_upload(args, st: dict) -> None:
    if st.get("uploaded") or not args.upload:
        return
    fp.upload_dir(Path(args.upload).expanduser(), fp._resolve("datasets", args.dataset))
    st["uploaded"] = True
    save_state(st)
    print("※ 上傳的影像未標註 — 請在平台 UI 完成標註/確認後再訓練，"
          "否則 positives 會被當成 background 學進去。--wait-labels 後繼續。")
    if args.wait_labels:
        input("平台標註完成後按 Enter 繼續訓練 …")


def stage_train(args, st: dict) -> None:
    if st.get("model_id"):
        return
    proj = fp._resolve("projects", args.project)
    ds_all = fp._items(fp._get("datasets", limit=1000))
    rec = next(x for x in ds_all if fp._id(x) == fp._id(fp._resolve("datasets", args.dataset)))
    body = {
        "projectId": fp._id(proj),
        "gpuType": args.gpu,
        "trainArgs": {"model": args.base, "data": f"ul://{rec['username']}/datasets/{rec['slug']}",
                      "epochs": args.epochs, "imgsz": args.imgsz},
        "name": args.name,
    }
    print("訓練請求:", json.dumps(body, ensure_ascii=False, indent=1))
    if not args.yes:
        sys.exit("訓練會消耗 GPU credits — 確認無誤後加 --yes 重跑（--resume 可續）")
    out = fp._post("training/start", body)
    st["model_id"] = out.get("modelId") or fp._id(out) or out.get("id")
    save_state(st)
    print("training started:", json.dumps(out, ensure_ascii=False)[:300])


def stage_wait(args, st: dict) -> None:
    if st.get("trained"):
        return
    mid = st["model_id"]
    while True:
        try:
            job = fp._get(f"models/{mid}/training")
        except Exception as exc:
            print("status poll failed:", exc)
            time.sleep(120)
            continue
        status = str(job.get("status", job)).lower()
        print(f"[{time.strftime('%H:%M')}] {str(job)[:160]}")
        if any(k in status for k in ("completed", "success", "finished")):
            break
        if any(k in status for k in ("failed", "error", "cancel")):
            sys.exit(f"training failed: {str(job)[:400]}")
        time.sleep(300)
    st["trained"] = True
    save_state(st)


def stage_gate(args, st: dict) -> None:
    if st.get("gated"):
        return
    cand = REPO / "ml" / "models" / f"{args.name}.pt"
    sh(*FETCH, "--weights", st["model_id"], "-o", str(cand))

    # Both splits are always scored, forced or not — see the module docstring.
    # check=False so one FAIL does not abort before the other split is measured;
    # a forced promotion needs the whole scorecard, not the first failure.
    verdicts = {}
    for split in ("test", "val"):
        r = sh(*RATCHET, "--model", str(cand), "--device", args.eval_device,
               *GATE_BASIS, "--split", split, check=False)
        verdicts[split] = r.returncode == 0
    st["verdicts"] = verdicts
    save_state(st)

    failed = sorted(s for s, ok in verdicts.items() if not ok)
    if not failed:
        st["gated"] = True
        save_state(st)
        return

    if not args.force_promote:
        sys.exit(f"RATCHET FAIL on {', '.join(failed)} — 不上版。"
                 f"確定要無視棘輪就加 --force-promote（會記在 commit 訊息裡，"
                 f"且不會動 baseline）")

    print(f"\n※ --force-promote：{', '.join(failed)} 沒過棘輪，仍照指示上版。\n"
          f"  baseline 不動 —— 之後 eval_ratchet 會持續對生產中的模型報 FAIL，"
          f"這是刻意的。\n"
          f"  上版後請在 ml/PROMOTIONS.md 補一列，寫清楚放棄了什麼。")
    st["gated"] = True
    st["forced"] = True
    save_state(st)


def stage_ship(args, st: dict) -> None:
    forced = bool(st.get("forced"))
    failed = sorted(s for s, ok in (st.get("verdicts") or {}).items() if not ok)

    # `forced` is persisted, so a later --resume would inherit it and ship a
    # rejected model with no flag on the command line. Consent has to be
    # present in THIS invocation, not remembered from an earlier one.
    if forced and not args.force_promote:
        sys.exit(f"cycle_state.json 記著上次是強制上版（{', '.join(failed)} FAIL）—— "
                 f"要繼續就再加一次 --force-promote，或刪掉 {STATE.name} 重跑棘輪")

    if not args.yes:
        state = f"棘輪 FAIL（{', '.join(failed)}）但 --force-promote" if forced \
            else "雙 split 都 PASS"
        sys.exit(f"{state} — 加 --yes 執行換模與 push")

    # Only push from the branch CI deploys. The old code committed to whatever
    # branch was checked out and then pushed `origin main` — on a feature branch
    # that pushes main's *old* tip, so the swap silently never deployed.
    branch = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != args.branch:
        sys.exit(f"在 {branch}，但部署分支是 {args.branch} —— "
                 f"切過去再跑，或用 --branch {branch} 明確指定")

    cand = REPO / "ml" / "models" / f"{args.name}.pt"
    shutil.copy2(cand, REPO / "backend" / "best.pt")

    paths = ["backend/best.pt"]
    if forced:
        # A forced promotion is below the floor by definition. Re-anchoring here
        # would quietly rewrite the bar to fit the model it just failed — the
        # exact move PROMOTIONS.md records as the cost of the 2026-08-17 pair.
        print("※ 強制上版：不執行 --update-baseline，baseline 保持原值")
    else:
        sh(*RATCHET, "--device", args.eval_device, *GATE_BASIS, "--update-baseline")
        paths.append("tests/f1_baseline.json")

    msg = (f"chore: force-promote {args.name} over a ratchet fail on "
           f"{', '.join(failed)}; the floor is left alone" if forced
           else f"chore: swap best.pt to {args.name} (ratchet pass)")
    sh("git", "-C", str(REPO), "add", *paths)
    sh("git", "-C", str(REPO), "commit", "-m", msg)
    sh("git", "-C", str(REPO), "push", "origin", args.branch)
    STATE.unlink(missing_ok=True)
    print("已推上 — CI 會部署雲端與 mini；部署後到節點分頁確認 FPS/空轉正常")
    if forced:
        print("※ 這是強制上版：ml/PROMOTIONS.md 還缺這一列，"
              "而且 eval_ratchet 從現在起會對生產中的模型報 FAIL")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", help="dir of images to add (backgrounds/positives)")
    ap.add_argument("--wait-labels", action="store_true",
                    help="pause after upload until platform labelling is done")
    ap.add_argument("--dataset", default="monkey cocotree")
    ap.add_argument("--project", default="monkey_v2")
    ap.add_argument("--base", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--gpu", default="rtx-6000")
    ap.add_argument("--name", required=True, help="new model name (e.g. exp-3)")
    ap.add_argument("--yes", action="store_true", help="consent for credits + push")
    ap.add_argument("--force-promote", action="store_true",
                    help="ship even though the ratchet FAILED. Still measures both "
                         "splits and still records the verdict; skips only the abort. "
                         "Never rewrites tests/f1_baseline*.json.")
    ap.add_argument("--branch", default="main",
                    help="branch CI deploys from; ship refuses to run elsewhere")
    ap.add_argument("--eval-device", default="mps",
                    help="torch device for the gate/baseline eval runs")
    ap.add_argument("--resume", action="store_true", help="continue from cycle_state.json")
    args = ap.parse_args()

    st = load_state() if args.resume else {}
    stage_upload(args, st)
    stage_train(args, st)
    stage_wait(args, st)
    stage_gate(args, st)
    stage_ship(args, st)


if __name__ == "__main__":
    main()
