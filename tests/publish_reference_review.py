"""Publish a completed reference rebuild to the isolated review prefix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    # The script is copied to /tmp on the mini, so make the backend package
    # explicit instead of relying on Python's script-directory sys.path entry.
    sys.path.insert(0, str(Path.cwd()))
    # Import from the mini's backend environment, where the production Blob
    # connection string is configured. No database mutation is performed.
    from services.storage import storage

    manifest = json.loads(args.manifest.read_text())
    published_manifest = json.loads(args.manifest.read_text())
    if args.reference:
        reference = json.loads(args.reference.read_text())
        # Keep the provenance next to the copied summary. Without this, a
        # rebuild executed from an earlier Reference revision can display v5
        # counts while still claiming it was produced from v2.
        published_manifest.setdefault("source", {})["reference"] = str(args.reference)
        published_manifest["reference"] = {
            "reference_clips": reference.get("reference_clips"),
            "canonical_clips": reference.get("canonical_clips"),
            "missing_sources": reference.get("missing_sources", []),
            "review_required": reference.get("review_required", []),
            "time_overlap_warnings": reference.get("time_overlap_warnings", []),
        }
    checkpoint_path = args.run_dir / "publish_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {
        "uploaded": []
    }
    uploaded = set(checkpoint.get("uploaded", []))

    def save_checkpoint() -> None:
        checkpoint_path.write_text(json.dumps({"uploaded": sorted(uploaded)}, indent=2))

    def upload_asset(path_value: str, label: str) -> str:
        filename = Path(path_value).name
        source = args.run_dir / filename
        if not source.is_file():
            raise SystemExit(f"review {label} missing: {source}")
        if filename not in uploaded:
            storage.upload(filename, source.read_bytes())
            uploaded.add(filename)
            save_checkpoint()
        return filename

    clips = manifest.get("clips", [])
    for index, clip in enumerate(clips, 1):
        output = clip.get("output_path") or ""
        # The current production API still exposes the legacy flat snapshot
        # route. Keep the published review names unique, but use that route
        # until the path-aware backend change is deployed.
        published_manifest["clips"][index - 1]["output_path"] = upload_asset(
            output, "artifact"
        )
        rawframes = clip.get("rawframes_path") or ""
        if rawframes:
            published_manifest["clips"][index - 1]["rawframes_path"] = upload_asset(
                rawframes, "raw frames"
            )
        poster = clip.get("poster_path") or ""
        if poster:
            published_manifest["clips"][index - 1]["poster_path"] = upload_asset(
                poster, "poster"
            )
        for row in published_manifest["clips"][index - 1].get("rows", []):
            for field in ("image_path", "annotated_path", "crop_path"):
                value = row.get(field) or ""
                if value:
                    row[field] = upload_asset(value, f"row {field}")
        if index % 10 == 0 or index == len(clips):
            print(f"[{index}/{len(clips)}] uploaded", flush=True)

    storage.upload(
        "rebuild-review/latest.json",
        json.dumps(published_manifest, ensure_ascii=False, indent=2).encode(),
    )
    print(json.dumps({
        "uploaded_clips": len(uploaded),
        "manifest": "rebuild-review/latest.json",
        "status": manifest.get("status"),
        "apply_allowed": manifest.get("apply_allowed", False),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
