"""Build the destructive-step plan for a Reference rebuild.

This module deliberately plans only.  It never opens the database, deletes a
row, or uploads a blob.  The plan is the contract an eventual human-gated
apply must consume: every source is either mapped to exactly one rebuilt clip,
or explicitly preserved because its raw archive failed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_apply_plan(
    manifest: dict,
    reference: dict,
    *,
    skip_failed: bool = False,
) -> dict:
    errors: list[str] = []
    canonical_by_source: dict[str, dict] = {}
    source_row_ids: dict[str, list[int]] = {}
    for source in reference.get("clips", []):
        trail = source.get("trail_path")
        if not trail:
            errors.append("Reference clip without trail_path")
            continue
        canonical_by_source[trail] = source
        source_row_ids[trail] = [int(row_id) for row_id in source.get("row_ids", [])]

    expected_sources = set(canonical_by_source)
    failed_sources = {
        item.get("source_trail_path") or item.get("trail_path")
        for item in manifest.get("failures", [])
    }
    failed_sources.discard(None)
    failed_sources.update(
        item.get("trail_path")
        for item in reference.get("missing_sources", [])
        if item.get("trail_path")
    )

    output_by_source: dict[str, dict] = {}
    output_paths: set[str] = set()
    for clip in manifest.get("clips", []):
        if clip.get("status") != "ready":
            errors.append(f"{clip.get('clip_id')}: output is not ready")
        if clip.get("source_kind") != "rawframes":
            errors.append(f"{clip.get('clip_id')}: output is not raw-frame based")
        output = clip.get("output_path")
        if not output:
            errors.append(f"{clip.get('clip_id')}: missing output_path")
        elif output in output_paths:
            errors.append(f"duplicate output_path: {output}")
        else:
            output_paths.add(output)
        if not clip.get("poster_path"):
            errors.append(f"{clip.get('clip_id')}: missing poster_path")
        sources = clip.get("reference_source_trails") or []
        if not sources:
            errors.append(f"{clip.get('clip_id')}: missing Reference sources")
        for source in sources:
            if source in output_by_source:
                errors.append(f"source assigned to multiple outputs: {source}")
            output_by_source[source] = clip
        if clip.get("track_count", len(clip.get("rows", []))) != len(clip.get("rows", [])):
            errors.append(f"{clip.get('clip_id')}: track_count does not match rows")

    output_sources = set(output_by_source)
    reference_source_count = reference.get("reference_clips")
    if reference_source_count is not None:
        covered_sources = len(output_sources | failed_sources)
        if covered_sources != int(reference_source_count):
            errors.append(
                f"Reference source coverage 不完整：{covered_sources}/{reference_source_count}"
            )
    unknown = sorted(output_sources - expected_sources)
    if unknown:
        errors.append(f"output contains sources outside Reference: {unknown}")
    missing = sorted(expected_sources - output_sources - failed_sources)
    if missing:
        errors.append(f"Reference sources have no output or failure: {missing}")
    unresolved = reference.get("review_required", [])
    if unresolved:
        errors.append(f"{len(unresolved)} unresolved overlap candidates")

    skipped_sources = sorted(expected_sources & failed_sources)
    if skipped_sources and not skip_failed:
        errors.append(f"raw sources failed; rerun with skip_failed to preserve them: {skipped_sources}")

    ready_sources = sorted(output_sources & expected_sources)
    old_ids = sorted(
        row_id
        for source in ready_sources
        for row_id in source_row_ids.get(source, [])
    )
    new_rows = []
    for clip in manifest.get("clips", []):
        if not set(clip.get("reference_source_trails") or []).issubset(set(ready_sources)):
            continue
        for row in clip.get("rows", []):
            new_rows.append({
                "clip_id": clip.get("clip_id"),
                "output_path": clip.get("output_path"),
                "rawframes_path": clip.get("rawframes_path"),
                "poster_path": clip.get("poster_path"),
                "image_path": row.get("image_path") or clip.get("poster_path"),
                "annotated_path": row.get("annotated_path") or clip.get("poster_path"),
                "crop_path": row.get("crop_path") or "",
                "camera_id": clip.get("camera_id"),
                "timestamp": clip.get("timestamp"),
                "track_id": row.get("track_id"),
                "animal_id": row.get("animal_id"),
                "confidence": row.get("confidence"),
                "bbox": row.get("bbox"),
                "clip_offset_seconds": row.get("clip_offset_seconds"),
                "clip_end_offset_seconds": row.get("clip_end_offset_seconds"),
                "clip_duration_seconds": row.get("clip_duration_seconds"),
                "clip_segments": row.get("clip_segments"),
                "track_boxes": row.get("boxes"),
                "track_box_fps": row.get("track_box_fps"),
            })

    if not new_rows and ready_sources:
        errors.append("ready sources produced no replacement rows or zero-track clips were not explicit")

    return {
        "run_id": manifest.get("run_id"),
        "dry_run": True,
        "apply_allowed": not errors,
        "reference_sources": len(expected_sources),
        "ready_sources": len(ready_sources),
        "skipped_sources": skipped_sources if skip_failed else [],
        "old_detection_ids_to_archive": old_ids,
        "replacement_rows": new_rows,
        "replacement_row_count": len(new_rows),
        "errors": errors,
        "rollback_required": not bool(errors),
        "rollback_note": (
            "Persist the archived old rows before deleting them; rollback must "
            "restore those rows and remove only this run's replacement rows."
            if not errors else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--skip-failed", action="store_true")
    args = parser.parse_args()
    plan = build_apply_plan(
        json.loads(args.manifest.read_text()),
        json.loads(args.reference.read_text()),
        skip_failed=args.skip_failed,
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0 if plan["apply_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
