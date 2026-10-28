"use client";

import useSWR from "swr";
import { useState, useEffect, useRef, useCallback } from "react";
import type { MutableRefObject, RefObject } from "react";
import { format } from "date-fns";
import { ChevronLeft, ChevronRight, ZoomIn, Route, Download, Loader2, Play, Pause,
  SkipBack, SkipForward, Maximize2, Eye, Trash2 } from "lucide-react";

import type { Detection, DetectionList } from "@/lib/api";
import { API_BASE, detectionFramesUrl, fetcher, setDetectionVisibility, snapshotUrl } from "@/lib/api";
import { SnapshotImg } from "@/components/snapshot-img";

interface DetectionReviewProps {
  detections: Detection[];
  isLoading: boolean;
  total?: number;
  page?: number;
  pageSize?: number;
  totalPages?: number;
  // Controlled selection (shared with the table below) + cross-page nav.
  selectedIdx: number;
  onPrev: () => void;
  onNext: () => void;
  // Phase 2: hidden when the user's `downloads` feature is off (backend also 403s).
  canDownload?: boolean;
  // Hidden for non-admins: confidence badge + metadata cell are not rendered at all.
  showConfidence?: boolean;
  // Admin-only: manual override of the non-admin confidence gate.
  isAdmin?: boolean;
  onVisibilityChange?: () => void;
  /**
   * Opt-in per-record delete, mirroring the `VisibilityAction` shape below it
   * (action + busy + error all supplied by the owner). Every consumer that only
   * reads omits these, so its pane renders exactly as before — DetectionLog
   * passes nothing. Deleting is a HARD delete with no undo, so the confirm and
   * the lock rule stay with the owner that already implements them.
   */
  onDelete?: (detection: Detection) => void;
  deleting?: boolean;
  deleteError?: string | null;
}

// Two views, not three. A detection has ONE clip: whatever the detector wrote at
// emit time (`trail_path`). A re-drawn per-visit clip used to be offered beside
// it, resolved through an index blob — that blob is gone and the code reading it
// had been returning nothing on every row for some time, so it is deleted rather
// than left looking live. When one visit really does become one clip
// it will replace `trail_path` itself rather than shadow it.
// The original draws only tracks that cleared the emit gate, so a wall holding six
// macaques rendered with one box on it, and one visit was filed as several
// near-identical fragments. Showing the original as the primary clip showed
// users the system's own false negatives.
type ViewMode = "annotated" | "clip";

interface LightboxTarget {
  filename: string;
  fallbackFilename?: string;
}

export function DetectionReview({
  detections,
  isLoading,
  total,
  page = 1,
  pageSize = 30,
  totalPages = 1,
  selectedIdx,
  onPrev,
  onNext,
  canDownload = true,
  showConfidence = true,
  isAdmin = false,
  onVisibilityChange,
  onDelete,
  deleting = false,
  deleteError = null,
}: DetectionReviewProps) {
  const [visSaving, setVisSaving] = useState(false);
  const [visError, setVisError] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<LightboxTarget | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("annotated");
  const [heroLoading, setHeroLoading] = useState(true);
  const [showSpinner, setShowSpinner] = useState(false);
  // The clip loads ONLY after an explicit user action (播放影片 / 影片 toggle),
  // never eagerly. Once fetched for a detection the element stays mounted, so
  // toggling 標註↔影片 never refetches.
  const [trailRequested, setTrailRequested] = useState(false);
  const [trailReady, setTrailReady] = useState(false);
  const [trailBuffering, setTrailBuffering] = useState(false);
  const playbackRef = useRef<{
    clip: string | null;
    time: number;
    playing: boolean;
  }>({ clip: null, time: 0, playing: false });
  const playbackRateRef = useRef(1);
  const pendingPlayRef = useRef(false);

  const clampedIdx = Math.min(Math.max(selectedIdx, 0), Math.max(detections.length - 1, 0));
  const selected = detections[clampedIdx] ?? null;
  /* Declared here rather than beside the other clip fields because the effects
     below key on it: what a row shows is decided by which clip it points at. */
  const clipFile = selected?.trail_path || null;
  /** The row the pointer is over in the deck, outlined on the frame below. */
  const [hoveredRow, setHoveredRow] = useState<Detection | null>(null);
  const overallIndex = (page - 1) * pageSize + clampedIdx + 1;

  const atGlobalFirst = page === 1 && clampedIdx === 0;
  const atGlobalLast =
    page >= totalPages && clampedIdx >= detections.length - 1;

  // Switching records abandons any in-flight trail download: the trail element
  // unmounts (SnapshotImg aborts incomplete loads) and we fall back to the
  // small annotated JPEG for the new record.
  const selectedId = selected?.id ?? null;
  /* Reset when the CLIP changes, not when the selection does.
   *
   * One clip now carries every animal of a visit — up to twenty rows — so
   * stepping through them is stepping through one video. Resetting per row
   * tore the player down and rebuilt it each time, which is why the readout
   * fell back to 0.0 and every switch landed on a different frame instead of
   * on the animal you had just selected. Same clip: keep playing, and let the
   * seek effect below move the playhead to the new row.
   *
   * A different clip only clears `trailReady`, and deliberately not the mode
   * or the request: someone stepping through records with the player open
   * wants the next clip, not a still picture and another click. The in-flight
   * download of the clip being left is aborted by the <img>/<video> teardown
   * that already exists, so stepping quickly does not pile up loads. */
  useEffect(() => {
    setTrailReady(false);
    setTrailBuffering(false);
  }, [clipFile]);

  /* Every row drawn from the CLIP on screen — not every row of the visit.
   *
   * A visit is not one video: 146 of 330 visits are spread over more than one
   * clip file and one of them over thirteen (measured 2026-08-29, 617 of 812
   * rows affected). Grouping the lanes by visit therefore drew animals from
   * OTHER videos onto this video's axis, which is why two rows of the same
   * visit showed different lengths, different colours and positions that did
   * not line up with anything on screen.
   *
   * The clip is the axis, so the clip decides who has a lane. They are already
   * in the page the list handed us, so no second request. */
  /* Asked for by clip, not filtered out of the page.
   *
   * The rows of one clip are scattered wherever their timestamps put them:
   * five rows of the 2026-08-27 18:35 visit sit at list positions 29-33, so
   * filtering the 30-row page showed a timeline with two animals on a clip
   * that holds five. The page is a reading order, not a grouping. */
  const { data: clipList, isLoading: fetchedClipRowsLoading } = useSWR<DetectionList>(
    clipFile
      ? `${API_BASE}/api/detections?limit=200&trail_path=${encodeURIComponent(clipFile)}`
      : null,
    fetcher,
  );
  const clipRows = clipFile
    ? clipList?.items.filter((row) => row.trail_path === clipFile) ?? []
    : selected
      ? [selected]
      : [];
  const clipRowsLoading = fetchedClipRowsLoading;
  const clipDuration = selected?.clip_duration_seconds ?? null;

  // Open the clip AT the animal, not at the beginning. This is the whole point
  // of storing the offsets: a visit with seven macaques used to mean seven
  // rows that each played the same video from 0:00 and left you to find them.
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const seekTo = useCallback((seconds: number, exact = false) => {
    const v = videoRef.current;
    if (!v || !Number.isFinite(seconds)) return;
    // A clip is short; landing a beat early shows the animal walking in rather
    // than mid-stride, and is far easier to read. Scrubbing the axis is the
    // exception — there the pointer names the frame, so honour it exactly.
    //
    // Seeking does NOT start playback. Selecting a row means "show me this
    // one", so the clip parks on that animal and waits; pressing play is a
    // separate decision. Nothing here autoplays for the same reason.
    const target = Math.max(0, exact ? seconds : seconds - 0.4);
    if (typeof v.fastSeek === "function" && !exact) v.fastSeek(target);
    else v.currentTime = target;
  }, []);
  const seekTarget = selected?.clip_offset_seconds ?? null;
  /* Seek the moment the element knows its duration, not when it can play.
   *
   * Merging a visit onto one clip made the files much larger — 3.5 MB on
   * average, 40 MB at the worst — so waiting for `canplay` meant waiting for
   * most of the download before the playhead moved, and the readout sat at 0.0
   * the whole time. Setting `currentTime` at `loadedmetadata` instead makes the
   * browser range-request the part the animal is actually in, which is the
   * whole reason the offsets exist. */
  useEffect(() => {
    const v = videoRef.current;
    if (!v || seekTarget === null) return;
    // A row's offset is the first frame where that animal appears. Selecting
    // it must land on that frame, not a neighbouring row's later position.
    const go = () => seekTo(seekTarget, true);
    if (v.readyState >= 1) {
      go();
      return;
    }
    v.addEventListener("loadedmetadata", go, { once: true });
    return () => v.removeEventListener("loadedmetadata", go);
  }, [selectedId, seekTarget, seekTo]);

  const annotatedFile = selected
    ? selected.annotated_path || selected.image_path || selected.crop_path
    : null;
  const annotatedFallbackFile = selected && annotatedFile !== selected.image_path
    ? selected.image_path || selected.crop_path || undefined
    : undefined;
  const cropFile = selected
    ? selected.crop_path || selected.image_path || selected.annotated_path
    : null;
  const cropFallbackFile = selected && cropFile !== selected.image_path
    ? selected.image_path || selected.annotated_path || undefined
    : undefined;
  // Clips written since 2026-08-16 are H.264 MP4 (84x smaller than the GIF they
  // replaced); everything older is still a GIF, which renders through <img> and
  // therefore gets no playback controls — that is what the rebuild fixes.
  const clipIsVideo = !!clipFile?.toLowerCase().endsWith(".mp4");
  const clipShown = viewMode === "clip" && trailRequested && trailReady && !!clipFile;
  const clipLoading = viewMode === "clip" && trailRequested && !trailReady && !!clipFile;
  // The filename currently shown in the hero. Videos use the native player and
  // therefore intentionally do not enter the image-only lightbox.
  const heroFile = clipShown && clipFile && !clipIsVideo ? clipFile : annotatedFile;

  const requestTrail = useCallback((autoPlay = false) => {
    if (!clipFile) return;
    pendingPlayRef.current = autoPlay;
    setTrailRequested(true);
    setViewMode("clip");
  }, [clipFile]);

  // Show a loader on the hero while a newly-selected annotated still downloads.
  // Cleared by the annotated <img>'s onLoad; the trail has its own overlay.
  useEffect(() => {
    setHeroLoading(true);
  }, [annotatedFile]);

  // Anti-flash: only surface the spinner if loading lingers past 200ms — cached
  // images paint faster and shouldn't flash a spinner.
  useEffect(() => {
    if (!heroLoading) {
      setShowSpinner(false);
      return;
    }
    const t = setTimeout(() => setShowSpinner(true), 200);
    return () => clearTimeout(t);
  }, [heroLoading]);

  // Keyboard navigation: ←/→ to step through detections (drives shared state).
  useEffect(() => {
    if (lightbox) return; // lightbox handler owns Escape; arrows still nav
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) {
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        onPrev();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onNext();
      } else if (e.key === " " || e.key === "," || e.key === ".") {
        // Only while a clip is mounted, so these keys stay free everywhere
        // else. `,` and `.` are the editor's frame-step bindings; Left and
        // Right keep the record navigation they already owned.
        const v = videoRef.current;
        if (!v) return;
        e.preventDefault();
        if (e.key === " ") {
          void (v.paused ? v.play().catch(() => {}) : v.pause());
        } else {
          v.pause();
          v.currentTime = Math.max(
            0,
            Math.min(v.duration || 0, v.currentTime + (e.key === "." ? FRAME : -FRAME)),
          );
        }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [lightbox, onPrev, onNext]);

  const closeLightbox = useCallback(() => setLightbox(null), []);
  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => {
      // Stop the parent modal from also closing on the same Escape press.
      if (e.key === "Escape") {
        e.stopPropagation();
        closeLightbox();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [lightbox, closeLightbox]);

  return (
    <>
      {lightbox && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={closeLightbox}
        >
          <SnapshotImg
            filename={lightbox.filename}
            fallbackFilename={lightbox.fallbackFilename}
            alt="放大檢視"
            className="max-h-[92vh] max-w-[95vw] rounded-lg object-contain"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}

      <div className="@container flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border bg-white">
        {/* Header */}
        <div className="flex shrink-0 items-center justify-between border-b px-5 py-3">
          <div className="flex items-baseline gap-2">
            <h3 className="text-sm font-semibold">偵測檢閱</h3>
            <span className="text-[11px] text-muted-foreground">
              ← / → 切換
            </span>
          </div>
          {detections.length > 0 && (
            <span className="text-xs text-muted-foreground">
              第 <span className="font-mono font-semibold text-foreground">{overallIndex}</span> 筆
              {typeof total === "number" && <> / 共 {total}</>}
              {totalPages > 1 && (
                <> · 頁 <span className="font-mono">{page}/{totalPages}</span></>
              )}
            </span>
          )}
        </div>

        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto p-3 sm:p-5">
          {isLoading ? (
            <div className="flex flex-1 gap-4">
              <div className="flex-1 animate-pulse rounded bg-muted" />
            </div>
          ) : detections.length === 0 ? (
            <div className="flex flex-1 items-center justify-center py-16 text-center text-sm text-muted-foreground">
              尚無偵測資料可供檢閱。
            </div>
          ) : selected ? (
            <div className="flex flex-col gap-4">
              {/* View toggle + download frames */}
              <div className="flex shrink-0 flex-wrap items-center gap-3">
              <div className="flex gap-1 rounded-lg border bg-muted p-0.5 w-fit">
                <button
                  className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                    viewMode === "clip"
                      ? "bg-white text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  onClick={() => requestTrail()}
                  disabled={!clipFile}
                  title={clipFile ? "載入並播放這次造訪的影片" : "此偵測沒有影片"}
                >
                  <Route className="h-3 w-3" />
                  影片
                </button>
                <button
                  className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                    viewMode === "annotated"
                      ? "bg-white text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  onClick={() => setViewMode("annotated")}
                >
                  <ZoomIn className="h-3 w-3" />
                  標註
                </button>
              </div>
              {selected.trail_path && canDownload && (
                <a
                  href={detectionFramesUrl(selected.id)}
                  download
                  className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground"
                  title="下載此偵測的所有原始影格為 JPG（適用於資料集標記）"
                >
                  <Download className="h-3 w-3" />
                  下載全部幀
                </a>
              )}
              {onDelete && (
                <div className="ml-auto flex items-center gap-2">
                  {deleteError && (
                    <span className="text-[11px] text-red-600">{deleteError}</span>
                  )}
                  <button
                    onClick={() => onDelete(selected)}
                    disabled={deleting || selected.locked}
                    title={
                      selected.locked
                        ? "已鎖定：需先解除鎖定才能刪除"
                        : "刪除這筆紀錄及其圖片（硬刪，無法復原）"
                    }
                    className="flex items-center gap-1.5 rounded-md border border-red-200 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-40"
                  >
                    {deleting ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Trash2 className="h-3 w-3" />
                    )}
                    刪除
                  </button>
                </div>
              )}
              {isAdmin && (
                <VisibilityAction
                  detection={selected}
                  saving={visSaving}
                  error={visError}
                  onSet={async (visible) => {
                    setVisSaving(true);
                    setVisError(null);
                    try {
                      await setDetectionVisibility(selected.id, visible);
                      onVisibilityChange?.();
                    } catch (e) {
                      setVisError(e instanceof Error ? e.message : "更新失敗");
                    } finally {
                      setVisSaving(false);
                    }
                  }}
                />
              )}
              </div>

              {/* Player and its timeline hold the left; the record's own
                  fields move to a rail on the right, where a wide screen
                  has the room. Stack according to the pane width, including in modals. */}
              <div className="flex min-w-0 shrink-0 flex-col gap-3 @min-[56rem]:flex-row @min-[56rem]:gap-4">
                <div className="flex min-w-0 flex-1 flex-col gap-3 fullscreen:overflow-y-auto fullscreen:bg-neutral-900 fullscreen:p-3">
                {/* The annotated JPEG appears immediately; the clip mounts only
                    after 播放影片, so browsing records never preloads video. */}
                <div className="group relative flex h-[clamp(10rem,min(50cqw,40dvh),32rem)] shrink-0 items-center justify-center overflow-hidden rounded-lg border bg-neutral-900">
                  {annotatedFile ? (
                    <SnapshotImg
                      filename={annotatedFile}
                      fallbackFilename={annotatedFallbackFile}
                      alt="標註畫面"
                      className={`h-full w-full cursor-pointer object-contain ${
                        clipShown ? "hidden" : ""
                      }`}
                      onClick={() =>
                        setLightbox({
                          filename: annotatedFile,
                          fallbackFilename: annotatedFallbackFile,
                        })
                      }
                      onLoad={() => setHeroLoading(false)}
                      onError={() => setHeroLoading(false)}
                    />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-xs text-white/50">
                      無圖片
                    </div>
                  )}
                  {trailRequested && clipFile && clipIsVideo ? (
                    <video
                      key={clipFile}
                      ref={videoRef}
                      src={snapshotUrl(clipFile)}
                      muted
                      playsInline
                      preload="auto"
                      autoPlay={pendingPlayRef.current}
                      // No `controls`. The browser's bar is a second scrubber
                      // sitting on top of the one below, drawn in a style
                      // nothing else here shares, and it covered the badges.
                      // The deck under the player owns transport and scrubbing,
                      // and its scrubber is the animal tracks themselves — the
                      // thing you drag is the thing that shows you where the
                      // monkeys are.
                      //
                      // `loop` is safe now that it is: a restart only happens
                      // at a natural end, and scrubbing sets currentTime
                      // directly, so dragging through the last second no
                      // longer fights a clip that keeps jumping back to zero.
                      // These are 5-30 s of evidence watched over and over to
                      // decide whether two rows are the same animal — having
                      // to press play for each pass was the friction.
                      loop
                      onLoadedMetadata={(e) => {
                        const v = e.currentTarget;
                        const saved = playbackRef.current;
                        const sameClip = saved.clip === clipFile;
                        const target = sameClip
                          ? saved.time
                          : selected?.clip_offset_seconds !== null && selected?.clip_offset_seconds !== undefined
                            ? selected.clip_offset_seconds
                            : 0;
                        if (Number.isFinite(target)) v.currentTime = target;
                        // Preserve an active review pass when the selected row
                        // points at another canonical clip. The old check only
                        // resumed when `sameClip`, so crossing a clip boundary
                        // silently paused playback and required a second click.
                        if (pendingPlayRef.current || saved.playing) {
                          v.playbackRate = playbackRateRef.current;
                          pendingPlayRef.current = true;
                          void v.play().then(() => {
                            pendingPlayRef.current = false;
                          }).catch(() => {
                            // Safari/WebKit may reject while the range request is
                            // still warming; onCanPlay will retry once it is usable.
                          });
                        }
                      }}
                      onClick={() => {
                        const v = videoRef.current;
                        if (!v) return;
                        if (v.paused) {
                          pendingPlayRef.current = true;
                          void v.play().then(() => {
                            pendingPlayRef.current = false;
                          }).catch(() => {});
                        } else {
                          pendingPlayRef.current = false;
                          v.pause();
                        }
                      }}
                      onTimeUpdate={(e) => {
                        const v = e.currentTarget;
                        playbackRef.current = {
                          clip: clipFile,
                          time: v.currentTime,
                          playing: !v.paused,
                        };
                      }}
                      onPlay={(e) => {
                        pendingPlayRef.current = false;
                        setTrailBuffering(false);
                        const v = e.currentTarget;
                        playbackRef.current = { clip: clipFile, time: v.currentTime, playing: true };
                      }}
                      onPause={(e) => {
                        const v = e.currentTarget;
                        playbackRef.current = { clip: clipFile, time: v.currentTime, playing: false };
                      }}
                      onWaiting={() => {
                        setTrailBuffering(true);
                      }}
                      onStalled={() => {
                        setTrailBuffering(true);
                      }}
                      onPlaying={() => {
                        setTrailBuffering(false);
                      }}
                      className={`h-full w-full cursor-pointer object-contain ${
                        clipShown ? "" : "hidden"
                      }`}
                      onCanPlay={(e) => {
                        e.currentTarget.playbackRate = playbackRateRef.current;
                        setTrailReady(true);
                        setTrailBuffering(false);
                        if (pendingPlayRef.current) {
                          void e.currentTarget.play().then(() => {
                            pendingPlayRef.current = false;
                          }).catch(() => {});
                        }
                      }}
                      onError={() => {
                        pendingPlayRef.current = false;
                        setTrailRequested(false);
                        setTrailReady(false);
                        setTrailBuffering(false);
                        setViewMode("annotated");
                      }}
                    />
                  ) : trailRequested && clipFile ? (
                    <SnapshotImg
                      key={clipFile}
                      filename={clipFile}
                      alt="這次造訪的影片"
                      className={`h-full w-full cursor-pointer object-contain ${
                        clipShown ? "" : "hidden"
                      }`}
                      onClick={() => setLightbox({ filename: clipFile })}
                      onLoad={() => setTrailReady(true)}
                    />
                  ) : null}

                  {/* Trail streaming — annotated still stays visible behind */}
                  {clipLoading && (
                    <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/40 text-white/80">
                      <Loader2 className="h-6 w-6 animate-spin" />
                      <span className="text-xs">載入影片中…</span>
                    </div>
                  )}
                  {clipShown && trailBuffering && (
                    <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-black/20 text-white/80">
                      <Loader2 className="h-5 w-5 animate-spin" />
                    </div>
                  )}

                  {/* Loading overlay — while the annotated still downloads */}
                  {showSpinner && annotatedFile && !clipShown && !clipLoading && (
                    <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 text-white/60">
                      <Loader2 className="h-6 w-6 animate-spin" />
                      <span className="text-xs">載入中…</span>
                    </div>
                  )}

                  {/* Hovering a lane outlines that animal on the frame.
                      A canvas rather than a positioned <div>: it paints only
                      while the pointer is on a lane and clears the moment it
                      leaves, so nothing is laid out, nothing re-renders, and
                      the browser does no work at all the rest of the time.

                      The box is the one RECORDED for that detection, not a
                      per-frame track — we do not store per-frame boxes — so it
                      answers "which of these is #620", not "where is it right
                      now". It is drawn dimmed once the playhead is outside
                      that animal's stretches, which is the honest way to say
                      the position is from another moment. */}
                  {clipShown && (
                    <HoverOutline row={hoveredRow} videoRef={videoRef} />
                  )}

                  {/* Overlay info — top left, where nothing else sits: the
                      zoom button owns top-right and the deck owns everything
                      below the frame. The id carries the animal's own colour,
                      the same one its lane is drawn in. */}
                  <div className="absolute left-2 top-2 z-10 flex max-w-[calc(100%-3.5rem)] flex-wrap gap-1.5">
                    {selected.track_id >= 0 && (
                      <span
                        className="rounded px-2 py-0.5 font-mono text-[11px] font-semibold tabular-nums text-neutral-950"
                        style={{ backgroundColor: trackFill(trackToneIndex(trackColourKey(selected))) }}
                      >
                        #{selected.track_id}
                      </span>
                    )}
                    {showConfidence && (
                      <span className="rounded bg-black/60 px-2 py-0.5 text-[11px] font-semibold text-white">
                        {(selected.confidence * 100).toFixed(1)}%
                      </span>
                    )}
                    <span className="rounded bg-black/60 px-2 py-0.5 text-[11px] text-white/80">
                      {selected.camera_id}
                    </span>
                    {isAdmin && selected.visible_to_users === false && (
                      <span className="rounded bg-slate-700/90 px-2 py-0.5 text-[11px] font-medium text-white">
                        使用者不可見
                      </span>
                    )}
                    {isAdmin && selected.user_visible === true && (
                      <span className="rounded bg-emerald-600/90 px-2 py-0.5 text-[11px] font-medium text-white">
                        已手動推送
                      </span>
                    )}
                    {isAdmin && selected.user_visible === false && (
                      <span className="rounded bg-amber-600/90 px-2 py-0.5 text-[11px] font-medium text-white">
                        已手動隱藏
                      </span>
                    )}
                  </div>

                  {/* Zoom */}
                  <button
                    aria-label="放大標註畫面"
                    className="absolute right-2 top-2 rounded bg-black/50 p-1.5 text-white opacity-0 transition group-hover:opacity-100"
                    onClick={() =>
                      heroFile &&
                      setLightbox({
                        filename: heroFile,
                        fallbackFilename: heroFile === annotatedFile ? annotatedFallbackFile : undefined,
                      })
                    }
                  >
                    <ZoomIn className="h-4 w-4" />
                  </button>
                </div>

                {/* The row's timeline, not the player's accessory: it shows
                    the moment a row with a clip is selected, because "when was
                    each animal on screen" is readable before anything is
                    playing. Its play button is what fetches the video — the
                    overlay button in the middle of the still was the same
                    control drawn a second time. */}
                {clipFile && clipIsVideo && (
                  <VisitTimeline
                    key={clipFile || `detection-${selected.id}`}
                    rows={clipRows}
                    duration={clipDuration}
                    selectedId={selected?.id ?? null}
                    videoRef={videoRef}
                    onSeek={seekTo}
                    onLoadClip={requestTrail}
                    loading={clipLoading}
                    rowsLoading={clipRowsLoading}
                    onHoverRow={setHoveredRow}
                    playbackRateRef={playbackRateRef}
                  />
                )}
                </div>

                <aside className="flex shrink-0 flex-col gap-3 @min-[56rem]:w-72 @min-[56rem]:self-start">
                {/* The record's own fields. A strip under the player on narrow screens,
                    a rail beside it once there is width to spare. */}
                <div className="flex shrink-0 flex-col gap-3 rounded-lg border p-3 sm:flex-row sm:items-center @min-[56rem]:flex-col @min-[56rem]:items-stretch">
                  <button
                    className="flex h-24 w-32 shrink-0 items-center justify-center overflow-hidden rounded-md border bg-neutral-900 @min-[56rem]:h-36 @min-[56rem]:w-full"
                    onClick={() =>
                      cropFile &&
                      setLightbox({
                        filename: cropFile,
                        fallbackFilename: cropFallbackFile,
                      })
                    }
                    title="放大裁切圖"
                  >
                    {cropFile ? (
                      <SnapshotImg
                        filename={cropFile}
                        fallbackFilename={cropFallbackFile}
                        alt="裁切圖"
                        className="h-full w-full cursor-pointer object-contain"
                      />
                    ) : (
                      <span className="text-[11px] text-white/30">無裁切圖</span>
                    )}
                  </button>
                  <div
                    className={`grid flex-1 grid-cols-2 gap-x-6 gap-y-3 @min-[56rem]:grid-cols-1 @min-[56rem]:gap-y-2.5 ${
                      showConfidence ? "sm:grid-cols-4" : "sm:grid-cols-3"
                    }`}
                    >
                      <div>
                        <div className="text-[11px] text-muted-foreground">追蹤 ID</div>
                        <div
                          className="font-mono text-sm font-semibold tabular-nums"
                          style={
                            selected.track_id >= 0
                              ? { color: trackOnLight(trackToneIndex(trackColourKey(selected))) }
                              : undefined
                          }
                        >
                          {selected.track_id >= 0 ? `#${selected.track_id}` : "--"}
                        </div>
                      </div>
                      {showConfidence && (
                        <div>
                          <div className="text-[11px] text-muted-foreground">信心度</div>
                          <div className="font-mono text-sm font-semibold">
                            {(selected.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      )}
                      <div>
                        <div className="text-[11px] text-muted-foreground">框選範圍</div>
                        <div className="font-mono text-sm text-foreground">
                          {Math.round(selected.bbox_w)}×{Math.round(selected.bbox_h)}
                          <span className="ml-1 text-muted-foreground">
                            @{Math.round(selected.bbox_x)},{Math.round(selected.bbox_y)}
                          </span>
                        </div>
                      </div>
                      <div>
                        <div className="text-[11px] text-muted-foreground">時間</div>
                        <div className="font-mono text-sm text-foreground">
                          {format(new Date(selected.timestamp), "MM-dd HH:mm:ss")}
                        </div>
                      </div>
                  </div>
                </div>

                {/* Nav — prev/next centered. Crosses page boundaries
                    automatically; selection is shared with the table below. */}
                <div className="flex shrink-0 items-center justify-center gap-3 @min-[56rem]:flex-col @min-[56rem]:gap-2">
                  <button
                    className="flex items-center justify-center gap-1.5 rounded-md border bg-white px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-30 @min-[56rem]:w-full"
                    disabled={atGlobalFirst}
                    onClick={onPrev}
                    title="上一筆 (←)"
                  >
                    <ChevronLeft className="h-4 w-4" />
                    上一筆
                  </button>
                  <div className="min-w-12 text-center font-mono text-sm tabular-nums text-muted-foreground @min-[56rem]:order-first">
                    {detections.length > 0 ? overallIndex : "—"}
                    {detections.length > 0 && typeof total === "number" && <> / {total}</>}
                  </div>
                  <button
                    className="flex items-center justify-center gap-1.5 rounded-md border bg-white px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-30 @min-[56rem]:w-full"
                    disabled={atGlobalLast}
                    onClick={onNext}
                    title="下一筆 (→)"
                  >
                    下一筆
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
                </aside>
              </div>

              <p className="shrink-0 text-center text-[11px] text-muted-foreground">
                從左側清單點選任一筆，即可在此檢視其軌跡與標註
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </>
  );
}


/** Admin-only override of the non-admin confidence gate.
 *
 * The gate is a blunt rule (new records need >= the confidence threshold), so
 * it gets two cases wrong: a genuine low-confidence sighting the client should
 * see, and a false positive above the line. This is the human in the loop.
 */
function VisibilityAction({
  detection,
  saving,
  error,
  onSet,
}: {
  detection: Detection;
  saving: boolean;
  error: string | null;
  onSet: (visible: boolean | null) => void;
}) {
  const override = detection.user_visible;
  const visible = detection.visible_to_users !== false;

  let label: string;
  let next: boolean | null;
  let tone: string;
  if (override === true) {
    label = "恢復預設（目前手動推送）";
    next = null;
    tone = "border-emerald-300 text-emerald-700 hover:bg-emerald-50";
  } else if (override === false) {
    label = "恢復預設（目前手動隱藏）";
    next = null;
    tone = "border-amber-300 text-amber-700 hover:bg-amber-50";
  } else if (!visible) {
    label = "推送給使用者";
    next = true;
    tone = "border-emerald-300 text-emerald-700 hover:bg-emerald-50";
  } else {
    label = "對使用者隱藏";
    next = false;
    tone = "text-muted-foreground hover:bg-muted hover:text-foreground";
  }

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        disabled={saving}
        onClick={() => onSet(next)}
        className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium disabled:opacity-50 ${tone}`}
        title="覆寫信心度門檻，決定一般使用者能否看到這筆偵測"
      >
        {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Eye className="h-3 w-3" />}
        {label}
      </button>
      {error && <span className="text-xs text-red-600">{error}</span>}
    </div>
  );
}


/* ── The visit timeline ──────────────────────────────────────────────────
 *
 * Modelled on SAM 2's demo player, minus the masks: the lanes are the object
 * list, a block is a stretch where that object is on screen, and a playhead
 * ties both to the frame you are actually looking at.
 *
 * The playhead is the part worth copying. Without it the lanes are a legend —
 * you can click one and the video jumps, but nothing tells you where the video
 * now is or which animals are on screen at this instant. With it the same
 * markup becomes a control, and "which of these rows are the same monkey" is a
 * question you can answer by watching rather than by guessing from timestamps.
 */

/** A stable colour per Reference animal.
 *
 * Tracker ids are implementation details and can change after a rebuild. Use
 * the Reference animal id when available, then fall back to camera + tracker
 * for old rows. A fixed, colour-blind-friendly palette keeps adjacent lanes
 * distinguishable without relying on row order.
 */
const TRACK_PALETTE = [
  { dark: "#67e8f9", light: "#0e7490", fill: "#06b6d4" },
  { dark: "#fbbf24", light: "#a16207", fill: "#f59e0b" },
  { dark: "#fb7185", light: "#be123c", fill: "#f43f5e" },
  { dark: "#86efac", light: "#15803d", fill: "#22c55e" },
  { dark: "#c4b5fd", light: "#6d28d9", fill: "#8b5cf6" },
  { dark: "#7dd3fc", light: "#0369a1", fill: "#0ea5e9" },
  { dark: "#f0abfc", light: "#a21caf", fill: "#d946ef" },
  { dark: "#fdba74", light: "#c2410c", fill: "#f97316" },
] as const;

function trackToneIndex(value: string | number | null | undefined): number {
  const text = String(value ?? "0");
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) % TRACK_PALETTE.length;
}

function trackColourKey(row: Pick<Detection, "animal_id" | "camera_id" | "track_id">): string {
  return row.animal_id || String(row.track_id);
}

/* One animal, one colour, everywhere it appears — the badge burned over the
   video, the id in the rail, the lane and its blocks. Whether two rows are the
   same animal is the judgement this panel exists for, and making the reader
   carry a number between three differently-coloured places is the part that
   made it hard. Two inks because the panel is dark and the rail is light, and
   a single hue is not legible on both. */
const trackOnDark = (tone: number) => TRACK_PALETTE[tone].dark;
const trackOnLight = (tone: number) => TRACK_PALETTE[tone].light;
const trackFill = (tone: number) => TRACK_PALETTE[tone].fill;

/** A nominal frame for stepping. Clips are written at 5–25 fps (see
 *  `_clip_fps` on the producer), and 25 is the ceiling — a step of 1/25 s
 *  never skips past a frame on a slower clip, it lands on the same one twice. */
const FRAME = 1 / 25;

/** Ticks the eye can actually count, at any clip length. */
function tickStep(duration: number): number {
  if (duration <= 6) return 1;
  if (duration <= 15) return 2;
  if (duration <= 40) return 5;
  if (duration <= 120) return 15;
  return 30;
}

interface VisitTimelineProps {
  rows: Detection[];
  /** What the row claims, which may be nothing: rows written before the
   *  producer recorded clip timing have none, and the element's own duration
   *  is the axis regardless. */
  duration: number | null;
  selectedId: number | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  onSeek: (seconds: number, exact?: boolean) => void;
  /** Fetch and mount the clip. The deck shows before the video does, so its
   *  play button is what starts the download — there is no second one. */
  onLoadClip: (autoPlay?: boolean) => void;
  loading: boolean;
  rowsLoading: boolean;
  /** Which row the pointer is over, so the player can outline that animal. */
  onHoverRow: (row: Detection | null) => void;
  /** Playback speed survives keyed video replacement between canonical clips. */
  playbackRateRef: MutableRefObject<number>;
}

function VisitTimeline({
  rows, duration: storedDuration, selectedId, videoRef, onSeek, onLoadClip, loading,
  rowsLoading, onHoverRow,
  playbackRateRef,
}: VisitTimelineProps) {
  const [scrubbing, setScrubbing] = useState(false);
  /** The axis is the VIDEO's length, not the length the row claims.
   *
   * The producer used to measure offsets in wall clock while writing the clip
   * at a constant rate, so a row could describe a video 1.8x longer than the
   * one it points at (measured on rows 16318/16319, 2026-08-28). That is fixed
   * at the source, but the element in front of us is the thing being scrubbed,
   * so its own duration is what the axis has to be drawn in. */
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  // While a keyed video is being replaced, do not let the previous video's
  // measured duration leak into the new axis. The loading flag is already the
  // external state transition; deriving from it avoids a synchronous effect
  // setState and the extra render that caused.
  const duration = loading ? 0 : videoDuration ?? storedDuration ?? 0;
  const stripRef = useRef<HTMLDivElement | null>(null);
  const lineRef = useRef<HTMLSpanElement | null>(null);
  const bubbleRef = useRef<HTMLSpanElement | null>(null);
  /** The playhead in tenths of a second. Only what the eye can read: the line
   *  itself is moved by writing to the DOM every frame, so React re-renders at
   *  most ten times a second, and only to relabel the readout and re-light the
   *  lanes whose animal is on screen. Driving 20 lanes off a 60 fps setState
   *  re-renders the whole panel every frame for a number nobody can read. */
  const [tenths, setTenths] = useState(0);
  const playhead = tenths / 10;
  const [paused, setPaused] = useState(true);
  const selectedLaneRef = useRef<HTMLDivElement | null>(null);
  const [rate, setRate] = useState(1);
  const deckRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const read = () => {
      if (Number.isFinite(v.duration) && v.duration > 0) setVideoDuration(v.duration);
    };
    read();
    v.addEventListener("loadedmetadata", read);
    v.addEventListener("durationchange", read);
    return () => {
      v.removeEventListener("loadedmetadata", read);
      v.removeEventListener("durationchange", read);
    };
  }, [videoRef, loading]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !(duration > 0)) return;
    let raf = 0;
    const paint = () => {
      const t = Math.min(duration, Math.max(0, v.currentTime));
      const ratio = t / duration;
      const left = `${ratio * 100}%`;
      if (lineRef.current) lineRef.current.style.left = left;
      if (bubbleRef.current) {
        bubbleRef.current.style.left = left;
        bubbleRef.current.style.transform =
          ratio > 0.9 ? "translateX(-100%)" : ratio < 0.1 ? "none" : "translateX(-50%)";
        bubbleRef.current.textContent = t.toFixed(1);
      }
      // Same value in means React bails before re-rendering, so this is cheap
      // on the ~59 frames a second where the tenth has not turned over.
      setTenths(Math.round(t * 10));
    };
    const loop = () => {
      paint();
      raf = requestAnimationFrame(loop);
    };
    const start = () => {
      setPaused(false);
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(loop);
    };
    // Only while paused: during playback the rAF loop already owns the
    // playhead, and cancelling it four times a second is what freezes it.
    const syncIfIdle = () => {
      if (v.paused) paint();
    };
    const stop = () => {
      setPaused(true);
      cancelAnimationFrame(raf);
      raf = 0;
      paint();
    };
    paint();
    setPaused(v.paused);
    if (!v.paused) start();
    v.addEventListener("play", start);
    v.addEventListener("playing", start);
    v.addEventListener("pause", stop);
    v.addEventListener("ended", stop);
    v.addEventListener("seeked", syncIfIdle);
    v.addEventListener("timeupdate", syncIfIdle);
    return () => {
      cancelAnimationFrame(raf);
      v.removeEventListener("play", start);
      v.removeEventListener("playing", start);
      v.removeEventListener("pause", stop);
      v.removeEventListener("ended", stop);
      v.removeEventListener("seeked", syncIfIdle);
      v.removeEventListener("timeupdate", syncIfIdle);
    };
  }, [videoRef, duration, loading]);

  /** `bf` is the prefix the visit backfill stamps; a producer visit id is a
   *  bare timestamp. It is the only thing that keeps the two kinds apart once
   *  their numbers are in the same columns. */
  const reconstructed = !!rows[0]?.visit_id?.startsWith("bf");

  /* Merging a visit onto one clip makes the lane list longer — a visit that
     was spread over thirteen files now shows all of its animals at once — so
     the row you are actually looking at has to be scrolled to rather than
     hunted for. `nearest` keeps the list still when it is already visible. */
  useEffect(() => {
    selectedLaneRef.current?.scrollIntoView({ block: "nearest" });
    }, [selectedId, rows.length]);

  const clamp = (s: number) => Math.min(duration, Math.max(0, s));
  const pct = (s: number) => `${duration > 0 ? (clamp(s) / duration) * 100 : 0}%`;

  /** `clip_segments` when the producer wrote them, else the single start→end
   *  span, else nothing — rows predating the offsets have no place on an axis. */
  const segmentsOf = (row: Detection): [number, number][] => {
    if (row.clip_segments?.length) return row.clip_segments;
    const from = row.clip_offset_seconds;
    if (from === null || from === undefined) return [];
    return [[from, row.clip_end_offset_seconds ?? from]];
  };

  const seekFromPointer = (clientX: number) => {
    const strip = stripRef.current;
    if (!strip) return;
    const rect = strip.getBoundingClientRect();
    if (rect.width <= 0) return;
    onSeek(clamp(((clientX - rect.left) / rect.width) * duration), true);
  };

  const ticks: number[] = [];
  for (let t = 0; t <= duration + 1e-6; t += tickStep(duration)) ticks.push(Number(t.toFixed(3)));

  const positionedRows = rows.filter((row) => segmentsOf(row).length > 0);
  const unknownRows = rows.filter((row) => segmentsOf(row).length === 0);

  // One template shared by the ruler, the lanes and the playhead overlay, so a
  // tick and a block that read as the same instant really are at the same x.
  const COLS = "grid grid-cols-[3rem_1fr_3.5rem] gap-2";

  return (
    <div
      ref={deckRef}
      className="min-w-0 shrink-0 overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900 text-neutral-200"
    >
      {/* Transport. The deck owns playback because it already owns the
          scrubber — the tracks below ARE the scrub surface, so a second
          progress bar would be the same information drawn twice. */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-neutral-800 px-3 py-2">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => {
              const v = videoRef.current;
              // `isConnected` as well as null: a ref left pointing at a node
              // React has already thrown away answers every call silently, and
              // the button then does nothing at all.
              if (!v || !v.isConnected) return onLoadClip(true);
              if (v.paused) {
                void v.play().catch(() => {});
              } else {
                v.pause();
              }
            }}
            disabled={loading}
            title={loading ? "載入影片中…" : paused ? "播放（空白鍵）" : "暫停（空白鍵）"}
            className="flex h-7 w-7 items-center justify-center rounded-full bg-neutral-100 text-neutral-900 transition hover:bg-white disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : paused ? (
              <Play className="h-3.5 w-3.5 translate-x-px" />
            ) : (
              <Pause className="h-3.5 w-3.5" />
            )}
          </button>
          {([[-1, SkipBack, "上一格（,）"], [1, SkipForward, "下一格（.）"]] as const).map(
            ([dir, Icon, label]) => (
              <button
                key={dir}
                type="button"
                onClick={() => {
                  const v = videoRef.current;
                  if (!v) return;
                  v.pause();
                  v.currentTime = Math.max(
                    0, Math.min(v.duration || 0, v.currentTime + dir * FRAME));
                }}
                title={label}
                className="flex h-7 w-7 items-center justify-center rounded text-neutral-400 transition hover:bg-white/10 hover:text-neutral-100"
              >
                <Icon className="h-3.5 w-3.5" />
              </button>
            ),
          )}
        </div>

        <span className="font-mono text-[11px] tabular-nums text-neutral-400">
          <span className="text-neutral-100">{playhead.toFixed(1)}</span>
          {" / "}
          {duration.toFixed(1)} 秒
        </span>

        <button
          type="button"
          onClick={() => {
            const next = rate === 1 ? 0.5 : rate === 0.5 ? 2 : 1;
            setRate(next);
            playbackRateRef.current = next;
            if (videoRef.current) videoRef.current.playbackRate = next;
          }}
          title="播放速度"
          className="rounded border border-neutral-700 px-1.5 py-px font-mono text-[10px] tabular-nums text-neutral-300 transition hover:border-neutral-500 hover:text-neutral-100"
        >
          {rate}×
        </button>

        <span className="ml-auto flex items-center gap-2 text-[11px] font-medium tracking-wide text-neutral-400">
          <span>
            這支影片的 <span className="font-mono text-neutral-100">{rows.length}</span> 條軌跡
          </span>
          {/* Recorded and reconstructed land in the same columns and read the
              same, and this panel is used to judge whether two rows are the
              same animal. A reconstruction cannot answer that: its seconds are
              interpolated across an archive with no timestamps, and its track
              is matched by box overlap because re-tracking mints ids the
              producer never had. Say which one you are looking at. */}
          {reconstructed && (
            <span
              className="rounded-sm border border-amber-500/40 px-1 py-px text-[10px] font-normal text-amber-400/90"
              title="這條時間軸是從封存影格重建的：秒數是內插的，軌跡是用框重疊配對的，不是當時記錄下來的"
            >
              重建
            </span>
          )}
        </span>
        <button
          type="button"
          onClick={() => {
            // The whole instrument, not just the picture: in fullscreen the
            // tracks are what you are reading the video against.
            const deck = deckRef.current?.parentElement;
            if (document.fullscreenElement) void document.exitFullscreen();
            else void deck?.requestFullscreen?.().catch(() => {});
          }}
          title="全螢幕"
          className="flex h-7 w-7 items-center justify-center rounded text-neutral-400 transition hover:bg-white/10 hover:text-neutral-100"
        >
          <Maximize2 className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Until the element reports its length there is no axis to draw on, and
          a ruler with no scale is worse than none. The transport above still
          works — that is why it does not live down here. */}
      {duration <= 0 ? (
        <div className="px-3 py-3 text-[11px] text-neutral-500">讀取影片長度中…</div>
      ) : (
      <div className="max-h-[min(30dvh,16rem)] overflow-y-auto" style={{ scrollbarGutter: "stable" }}>
      <div className="relative min-w-0 overflow-x-hidden px-3 py-2">
        {/* Ruler */}
        <div className={`${COLS} mb-1.5 items-end`}>
          <span />
          <div className="relative h-4">
            {ticks.map((t, i) => {
              const last = i === ticks.length - 1;
              return (
                <span
                  key={t}
                  className="absolute bottom-0 flex flex-col items-center"
                  style={{
                    left: pct(t),
                    transform: i === 0 ? "none" : last ? "translateX(-100%)" : "translateX(-50%)",
                  }}
                >
                  <span className="font-mono text-[9px] leading-none tabular-nums text-neutral-500">
                    {t}s
                  </span>
                  <span className="mt-0.5 h-1.5 w-px bg-neutral-700" />
                </span>
              );
            })}
          </div>
          <span />
        </div>

        {/* Lanes */}
        <div
          className="flex min-w-0 flex-col gap-1"
        >
          {rowsLoading ? (
            <div className="px-1 py-2 text-[11px] text-neutral-500">載入這支影片的軌跡中…</div>
          ) : positionedRows.map((row) => {
            const segs = segmentsOf(row);
            const from = segs.length ? segs[0][0] : null;
            const tone = trackToneIndex(trackColourKey(row));
            const isSelected = row.id === selectedId;
            // On screen at this instant. The tolerance covers the gap between
            // the frame a segment ends on and the next frame's timestamp.
            const onScreen = segs.some(([a, b]) => playhead >= a - 0.05 && playhead <= b + 0.05);
            return (
              <div
                key={row.id}
                ref={isSelected ? selectedLaneRef : undefined}
                onPointerEnter={() => onHoverRow(row)}
                onPointerLeave={() => onHoverRow(null)}
                className={`${COLS} shrink-0 items-center rounded ${
                  isSelected ? "bg-white/[0.06]" : "hover:bg-white/[0.04]"
                }`}
              >
                <button
                  type="button"
                  disabled={from === null}
                  // A lane label means "show this animal's first frame".
                  // Do not apply the normal context pad: it can land on the
                  // previous lane and makes the timeline appear to jump.
                  onClick={() => from !== null && onSeek(from, true)}
                  title={
                    from === null
                      ? "這筆紀錄早於「第幾秒」開始被記錄"
                      : `#${row.track_id}：第 ${from.toFixed(1)}–${segs[segs.length - 1][1].toFixed(1)} 秒` +
                        (segs.length > 1 ? `，中間離開畫面 ${segs.length - 1} 次` : "")
                  }
                  className={`rounded px-1 py-0.5 text-left font-mono text-[10px] tabular-nums transition-colors ${
                    from === null ? "cursor-default text-neutral-600" : "hover:bg-white/10"
                  } ${isSelected ? "ring-1 ring-white/40" : ""}`}
                  style={
                    from === null
                      ? undefined
                      : {
                          color: trackOnDark(tone),
                          opacity: onScreen ? 1 : 0.62,
                          backgroundColor: onScreen ? `${trackFill(tone)}2e` : undefined,
                        }
                  }
                >
                  #{row.track_id}
                </button>

                <div className="relative h-3 overflow-hidden rounded-sm bg-white/5">
                  {/* One block per stretch it was visible. A macaque that walks
                      out and comes back gets two, and the gap between them is
                      the point — a single bar would paint it as never having
                      left, which is exactly the judgement Re-ID needs. */}
                  {segs.map(([a, b], i) => (
                    <span
                      key={i}
                      className="absolute inset-y-0 rounded-sm transition-opacity"
                      style={{
                        left: pct(a),
                        width: `${Math.max(0.6, ((clamp(b) - clamp(a)) / duration) * 100)}%`,
                        backgroundColor: trackFill(tone),
                        opacity: onScreen ? 1 : isSelected ? 0.85 : 0.45,
                      }}
                    />
                  ))}
                </div>

                <span className="text-right font-mono text-[10px] tabular-nums text-neutral-500">
                  {from === null ? "—" : `${from.toFixed(1)}s`}
                </span>
              </div>
            );
          })}
          {!rowsLoading && unknownRows.length > 0 && (
            <div
              className="mt-1 rounded border border-dashed border-neutral-700 bg-neutral-950/40 px-2 py-1.5 text-[10px] leading-relaxed text-neutral-500"
              title="這些紀錄有影片，但目前無法從封存影格確定它們在影片中的位置"
            >
              另有 {unknownRows.length} 筆位置不明
              <span className="ml-1 text-neutral-600">（有影片，但無法確定第幾秒）</span>
            </div>
          )}
        </div>

        {/* Playhead + scrub surface. It spans the ruler and every visible lane,
            and it owns the pointer for the track column only — the lane labels
            sit in a different column and stay clickable. */}
        <div className={`pointer-events-none absolute inset-x-3 inset-y-2 ${COLS}`}>
          <span />
          <div
            ref={stripRef}
            className="pointer-events-auto relative cursor-col-resize touch-pan-y"
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              setScrubbing(true);
              seekFromPointer(e.clientX);
            }}
            onPointerMove={(e) => scrubbing && seekFromPointer(e.clientX)}
            onPointerUp={(e) => {
              e.currentTarget.releasePointerCapture(e.pointerId);
              setScrubbing(false);
            }}
            onPointerCancel={() => setScrubbing(false)}
          >
            <span ref={lineRef} className="absolute inset-y-0 w-px bg-white/70" style={{ left: 0 }} />
            <span
              ref={bubbleRef}
              className="absolute top-0 -translate-y-px rounded-sm bg-white px-1 font-mono text-[9px] leading-[1.4] tabular-nums text-neutral-900"
              style={{ left: 0 }}
            >
              0.0
            </span>
          </div>
          <span />
        </div>
      </div>
      </div>
      )}
    </div>
  );
}


/**
 * Outlines one animal on the frame while its lane is hovered.
 *
 * Canvas rather than a positioned element: it paints on hover and clears on
 * leave, so between those two moments the browser has nothing to lay out,
 * nothing to composite and no React render to run. The whole cost is one
 * `drawImage`-free stroke of a rectangle.
 *
 * The box is the one RECORDED for that detection — we do not store per-frame
 * boxes, so this says "which of the animals on screen is #620", not "where is
 * it at this instant". When the playhead is outside that animal's stretches it
 * is drawn dimmed and dashed, because then it really is a position from
 * another moment and a solid box would be a claim we cannot make.
 */
function HoverOutline({
  row,
  videoRef,
}: {
  row: Detection | null;
  videoRef: RefObject<HTMLVideoElement | null>;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const cv = canvasRef.current;
    const v = videoRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    const clear = () => ctx.clearRect(0, 0, cv.width, cv.height);

    const paint = () => {
      const box = v?.getBoundingClientRect();
      if (!row || !v || !box || !v.videoWidth) return clear();
      // Match the backing store to the displayed size once per paint; the
      // player is inside a flex column that can resize under us.
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (cv.width !== Math.round(box.width * dpr)) {
        cv.width = Math.round(box.width * dpr);
        cv.height = Math.round(box.height * dpr);
      }
      clear();

      // `object-contain` letterboxes the frame inside the element, so the
      // scale is the smaller axis and the rest is padding.
      const scale = Math.min(box.width / v.videoWidth, box.height / v.videoHeight);
      const padX = (box.width - v.videoWidth * scale) / 2;
      const padY = (box.height - v.videoHeight * scale) / 2;
      const segs = row.clip_segments?.length
        ? row.clip_segments
        : row.clip_offset_seconds !== null && row.clip_offset_seconds !== undefined
          ? [[row.clip_offset_seconds, row.clip_end_offset_seconds ?? row.clip_offset_seconds]]
          : [];
      const t = v.currentTime;
      const present = segs.some(([a, b]) => t >= a - 0.05 && t <= b + 0.05);
      if (!present) return clear();

      // Review rows may carry the complete per-frame track. Use the box at the
      // playhead instead of the peak box from the row; otherwise hover appears
      // to "lose" the animal as soon as it walks away from its first frame.
      let sourceBox: [number, number, number, number] = [
        row.bbox_x,
        row.bbox_y,
        row.bbox_x + row.bbox_w,
        row.bbox_y + row.bbox_h,
      ];
      if (row.track_boxes?.length && row.track_box_fps && row.track_box_fps > 0) {
        const targetFrame = t * row.track_box_fps;
        const closest = row.track_boxes.reduce((best, candidate) =>
          Math.abs(candidate[0] - targetFrame) < Math.abs(best[0] - targetFrame)
            ? candidate
            : best,
        );
        sourceBox = [closest[1], closest[2], closest[3], closest[4]];
      }
      const x = (padX + sourceBox[0] * scale) * dpr;
      const y = (padY + sourceBox[1] * scale) * dpr;
      const w = (sourceBox[2] - sourceBox[0]) * scale * dpr;
      const h = (sourceBox[3] - sourceBox[1]) * scale * dpr;
      const ink = trackOnDark(trackToneIndex(trackColourKey(row)));

      ctx.save();
      ctx.strokeStyle = ink;
      ctx.lineWidth = 3 * dpr;
      ctx.globalAlpha = 1;
      ctx.lineJoin = "round";
      // A dark halo preserves the same animal colour on bright foliage and
      // on the pale review UI; the thinner colour stroke remains the identity
      // cue. This reads better than a single neon line over mixed scenery.
      ctx.strokeStyle = "rgba(0, 0, 0, 0.82)";
      ctx.lineWidth = 7 * dpr;
      ctx.strokeRect(x, y, w, h);
      ctx.strokeStyle = ink;
      ctx.lineWidth = 2.5 * dpr;
      ctx.strokeRect(x, y, w, h);

      const label = `#${row.track_id}${present ? "" : "（此刻不在畫面）"}`;
      ctx.setLineDash([]);
      ctx.font = `${12 * dpr}px ui-monospace, monospace`;
      const tw = ctx.measureText(label).width + 10 * dpr;
      // Keep the label inside the frame. An animal at the right-hand edge is
      // where this matters, and a label
      // that runs off the canvas is unreadable for precisely the row someone
      // is hovering to identify.
      const lx = Math.max(0, Math.min(x, cv.width - tw));
      const ly = y > 20 * dpr ? y - 20 * dpr : Math.min(cv.height - 18 * dpr, y + h + 2 * dpr);
      ctx.fillStyle = ink;
      ctx.fillRect(lx, ly, tw, 18 * dpr);
      ctx.fillStyle = "#0a0a0a";
      ctx.fillText(label, lx + 5 * dpr, ly + 13 * dpr);
      ctx.restore();
    };

    if (!row) {
      clear();
      return;
    }
    // Follow the playhead so the dimming is right while the clip plays; this
    // loop exists ONLY while a lane is hovered.
    const loop = () => {
      paint();
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      clear();
    };
  }, [row, videoRef]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 z-10 h-full w-full"
    />
  );
}
