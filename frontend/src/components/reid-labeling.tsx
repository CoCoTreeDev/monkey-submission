"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import useSWR from "swr";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Clock,
  Eye,
  Loader2,
  RefreshCw,
  X,
} from "lucide-react";

import {
  API_BASE,
  fetcher,
  labelReIDPair,
  snapshotUrl,
  type ReIDLabel,
  type ReIDPair,
  type ReIDPairList,
  type ReIDTrack,
  type ReIDTrackBox,
} from "@/lib/api";
import { SnapshotImg } from "@/components/snapshot-img";

type QueueStatus = "pending" | "labeled" | "auto" | "all";
const REID_REVIEW_BATCH_SIZE = 400;

const queueFilters: { value: QueueStatus; label: string }[] = [
  { value: "pending", label: "待人工" },
  { value: "labeled", label: "人工已標記" },
  { value: "auto", label: "系統已排除" },
  { value: "all", label: "本輪全部" },
];

const verdicts: {
  value: ReIDLabel;
  label: string;
  shortcut: string;
  description: string;
  className: string;
  icon: typeof Check;
}[] = [
  {
    value: "same",
    label: "同一隻",
    shortcut: "1",
    description: "外觀、身形或連續動作一致",
    className: "border-emerald-200 bg-emerald-50 text-emerald-800 hover:bg-emerald-100",
    icon: Check,
  },
  {
    value: "different",
    label: "不同隻",
    shortcut: "2",
    description: "同時出現兩隻，或外觀明顯不同",
    className: "border-rose-200 bg-rose-50 text-rose-800 hover:bg-rose-100",
    icon: X,
  },
  {
    value: "uncertain",
    label: "無法判斷",
    shortcut: "3",
    description: "遮擋、太小或資料不足，不要猜",
    className: "border-amber-200 bg-amber-50 text-amber-800 hover:bg-amber-100",
    icon: AlertTriangle,
  },
];

function formatSeconds(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "未記錄";
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

function formatTimestamp(timestamp: string): string {
  const value = /(?:Z|[+-]\d{2}:?\d{2})$/.test(timestamp)
    ? timestamp
    : `${timestamp}+08:00`;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return timestamp;
  return new Intl.DateTimeFormat("zh-TW", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function clipName(path: string): string {
  return path.split("/").pop() || path;
}

type TrackView = "annotated" | "raw" | "crop";

const trackViewLabels: Record<TrackView, string> = {
  annotated: "標註畫面",
  raw: "原始畫面",
  crop: "裁切圖",
};

function imageSources(track: ReIDTrack, view: TrackView): {
  filename: string;
  fallbackFilename?: string;
} {
  const sources: Record<TrackView, [string, string | undefined]> = {
    annotated: [track.annotated_path, track.image_path || track.crop_path || undefined],
    raw: [track.image_path, track.annotated_path || track.crop_path || undefined],
    crop: [track.crop_path, track.annotated_path || track.image_path || undefined],
  };
  const [filename, fallbackFilename] = sources[view];
  return { filename, fallbackFilename };
}

function TrackCard({ track, side }: { track: ReIDTrack; side: "A" | "B" }) {
  const [view, setView] = useState<TrackView>("annotated");
  const hasCrop = Boolean(track.crop_path);
  const hasRaw = Boolean(track.image_path);
  const hasAnnotated = Boolean(track.annotated_path);
  const displayView = view === "annotated" && !hasAnnotated
    ? hasRaw ? "raw" : "crop"
    : view;
  const sources = imageSources(track, displayView);

  return (
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-900 text-[11px] font-bold text-white">
            {side}
          </span>
          <div>
            <p className="text-sm font-semibold text-slate-900">Track #{track.track_id}</p>
            <p className="text-[11px] text-slate-500">偵測 #{track.id}</p>
          </div>
        </div>
        <div className="flex flex-wrap justify-end gap-1 rounded-lg bg-slate-50 p-1">
          {(
            [
              ["annotated", "標註畫面", hasAnnotated],
              ["raw", "原始畫面", hasRaw],
              ["crop", "裁切圖", hasCrop],
            ] as const
          ).map(([option, label, enabled]) => (
            <button
              type="button"
              key={option}
              onClick={() => setView(option)}
              disabled={!enabled}
              aria-pressed={view === option}
              className={`inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-35 ${
                view === option
                  ? "bg-white text-sky-700 shadow-sm"
                  : "text-slate-500 hover:text-slate-900"
              }`}
              title={enabled ? `查看${label}` : `此紀錄沒有${label}`}
            >
              {option === "annotated" && <Eye className="h-3.5 w-3.5" />}
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative aspect-[4/3] bg-slate-950">
        {sources.filename ? (
          <SnapshotImg
            filename={sources.filename}
            fallbackFilename={sources.fallbackFilename}
            alt={`${side} Track ${track.track_id} ${trackViewLabels[displayView]}`}
            className="h-full w-full object-contain"
            loading="lazy"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-400">
            無圖片
          </div>
        )}
        <span className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-1 text-[10px] font-medium text-white">
          {trackViewLabels[displayView]}
          {displayView === "annotated" && "（完整畫面＋框）"}
        </span>
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 px-4 py-3 text-xs">
        <div>
          <dt className="text-slate-500">信心度</dt>
          <dd className="mt-0.5 font-mono font-semibold text-slate-900">
            {(track.confidence * 100).toFixed(1)}%
          </dd>
        </div>
        <div>
          <dt className="text-slate-500">逐幀框</dt>
          <dd className="mt-0.5 font-mono font-semibold text-slate-900">
            {track.frame_count > 0 ? `${track.frame_count} 幀` : "未記錄"}
          </dd>
        </div>
        <div className="col-span-2">
          <dt className="text-slate-500">偵測時間</dt>
          <dd className="mt-0.5 font-mono text-slate-900">{formatTimestamp(track.timestamp)}</dd>
        </div>
      </dl>
    </section>
  );
}

function timelineStyle(
  track: ReIDTrack,
  duration: number,
): { left: string; width: string } | null {
  if (
    duration <= 0 ||
    track.start_seconds === null ||
    track.end_seconds === null
  ) {
    return null;
  }
  const start = Math.max(0, Math.min(duration, track.start_seconds));
  const end = Math.max(start, Math.min(duration, track.end_seconds));
  const left = (start / duration) * 100;
  const width = Math.max(1.5, ((end - start) / duration) * 100);
  return { left: `${left}%`, width: `${Math.min(width, 100 - left)}%` };
}

const reviewTrackStyles = {
  A: {
    stroke: "#38bdf8",
    fill: "rgba(14, 165, 233, 0.18)",
    label: "#075985",
  },
  B: {
    stroke: "#fb923c",
    fill: "rgba(249, 115, 22, 0.18)",
    label: "#9a3412",
  },
} as const;

function boxAtTime(track: ReIDTrack, time: number): ReIDTrackBox | null {
  const boxes = track.track_boxes;
  const fps = track.track_box_fps;
  if (!boxes?.length || !fps || !Number.isFinite(fps) || fps <= 0) return null;

  const targetFrame = Math.max(0, Math.round(time * fps));
  let low = 0;
  let high = boxes.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (boxes[middle][0] < targetFrame) low = middle + 1;
    else if (boxes[middle][0] > targetFrame) high = middle - 1;
    else return boxes[middle];
  }
  const previous = high >= 0 ? boxes[high] : null;
  const next = low < boxes.length ? boxes[low] : null;
  const previousDistance = previous ? Math.abs(previous[0] - targetFrame) : Number.POSITIVE_INFINITY;
  const nextDistance = next ? Math.abs(next[0] - targetFrame) : Number.POSITIVE_INFINITY;
  const closest = previousDistance <= nextDistance ? previous : next;
  const closestDistance = Math.min(previousDistance, nextDistance);
  // A small tolerance covers an encoded frame between two persisted detector
  // frames. Never paint a stale box after the track has disappeared.
  return closest && closestDistance <= Math.max(1, Math.ceil(fps * 0.08))
    ? closest
    : null;
}

function TrackOverlay({
  pair,
  videoRef,
}: {
  pair: ReIDPair;
  videoRef: { current: HTMLVideoElement | null };
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    let animationFrame = 0;
    const draw = () => {
      const width = video.clientWidth;
      const height = video.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      const pixelWidth = Math.max(1, Math.round(width * dpr));
      const pixelHeight = Math.max(1, Math.round(height * dpr));
      // Changing canvas.width/height clears and reallocates the backing store.
      // Only do that on a real resize; doing it on every video frame causes
      // unnecessary work exactly where the reviewer needs smooth playback.
      if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
      if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);

      if (!video.videoWidth || !video.videoHeight || !width || !height) return;
      const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
      const contentWidth = video.videoWidth * scale;
      const contentHeight = video.videoHeight * scale;
      const left = (width - contentWidth) / 2;
      const top = (height - contentHeight) / 2;
      const tracks: { side: "A" | "B"; track: ReIDTrack }[] = [
        { side: "A", track: pair.a },
        { side: "B", track: pair.b },
      ];

      for (const { side, track } of tracks) {
        const box = boxAtTime(track, video.currentTime);
        if (!box) continue;
        const style = reviewTrackStyles[side];
        const x = left + box[1] * scale;
        const y = top + box[2] * scale;
        const boxWidth = (box[3] - box[1]) * scale;
        const boxHeight = (box[4] - box[2]) * scale;
        const lineWidth = Math.max(2, Math.min(5, width / 360));
        context.fillStyle = style.fill;
        context.strokeStyle = style.stroke;
        context.lineWidth = lineWidth;
        context.fillRect(x, y, boxWidth, boxHeight);
        context.strokeRect(x, y, boxWidth, boxHeight);

        const label = `${side} · Track #${track.track_id}`;
        context.font = "600 13px ui-monospace, SFMono-Regular, Menlo, monospace";
        const labelHeight = 22;
        const labelWidth = context.measureText(label).width + 14;
        const labelX = Math.max(0, Math.min(width - labelWidth, x));
        const labelY = y >= labelHeight ? y - labelHeight : y;
        context.fillStyle = style.stroke;
        context.fillRect(labelX, labelY, labelWidth, labelHeight);
        context.fillStyle = style.label;
        context.fillText(label, labelX + 7, labelY + 15);
      }
    };

    const animate = () => {
      draw();
      if (!video.paused && !video.ended) animationFrame = requestAnimationFrame(animate);
    };
    const drawWhilePlaying = () => {
      cancelAnimationFrame(animationFrame);
      animate();
    };
    const stopAnimating = () => {
      cancelAnimationFrame(animationFrame);
      draw();
    };

    video.addEventListener("timeupdate", draw);
    video.addEventListener("loadedmetadata", draw);
    video.addEventListener("play", drawWhilePlaying);
    video.addEventListener("pause", stopAnimating);
    video.addEventListener("ended", stopAnimating);
    const resizeObserver = new ResizeObserver(draw);
    resizeObserver.observe(video);
    draw();

    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      video.removeEventListener("timeupdate", draw);
      video.removeEventListener("loadedmetadata", draw);
      video.removeEventListener("play", drawWhilePlaying);
      video.removeEventListener("pause", stopAnimating);
      video.removeEventListener("ended", stopAnimating);
    };
  }, [pair, videoRef]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 z-10 h-full w-full"
      aria-hidden="true"
    />
  );
}

function TimelineEvidence({
  pair,
  currentTime,
  onSeek,
}: {
  pair: ReIDPair;
  currentTime: number;
  onSeek: (seconds: number) => void;
}) {
  const duration = pair.clip_duration_seconds ?? 0;
  const barA = timelineStyle(pair.a, duration);
  const barB = timelineStyle(pair.b, duration);
  const relation = pair.temporal_relation === "overlap"
    ? "時間重疊"
    : pair.temporal_relation === "separate"
      ? `時間分開${pair.gap_seconds !== null ? `，間隔 ${formatSeconds(pair.gap_seconds)}` : ""}`
      : "時間軸資料不足";

  return (
    <section className="border-t border-slate-100 bg-slate-950 px-4 py-3 text-white sm:px-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Clock className="h-4 w-4 text-sky-300" />
          A／B 出現時間
        </div>
        <span className="rounded-full bg-white/10 px-2.5 py-1 text-[11px] text-slate-300">
          {relation}
        </span>
      </div>
      <div className="mt-2 overflow-hidden rounded-lg border border-white/10 bg-slate-900">
        <div className="flex items-center justify-between border-b border-white/10 px-3 py-2 text-[10px] text-slate-400">
          <span>0:00</span>
          <span>{duration > 0 ? formatSeconds(duration) : "片段長度未記錄"}</span>
        </div>
        <div
          className="relative space-y-2 px-3 py-3"
          role="slider"
          tabIndex={0}
          aria-label="影片時間軸，按左右鍵以每秒移動"
          aria-valuemin={0}
          aria-valuemax={duration}
          aria-valuenow={Math.min(duration, Math.max(0, currentTime))}
          onClick={(event) => {
            const bounds = event.currentTarget.getBoundingClientRect();
            if (bounds.width > 0 && duration > 0) {
              onSeek(((event.clientX - bounds.left) / bounds.width) * duration);
            }
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
              event.preventDefault();
              onSeek(currentTime + (event.key === "ArrowRight" ? 1 : -1));
            }
          }}
        >
          {duration > 0 && (
            <div
              className="pointer-events-none absolute bottom-2 top-2 z-10 w-px bg-white shadow-[0_0_0_1px_rgba(15,23,42,0.35)]"
              style={{ left: `${Math.min(100, Math.max(0, currentTime / duration * 100))}%` }}
            />
          )}
          {(["A", "B"] as const).map((side) => {
            const track = side === "A" ? pair.a : pair.b;
            const bar = side === "A" ? barA : barB;
            return (
              <div key={side} className="flex items-center gap-2">
                <span className="w-4 shrink-0 text-center text-[10px] font-bold text-slate-400">{side}</span>
                <div className="relative h-4 min-w-0 flex-1 rounded bg-white/5">
                  {bar ? (
                    <div
                      className={`absolute top-0.5 h-3 rounded ${side === "A" ? "bg-sky-400" : "bg-orange-400"}`}
                      style={bar}
                      title={`${side} ${formatSeconds(track.start_seconds)}–${formatSeconds(track.end_seconds)}`}
                    />
                  ) : (
                    <div className="flex h-full items-center px-2 text-[10px] text-slate-500">
                      未記錄此軌跡的起訖秒數
                    </div>
                  )}
                </div>
                <span className="w-[92px] shrink-0 text-right font-mono text-[10px] text-slate-400">
                  {formatSeconds(track.start_seconds)}–{formatSeconds(track.end_seconds)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
      <p className="mt-2 text-[11px] text-slate-400">
        點擊時間軸跳轉 · 目前 {formatSeconds(currentTime)} · 藍 A／橘 B
      </p>
    </section>
  );
}

function ReviewEvidence({ pair }: { pair: ReIDPair }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [loadError, setLoadError] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState(0);
  const duration = mediaDuration || pair.clip_duration_seconds || 0;
  const hasFrameOverlay = Boolean(pair.a.track_boxes?.length || pair.b.track_boxes?.length);

  const seek = useCallback((seconds: number) => {
    const video = videoRef.current;
    const nextTime = Math.min(duration || Number.POSITIVE_INFINITY, Math.max(0, seconds));
    if (video) video.currentTime = nextTime;
    setCurrentTime(nextTime);
  }, [duration]);

  return (
    <>
      <section className="border-t border-slate-100 bg-slate-50 px-4 py-3 sm:px-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm font-semibold text-slate-900">
            影片證據 <span className="ml-2 text-xs font-normal text-slate-500">A 藍框 · B 橘框</span>
          </p>
          <span className="text-[11px] text-slate-500">播放影片後直接判斷</span>
        </div>
        <div className="relative mt-2 aspect-video max-h-[360px] overflow-hidden rounded-lg border border-slate-200 bg-slate-950">
          {loadError ? (
            <div className="px-3 py-8 text-center text-xs leading-5 text-rose-200">
              影片目前載入失敗，請改看下方的標註畫面；這不會影響標記資料。
            </div>
          ) : (
            <video
              ref={videoRef}
              src={pair.trail_path ? snapshotUrl(pair.trail_path) : undefined}
              controls
              playsInline
              preload="metadata"
              onError={() => setLoadError(true)}
              onLoadedMetadata={(event) => {
                setMediaDuration(event.currentTarget.duration || 0);
                setCurrentTime(0);
              }}
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
              className="relative z-0 block h-full w-full bg-slate-950 object-contain"
              aria-label={`${clipName(pair.trail_path)} ReID 審查影片`}
            />
          )}
          {!loadError && <TrackOverlay pair={pair} videoRef={videoRef} />}
        </div>
        {hasFrameOverlay ? (
          <p className="mt-1 text-[11px] text-slate-600">
            框線會跟著影片時間移動；框消失代表該時間點沒有有效觀測。
          </p>
        ) : (
          <p className="mt-1 text-[11px] text-amber-700">
            這組沒有逐幀框資料，只能查看影片原有的燒錄框線。
          </p>
        )}
      </section>
      <TimelineEvidence pair={{ ...pair, clip_duration_seconds: duration || pair.clip_duration_seconds }} currentTime={currentTime} onSeek={seek} />
    </>
  );
}

export function ReIDLabeling() {
  const [status, setStatus] = useState<QueueStatus>("pending");
  const [offset, setOffset] = useState(0);
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    setOffset(0);
  }, [status]);

  const queryOffset = status === "pending" ? 0 : offset;
  const key = `${API_BASE}/api/admin/reid/pairs?status=${status}&limit=1&offset=${queryOffset}&review_size=${REID_REVIEW_BATCH_SIZE}`;
  const {
    data,
    error,
    isLoading,
    isValidating,
    mutate,
  } = useSWR<ReIDPairList>(key, fetcher, {
    refreshInterval: status === "pending" ? 10000 : 30000,
    revalidateOnFocus: false,
  });
  const pair = data?.pairs[0] ?? null;

  useEffect(() => {
    setNote(pair?.note ?? "");
    setMessage(null);
  }, [pair?.pair_id, pair?.label, pair?.note]);

  const batchTotal = data?.batch_total ?? data?.total ?? 0;
  const batchPending = data?.batch_pending ?? data?.pending ?? 0;
  const batchLabeled = data?.batch_labeled ?? 0;
  const batchAutoResolved = data?.batch_auto_resolved ?? Math.max(0, batchTotal - batchPending - batchLabeled);
  const batchReviewPending = data?.batch_review_pending ?? batchPending;
  const reviewedCount = Math.max(0, batchTotal - batchPending);
  const progress = batchTotal > 0 ? Math.round((reviewedCount / batchTotal) * 100) : 0;

  const saveVerdict = useCallback(async (label: ReIDLabel) => {
    if (!pair || saving) return;
    setSaving(true);
    setMessage(null);
    try {
      await labelReIDPair(pair.a.id, pair.b.id, label, note);
      setMessage(`已儲存「${verdicts.find((item) => item.value === label)?.label ?? label}」`);
      await mutate();
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : "儲存標記失敗");
    } finally {
      setSaving(false);
    }
  }, [mutate, note, pair, saving]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)
      ) {
        return;
      }
      const verdict = verdicts.find((item) => item.shortcut === event.key);
      if (verdict) {
        event.preventDefault();
        void saveVerdict(verdict.value);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [saveVerdict]);

  const canGoBack = offset > 0;
  const canGoForward = data ? offset + 1 < data.total : false;
  const currentPosition = status === "pending" ? reviewedCount + 1 : offset + 1;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-sky-700">
            <span className="h-2 w-2 rounded-full bg-sky-500" />
            Human-in-the-loop
          </div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900">ReID 人工審查</h2>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-600">
            這一頁只問一件事：畫面中的 A、B 兩條軌跡，是不是同一隻猴子？
          </p>
        </div>
        <button
          type="button"
          onClick={() => void mutate()}
          disabled={isValidating || saving}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isValidating ? "animate-spin" : ""}`} />
          更新佇列
        </button>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-1">
          {queueFilters.map((filter) => (
            <button
              type="button"
              key={filter.value}
              onClick={() => setStatus(filter.value)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                status === filter.value
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-slate-900"
              }`}
            >
              {filter.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500">
          {status === "pending" ? "已按優先順序排列；儲存後自動前往下一組" : "可用左右按鈕瀏覽本輪 pair"}
        </p>
      </div>

      <section className="rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-sm sm:px-5">
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <span className="font-semibold text-slate-900">本輪進度</span>
          <span className="font-mono text-xs text-slate-600">
            {data ? `${reviewedCount} / ${batchTotal} 組（上限 ${REID_REVIEW_BATCH_SIZE} 組）` : "載入中…"}
          </span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100" aria-label={`本輪完成 ${progress}%`}>
          <div className="h-full rounded-full bg-emerald-500 transition-[width]" style={{ width: `${progress}%` }} />
        </div>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
          <span>待人工 {data?.batch_pending ?? "—"}</span>
          <span>人工已標記 {data?.batch_labeled ?? "—"}</span>
          <span>系統已排除 {batchAutoResolved}</span>
          <span>全部候選 {data?.candidate_total ?? "—"}</span>
        </div>
      </section>

      {data && (
        <section className="rounded-xl border border-sky-100 bg-sky-50 px-4 py-3 text-xs leading-5 text-sky-900 sm:px-5">
          <p className="font-semibold">這一輪先由系統處理明顯不同隻</p>
          <p className="mt-1 text-sky-800">
            目前保留 {batchReviewPending} 組給人工；{batchAutoResolved} 組符合保守的逐幀幾何規則，僅標成系統預判，不會混入人工 seed。人工標記仍可覆寫系統預判。
          </p>
        </section>
      )}

      {error ? (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
          無法載入 ReID 標記佇列：{error instanceof Error ? error.message : "API error"}
        </div>
      ) : isLoading ? (
        <div className="flex min-h-[360px] items-center justify-center rounded-xl border border-slate-200 bg-white text-sm text-slate-500">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          載入候選 pair…
        </div>
      ) : !pair ? (
        <div className="flex min-h-[360px] flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white px-6 text-center">
          <Check className="h-8 w-8 text-emerald-500" />
          <p className="mt-3 text-sm font-semibold text-slate-900">
            {status === "pending" ? "目前沒有待標記 pair" : "目前篩選沒有資料"}
          </p>
          <p className="mt-1 max-w-md text-xs leading-5 text-slate-500">
            候選只會從同一支影片內的兩條軌跡產生；跨影片的身份關係不在這個標記頁猜測。若你只是看不清楚，請選「無法判斷」，不需要硬選答案。
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 px-4 py-4 sm:px-6">
            <div className="min-w-0">
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">Candidate pair</p>
              <h3 className="mt-1 truncate font-mono text-sm font-semibold text-slate-900" title={pair.trail_path}>
                {clipName(pair.trail_path)}
              </h3>
              <p className="mt-1 text-xs text-slate-500">
                {pair.camera_id} · pair {pair.pair_id} · 這兩條軌跡來自同一支影片
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              {pair.label && (
                <span className="rounded-full bg-slate-100 px-2.5 py-1 font-medium text-slate-700">
                  人工標記：{verdicts.find((item) => item.value === pair.label)?.label ?? pair.label}
                </span>
              )}
              {!pair.label && pair.system_label && (
                <span className="rounded-full bg-sky-100 px-2.5 py-1 font-medium text-sky-800">
                  系統預判：不同隻
                </span>
              )}
              {!pair.label && pair.review_required && (
                <span className="rounded-full bg-amber-100 px-2.5 py-1 font-medium text-amber-800">
                  需要人工確認 · {pair.review_priority} 分
                </span>
              )}
              {status !== "pending" && (
                <span className="font-mono">{offset + 1} / {data?.total ?? 0}</span>
              )}
              {status === "pending" && data && (
                <span className="font-mono">本輪第 {currentPosition} / {batchTotal} 組</span>
              )}
            </div>
          </div>

          <ReviewEvidence key={pair.pair_id} pair={pair} />

          {!pair.label && (
            <div className={`border-t px-4 py-3 text-xs leading-5 sm:px-6 ${pair.system_label ? "border-sky-100 bg-sky-50 text-sky-900" : "border-amber-100 bg-amber-50 text-amber-900"}`}>
              <p className="font-semibold">{pair.system_label ? "系統已先排除這組" : "系統把這組排在人工前面"}</p>
              <p className="mt-0.5">{pair.review_reason}</p>
              {pair.system_label && <p className="mt-0.5 text-sky-800">若你不同意，可直接選下方答案；儲存後會留下人工標記。</p>}
            </div>
          )}

          <div className="border-t border-slate-100 px-4 py-4 sm:px-6">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-slate-900">{pair.system_label ? "要覆寫系統預判嗎？" : "這兩條軌跡是同一隻猴子嗎？"}</p>
                <p className="mt-1 text-xs text-slate-500">可直接按鍵盤 1、2、3 快速標記；不確定就選 3</p>
              </div>
              {message && (
                <span className="text-xs font-medium text-slate-600" role="status">{message}</span>
              )}
            </div>
            <div className="grid gap-2 sm:grid-cols-3">
              {verdicts.map((verdict) => {
                const Icon = verdict.icon;
                return (
                  <button
                    type="button"
                    key={verdict.value}
                    onClick={() => void saveVerdict(verdict.value)}
                    disabled={saving}
                    className={`flex min-h-[66px] items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${verdict.className}`}
                  >
                    {saving ? <Loader2 className="h-4 w-4 shrink-0 animate-spin" /> : <Icon className="h-4 w-4 shrink-0" />}
                    <span className="min-w-0">
                      <span className="flex items-center gap-2 text-sm font-semibold">
                        {verdict.label}
                        <kbd className="rounded border border-current/20 px-1.5 py-0.5 font-mono text-[10px] opacity-70">{verdict.shortcut}</kbd>
                      </span>
                      <span className="mt-0.5 block text-[11px] opacity-75">{verdict.description}</span>
                    </span>
                  </button>
                );
              })}
            </div>
            <label className="mt-4 block">
              <span className="text-xs font-medium text-slate-700">備註（選填）</span>
              <textarea
                value={note}
                onChange={(event) => setNote(event.target.value)}
                maxLength={1000}
                rows={2}
                placeholder="例如：遮擋太多、只有一幀、框的位置不一致…"
                className="mt-1.5 block w-full resize-y rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 outline-none transition-colors placeholder:text-slate-400 focus:border-sky-400 focus:ring-2 focus:ring-sky-100"
              />
            </label>
          </div>

          <div className="grid gap-4 border-t border-slate-100 p-4 sm:p-6 lg:grid-cols-2">
            <TrackCard track={pair.a} side="A" />
            <TrackCard track={pair.b} side="B" />
          </div>

          {status !== "pending" && (
            <div className="flex items-center justify-between border-t border-slate-100 px-4 py-3 sm:px-6">
              <button
                type="button"
                onClick={() => setOffset((current) => Math.max(0, current - 1))}
                disabled={!canGoBack || saving}
                className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
                上一組
              </button>
              <button
                type="button"
                onClick={() => setOffset((current) => current + 1)}
                disabled={!canGoForward || saving}
                className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
              >
                下一組
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
