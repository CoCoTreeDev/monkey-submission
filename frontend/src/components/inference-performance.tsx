"use client";

import { useMemo } from "react";
import useSWR from "swr";
import { Gauge, Loader2 } from "lucide-react";
import {
  API_BASE,
  fetcher,
  cameraLabel,
  type Camera,
  type InferenceStats,
} from "@/lib/api";

/** "62 小時" beats "225283 秒" — a stall long enough to matter is unreadable in seconds. */
function formatSince(seconds: number | null | undefined): string {
  if (seconds == null) return "";
  if (seconds < 90) return `${seconds.toFixed(0)} 秒`;
  if (seconds < 5400) return `${(seconds / 60).toFixed(0)} 分鐘`;
  if (seconds < 172800) return `${(seconds / 3600).toFixed(0)} 小時`;
  return `${(seconds / 86400).toFixed(1)} 天`;
}

/** 推論效能 — per-camera YOLO throughput, grouped active/offline (Operations tab, admin). */
export function InferencePerformance() {
  const { data: inference, isLoading: inferenceLoading } = useSWR<InferenceStats>(
    `${API_BASE}/api/admin/inference-stats`,
    fetcher,
    { refreshInterval: 2000 }
  );
  const { data: cameras } = useSWR<Camera[]>(`${API_BASE}/api/cameras`, fetcher, {
    refreshInterval: 10000,
  });

  const groups = useMemo(() => {
    const rows = inference?.cameras ?? [];
    const labelById = new Map((cameras ?? []).map((c) => [c.camera_id, cameraLabel(c)]));
    return {
      active: rows.filter((r) => r.stream_connected),
      offline: rows.filter((r) => !r.stream_connected),
      label: (id: string) => labelById.get(id) ?? id,
    };
  }, [inference, cameras]);

  return (
    <div className="rounded-lg border bg-white">
      <div className="border-b px-5 py-3">
        <h3 className="text-sm font-semibold">推論效能</h3>
      </div>
      <div className="p-5">
        {inferenceLoading && !inference ? (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            載入中…
          </div>
        ) : !inference?.cameras.length ? (
          <p className="text-xs text-muted-foreground">目前沒有執行中的偵測 worker</p>
        ) : groups.active.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            沒有已連線的攝影機
            {groups.offline.length > 0 && (
              <> · 離線： {groups.offline.map((c) => groups.label(c.camera_id)).join("、")}</>
            )}
          </p>
        ) : (
          <div className="space-y-3">
            {groups.active.map((cam) => {
              const stale = cam.seconds_since_last != null && cam.seconds_since_last > 10;
              const cameraFrameSkip = cam.frame_skip ?? inference?.frame_skip ?? 1;
              const targetFps = 25 / cameraFrameSkip;
              // A gated camera is SUPPOSED to sit far below target — it carries
              // no image and is on heartbeat sampling — so it must not be
              // painted amber as if it were failing to keep up.
              const gated = !!cam.idle_gated;
              // A DEAD detector thread is not a slow camera, and the difference
              // has to be unmissable: a failed thread once left this panel
              // rendering its final rolling-window FPS for 62 hours,
              // which reads as "a bit slow" rather than "not being watched".
              const dead = cam.worker_alive === false;
              const keepingUp = gated || cam.sample_fps >= targetFps * 0.8;
              const fps = cam.sample_fps > 0 ? cam.sample_fps : cam.throughput_fps;
              // Coverage: of the frames the camera delivered, how many were
              // inferred. Derived client-side too so an older node still shows it.
              const captureFps = cam.capture_fps ?? 0;
              const ceiling = cam.coverage_ceiling ?? inference?.coverage_ceiling ?? 1;
              const coverage =
                cam.coverage_ratio ??
                (captureFps > 0 ? Math.min(1, cam.sample_fps / captureFps) : 0);
              const coverageOk = coverage >= ceiling * 0.8;
              return (
                <div
                  key={cam.camera_id}
                  className={`flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 rounded-lg border px-4 py-3 ${
                    dead ? "border-red-300 bg-red-50" : ""
                  }`}
                >
                  <div className="flex items-baseline gap-2">
                    <Gauge
                      className={`h-4 w-4 shrink-0 self-center ${
                        dead
                          ? "text-red-600"
                          : stale || gated
                            ? "text-muted-foreground"
                            : keepingUp
                              ? "text-green-600"
                              : "text-amber-600"
                      }`}
                    />
                    <span className="text-sm font-medium">{groups.label(cam.camera_id)}</span>
                    {dead && (
                      <span className="rounded bg-red-600 px-1.5 py-0.5 text-[10px] font-medium text-white">
                        推論已停止
                      </span>
                    )}
                    {gated && !dead && (
                      <span className="rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-medium text-slate-600">
                        無影像 · 心跳取樣
                      </span>
                    )}
                  </div>
                  <p className="text-sm tabular-nums">
                    {/* Inferred / delivered — the pair, not just the numerator. */}
                    <span
                      className={dead ? "font-semibold text-red-600" : "font-semibold"}
                      title="推論 FPS ／ 攝影機進幀 FPS"
                    >
                      {dead ? "—" : fps.toFixed(1)}
                      <span className="font-normal text-muted-foreground">
                        {` / ${captureFps.toFixed(1)} FPS`}
                      </span>
                    </span>
                    <span className="mx-2 text-muted-foreground">·</span>
                    <span
                      className={
                        dead
                          ? "text-red-600"
                          : gated
                            ? "text-muted-foreground"
                            : coverageOk
                              ? "text-green-700"
                              : "text-amber-600"
                      }
                      title={`推論覆蓋率：每秒進 ${captureFps.toFixed(1)} 幀，推論了 ${fps.toFixed(
                        1
                      )} 幀（上限 ${(ceiling * 100).toFixed(0)}%，FRAME_SKIP=${
                        cameraFrameSkip
                      }）`}
                    >
                      覆蓋率 {dead ? "0" : (coverage * 100).toFixed(0)}%
                    </span>
                    <span className="mx-2 text-muted-foreground">·</span>
                    <span className="text-muted-foreground">
                      {cam.inference_ms_avg > 0 && !dead
                        ? `${cam.inference_ms_avg.toFixed(0)} ms`
                        : "—"}
                    </span>
                  </p>
                  <p className="w-full text-[11px] sm:w-auto sm:ml-auto">
                    {dead ? (
                      <span className="font-medium text-red-600">
                        {`偵測執行緒已停止 · ${formatSince(cam.seconds_since_last)}無推論 · 這台攝影機目前沒有在監測，看門狗會自動重啟`}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">
                        {stale ? (
                          `${formatSince(cam.seconds_since_last)}無推論`
                        ) : gated ? (
                          `畫面無影像（變異數 ${cam.frame_stddev?.toFixed(1) ?? "—"}）· 心跳取樣中，GPU 讓給看得見的攝影機 · 累計 ${cam.frames_processed.toLocaleString()} 幀`
                        ) : (
                          <>
                            {`累計 ${cam.frames_processed.toLocaleString()} 幀 · 目標 ~${targetFps.toFixed(1)} FPS`}
                            {/* FPS above counts DISTINCT frames; 空轉 is how often
                                this camera's GPU turn found no new frame, 重複 is
                                inference re-run on a frame already processed and
                                must stay at zero. */}
                            {cam.idle_ratio > 0 && ` · 空轉 ${(cam.idle_ratio * 100).toFixed(0)}%`}
                            {cam.duplicate_inferences > 0 && (
                              <span className="ml-1 font-medium text-red-600">
                                重複推論 {cam.duplicate_inferences.toLocaleString()} 次
                              </span>
                            )}
                          </>
                        )}
                        {/* Restarts are the scar tissue: a worker that keeps
                            coming back is a bug that has not been found yet. */}
                        {!!cam.worker_revivals && ` · 已重啟 ${cam.worker_revivals} 次`}
                        {!!cam.worker_crashes && ` · 迴圈重啟 ${cam.worker_crashes} 次`}
                      </span>
                    )}
                  </p>
                </div>
              );
            })}
            {groups.offline.length > 0 && (
              <p className="text-[11px] text-muted-foreground">
                離線（不顯示推論）： {groups.offline.map((c) => groups.label(c.camera_id)).join("、")}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
