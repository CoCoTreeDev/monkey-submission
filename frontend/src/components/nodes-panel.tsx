"use client";

import { useState } from "react";
import useSWR from "swr";
import { ChevronRight, Cloud, Cpu, Loader2, Server, X } from "lucide-react";
import {
  API_BASE,
  fetcher,
  cameraLabel,
  deleteNode,
  type Camera,
  type EdgeStatus,
  type NodeHistory,
  type NodeHistoryPoint,
} from "@/lib/api";
import { NodeUtilChart } from "@/components/node-util-chart";

interface NodeInfo extends EdgeStatus {
  hostname: string;
}

const RANGES = [
  { label: "1小時", minutes: 60 },
  { label: "3小時", minutes: 180 },
  { label: "6小時", minutes: 360 },
  { label: "24小時", minutes: 1440 },
];

// Stable empty-history reference — a fresh [] every render would break the
// chart's React.memo even when nothing changed.
const EMPTY_HISTORY: NodeHistoryPoint[] = [];

function isApiNode(node: NodeInfo): boolean {
  return node.heartbeat?.node_role === "api" || node.hostname === "azure-api";
}

function UtilBar({ label, percent, detail }: { label: string; percent: number | null; detail?: string }) {
  const pct = percent == null ? null : Math.min(100, Math.max(0, percent));
  const tone =
    pct == null ? "bg-gray-300" : pct >= 90 ? "bg-red-500" : pct >= 70 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono tabular-nums">
          {pct == null ? "—" : `${pct.toFixed(0)}%`}
          {detail && <span className="ml-1 text-muted-foreground">{detail}</span>}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-gray-100">
        <div className={`h-full rounded-full transition-all ${tone}`} style={{ width: `${pct ?? 0}%` }} />
      </div>
    </div>
  );
}

function NodeCard({
  node,
  labelFor,
  onRemove,
  history,
  rangeMinutes,
  chartOpen,
  onToggleChart,
}: {
  node: NodeInfo;
  labelFor: (id: string) => string;
  onRemove: (hostname: string) => void;
  history: NodeHistoryPoint[];
  rangeMinutes: number;
  chartOpen: boolean;
  onToggleChart: () => void;
}) {
  const hb = node.heartbeat;
  const isApi = hb?.node_role === "api";
  return (
    <div className="overflow-hidden rounded-lg border bg-white">
      {/* RunPod-style header strip */}
      <div className="flex items-center justify-between border-b bg-neutral-900 px-5 py-3.5">
        <div className="flex items-center gap-2.5">
          {isApi ? <Cloud className="h-4 w-4 text-white/70" /> : <Server className="h-4 w-4 text-white/70" />}
          <span className="font-mono text-sm font-semibold text-white">
            {isApi ? "雲端 API" : node.hostname}
          </span>
          <span className="rounded bg-white/10 px-1.5 py-0.5 text-[11px] text-white/60">
            {isApi ? "Azure Container App" : "推論節點 · Apple Silicon"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
              node.online ? "bg-emerald-400/15 text-emerald-300" : "bg-red-400/15 text-red-300"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${node.online ? "animate-pulse bg-emerald-400" : "bg-red-400"}`}
            />
            {node.online ? "運轉中" : "失聯"}
          </span>
          {!node.online && (
            <button
              onClick={() => onRemove(node.hostname)}
              className="rounded p-1 text-white/50 transition hover:bg-white/10 hover:text-white"
              title="從列表移除此節點（節點若還活著會在下次心跳重新出現）"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="grid gap-5 p-5 sm:grid-cols-2">
        {/* Utilization */}
        <div className="space-y-3.5">
          {!isApi && <UtilBar label="GPU（推論）" percent={hb?.gpu_percent ?? null} />}
          <UtilBar label="CPU" percent={hb?.cpu_percent ?? null} />
          <UtilBar
            label="記憶體"
            percent={hb?.memory_percent ?? null}
            detail={
              hb?.memory_used_gb != null && hb?.memory_total_gb != null
                ? `${hb.memory_used_gb.toFixed(1)} / ${hb.memory_total_gb.toFixed(0)} GB`
                : undefined
            }
          />
          <p className="text-xs text-muted-foreground">
            {isApi ? "API 程序" : "推論程序"}{" "}
            {hb?.process_rss_gb != null ? `${hb.process_rss_gb.toFixed(2)} GB` : "—"} · 最後心跳{" "}
            {node.seconds_since != null ? `${Math.round(node.seconds_since)} 秒前` : "—"}
          </p>
        </div>

        {/* Cameras on this node */}
        <div>
          <p className="mb-2 text-xs font-medium text-muted-foreground">此節點上的攝影機</p>
          {isApi ? (
            <p className="text-xs text-muted-foreground">
              （雲端 API 節點 — 提供 API、資料庫與通知，不執行推論）
            </p>
          ) : !hb?.cameras?.length ? (
            <p className="text-xs text-muted-foreground">（尚無攝影機工作階段）</p>
          ) : (
            <div className="space-y-1.5">
              {hb.cameras.map((cam) => (
                <div
                  key={cam.camera_id}
                  className="flex items-center justify-between rounded border bg-neutral-50 px-2.5 py-1.5 text-xs"
                >
                  <span className="flex items-center gap-2">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        cam.worker_alive === false
                          ? "bg-red-500"
                          : cam.stream_connected
                            ? "bg-emerald-500"
                            : "bg-gray-300"
                      }`}
                    />
                    {labelFor(cam.camera_id)}
                    {/* A connected stream says nothing about the detector THREAD:
                        on 2026-08-08 one died and this dot stayed green for 62
                        hours while the camera was not being watched at all. */}
                    {cam.worker_alive === false && (
                      <span
                        className="rounded bg-red-600 px-1.5 py-0.5 text-[10px] font-medium text-white"
                        title="偵測執行緒已停止 — 串流仍連線，但畫面沒有在推論。看門狗會自動重啟。"
                      >
                        推論已停止
                      </span>
                    )}
                    {/* The idle gate is deliberate throttling, not a fault — a
                        black feed sampled at a heartbeat so the GPU goes to the
                        cameras that can see. Labelled so a low FPS here never
                        reads as a stalled camera. */}
                    {cam.idle_gated && cam.worker_alive !== false && (
                      <span
                        className="rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-medium text-slate-600"
                        title={`畫面無影像（像素變異數 ${cam.frame_stddev?.toFixed(1) ?? "—"}）— 已降為心跳取樣，GPU 讓給看得見的攝影機。畫面一亮即恢復滿速。`}
                      >
                        無影像 · 心跳取樣
                      </span>
                    )}
                  </span>
                  <span className="font-mono tabular-nums text-muted-foreground">
                    {cam.sample_fps.toFixed(1)}
                    {/* Inferred FPS alone cannot show what was missed — the
                        delivered rate and the coverage between them can. */}
                    {!!cam.capture_fps && `/${cam.capture_fps.toFixed(1)}`} FPS
                    {cam.coverage_ratio != null && (
                      <span title="推論覆蓋率 = 推論 FPS ÷ 進幀 FPS">
                        {` · 覆蓋 ${(cam.coverage_ratio * 100).toFixed(0)}%`}
                      </span>
                    )}
                    {` · ${cam.inference_ms_avg.toFixed(0)} ms`}
                    {/* FPS here counts DISTINCT frames only; duplicate inference
                        is shown separately so it can never read as throughput. */}
                    {(cam.gated_ratio ?? 0) > 0 && (
                      <> · 節流 {(cam.gated_ratio! * 100).toFixed(0)}%</>
                    )}
                    {cam.idle_ratio > 0 && <> · 空轉 {(cam.idle_ratio * 100).toFixed(0)}%</>}
                    {cam.duplicate_inferences > 0 && (
                      <span className="ml-1 text-red-600">
                        重複 {((cam.duplicate_inferences / Math.max(cam.total_inferences, 1)) * 100).toFixed(0)}%
                      </span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Utilisation over time — collapsed by default so the tab renders fast;
          the chart SVG is not rendered at all while collapsed. */}
      <div className="border-t px-5 pb-4 pt-3.5">
        <button
          onClick={onToggleChart}
          aria-expanded={chartOpen}
          className="flex w-full items-center gap-1.5 text-xs font-medium text-muted-foreground transition hover:text-foreground"
        >
          <ChevronRight
            className={`h-3.5 w-3.5 transition-transform ${chartOpen ? "rotate-90" : ""}`}
          />
          使用率趨勢
          {!chartOpen && <span className="font-normal text-muted-foreground/70">（點擊展開）</span>}
        </button>
        {chartOpen && (
          <div className="mt-2">
            <NodeUtilChart points={history} minutes={rangeMinutes} />
          </div>
        )}
      </div>
    </div>
  );
}

/** 節點 — RunPod-style inference-node dashboard (admin tab). */
export function NodesPanel() {
  const { data, isLoading, mutate } = useSWR<{ nodes: NodeInfo[] }>(
    `${API_BASE}/api/admin/nodes`,
    fetcher,
    { refreshInterval: 5000 }
  );
  const [rangeMinutes, setRangeMinutes] = useState(180);
  // Per-card chart visibility, keyed by hostname — all collapsed by default.
  const [openCharts, setOpenCharts] = useState<Record<string, boolean>>({});
  const anyChartOpen = Object.values(openCharts).some(Boolean);
  // History polling pauses entirely (null key) while every chart is collapsed.
  const { data: history } = useSWR<{ nodes: NodeHistory[] }>(
    anyChartOpen ? `${API_BASE}/api/admin/nodes/history?minutes=${rangeMinutes}` : null,
    fetcher,
    { refreshInterval: 30000 }
  );
  const historyById = new Map((history?.nodes ?? []).map((n) => [n.node_id, n.points]));
  const [error, setError] = useState<string | null>(null);

  const handleRemove = async (hostname: string) => {
    setError(null);
    try {
      await deleteNode(hostname);
      await mutate();
    } catch (e) {
      setError(e instanceof Error ? e.message : "移除節點失敗");
    }
  };
  const { data: cameras } = useSWR<Camera[]>(`${API_BASE}/api/cameras`, fetcher, {
    refreshInterval: 30000,
  });
  const labelById = new Map((cameras ?? []).map((c) => [c.camera_id, cameraLabel(c)]));
  const labelFor = (id: string) => labelById.get(id) ?? id;

  // 推論節點 (edge) cards first, the 雲端 API card last; ties by node id.
  const sortedNodes = [...(data?.nodes ?? [])].sort((a, b) => {
    const apiDiff = Number(isApiNode(a)) - Number(isApiNode(b));
    return apiDiff !== 0 ? apiDiff : a.hostname.localeCompare(b.hostname);
  });

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <Cpu className="h-4 w-4 text-muted-foreground" />
        <h2 className="text-sm font-semibold">運算節點</h2>
        {data && (
          <span className="text-xs text-muted-foreground">
            {data.nodes.filter((n) => n.online).length} / {data.nodes.length} 在線
          </span>
        )}
        {/* Chart time range — applies to every node card */}
        <div className="ml-auto flex items-center gap-1">
          {RANGES.map((r) => (
            <button
              key={r.minutes}
              onClick={() => setRangeMinutes(r.minutes)}
              className={`rounded px-2 py-1 text-xs transition ${
                rangeMinutes === r.minutes
                  ? "bg-neutral-900 text-white"
                  : "text-muted-foreground hover:bg-neutral-100"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {isLoading && !data ? (
        <div className="flex items-center gap-2 rounded-lg border bg-white p-5 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          載入中…
        </div>
      ) : !data?.nodes.length ? (
        <div className="rounded-lg border bg-white p-8 text-center text-sm text-muted-foreground">
          尚未收到任何節點心跳 — 推論節點上線後會自動出現在這裡
        </div>
      ) : (
        sortedNodes.map((n) => (
          <NodeCard
            key={n.hostname}
            node={n}
            labelFor={labelFor}
            onRemove={handleRemove}
            history={historyById.get(n.hostname) ?? EMPTY_HISTORY}
            rangeMinutes={rangeMinutes}
            chartOpen={!!openCharts[n.hostname]}
            onToggleChart={() =>
              setOpenCharts((m) => ({ ...m, [n.hostname]: !m[n.hostname] }))
            }
          />
        ))
      )}

      {error && <p className="text-xs text-red-600">{error}</p>}
    </div>
  );
}
