"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";
import type { NodeHistoryPoint } from "@/lib/api";

// Series colors — categorical slots 1–3 of the validated dataviz palette
// (CVD-safe as a set; the sub-3:1 aqua carries a visible value label).
// The node cards are always white (like the rest of the 節點 panel), so these
// light-surface steps hold in both app themes.
const CPU_COLOR = "#2a78d6"; // blue
const GPU_COLOR = "#eb6834"; // orange
const RSS_COLOR = "#1baf7a"; // aqua
const GRID_COLOR = "#e7e5e4"; // hairline gridlines
const MUTED_INK = "#8a8a85"; // axis/tick labels
const VALUE_INK = "#404040"; // last-value labels

// Fixed viewBox — scales uniformly to the card width.
const W = 640;
const H = 176;
const PAD_L = 30;
const PAD_R = 58;
const PCT_TOP = 10; // CPU/GPU pane, 0–100%
const PCT_H = 96;
const RSS_TOP = 132; // RSS pane, own scale
const RSS_H = 28;
const PLOT_W = W - PAD_L - PAD_R;

const RSS_CEILINGS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128];

// Downsample ceiling — a 24h range at 30s cadence is ~2880 raw points; more
// than ~120 is invisible at this plot width and just slows path building.
const MAX_POINTS = 120;

interface Pt {
  x: number; // epoch ms
  cpu: number | null;
  gpu: number | null;
  rss: number | null;
}

/** Backend timestamps are naive TW-local — pin them to +08:00 (same as parseTW). */
function parseTs(t: string): number {
  return new Date(/([+-]\d{2}:?\d{2}|Z)$/.test(t) ? t : `${t}+08:00`).getTime();
}

function fmtHHmm(ms: number): string {
  return new Date(ms).toLocaleTimeString("zh-TW", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Taipei",
  });
}

/** Evenly-strided downsample to ≤MAX_POINTS, always keeping the newest point. */
function downsample(pts: Pt[]): Pt[] {
  if (pts.length <= MAX_POINTS) return pts;
  const stride = Math.ceil(pts.length / MAX_POINTS);
  const out: Pt[] = [];
  for (let i = pts.length - 1; i >= 0; i -= stride) out.push(pts[i]);
  return out.reverse();
}

/** Polyline path with pen-up over null gaps (a lost heartbeat breaks the line). */
function buildPath(
  pts: Pt[],
  get: (p: Pt) => number | null,
  sx: (x: number) => number,
  sy: (v: number) => number
): string {
  let d = "";
  let pen = false;
  for (const p of pts) {
    const v = get(p);
    if (v == null || Number.isNaN(v)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${sx(p.x).toFixed(1)} ${sy(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}

function lastValue(pts: Pt[], get: (p: Pt) => number | null): { x: number; v: number } | null {
  for (let i = pts.length - 1; i >= 0; i--) {
    const v = get(pts[i]);
    if (v != null && !Number.isNaN(v)) return { x: pts[i].x, v };
  }
  return null;
}

/**
 * 節點使用率趨勢 — dependency-free SVG line chart.
 * CPU% / GPU% share the 0–100 pane; RSS gets a thin pane with its own scale.
 *
 * Perf: memo'd against parent re-renders (the panel polls every 5s); all
 * geometry + path strings are computed once per data refresh via useMemo, so
 * hover renders only move the crosshair; pointermove is rAF-throttled.
 */
export const NodeUtilChart = memo(function NodeUtilChart({
  points,
  minutes,
}: {
  points: NodeHistoryPoint[];
  minutes: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [now, setNow] = useState(0);
  const rafRef = useRef<number | null>(null);

  // Drop any pending hover frame on unmount.
  useEffect(() => {
    const updateNow = () => setNow(Date.now());
    const nowFrame = requestAnimationFrame(updateNow);
    const tick = window.setInterval(updateNow, 30_000);
    return () => {
      cancelAnimationFrame(nowFrame);
      window.clearInterval(tick);
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Parse + sort + downsample once per data refresh, not on every render.
  const pts = useMemo<Pt[]>(
    () =>
      downsample(
        points
          .map((p) => ({ x: parseTs(p.t), cpu: p.cpu, gpu: p.gpu, rss: p.rss_gb }))
          .filter((p) => !Number.isNaN(p.x))
          .sort((a, b) => a.x - b.x)
      ),
    [points]
  );

  // Scales, path strings, labels and ticks — keyed on the data reference so
  // hover-only renders (and the 5s node poll) never rebuild them.
  const model = useMemo(() => {
    if (pts.length < 2) return null;

    const hasGpu = pts.some((p) => p.gpu != null);
    const x1 = Math.max(now || pts[pts.length - 1].x, pts[pts.length - 1].x);
    const x0 = x1 - minutes * 60_000;
    const rssData = pts.map((p) => p.rss).filter((v): v is number => v != null);
    const rssMax = RSS_CEILINGS.find((c) => c >= Math.max(...rssData, 0.1) * 1.05) ?? 128;

    const sx = (x: number) => PAD_L + ((x - x0) / (x1 - x0)) * PLOT_W;
    const syPct = (v: number) => PCT_TOP + PCT_H * (1 - Math.min(100, Math.max(0, v)) / 100);
    const syRss = (v: number) => RSS_TOP + RSS_H * (1 - Math.min(rssMax, Math.max(0, v)) / rssMax);

    const cpuPath = buildPath(pts, (p) => p.cpu, sx, syPct);
    const gpuPath = hasGpu ? buildPath(pts, (p) => p.gpu, sx, syPct) : "";
    const rssPath = buildPath(pts, (p) => p.rss, sx, syRss);

    // Last-value labels in the right margin; nudge CPU/GPU apart if they collide.
    const cpuLast = lastValue(pts, (p) => p.cpu);
    const gpuLast = hasGpu ? lastValue(pts, (p) => p.gpu) : null;
    const rssLast = lastValue(pts, (p) => p.rss);
    let cpuLabelY = cpuLast ? syPct(cpuLast.v) : 0;
    let gpuLabelY = gpuLast ? syPct(gpuLast.v) : 0;
    if (cpuLast && gpuLast && Math.abs(cpuLabelY - gpuLabelY) < 11) {
      const mid = (cpuLabelY + gpuLabelY) / 2;
      const cpuAbove = cpuLabelY <= gpuLabelY;
      cpuLabelY = cpuAbove ? mid - 5.5 : mid + 5.5;
      gpuLabelY = cpuAbove ? mid + 5.5 : mid - 5.5;
    }

    const ticks = [0, 1 / 3, 2 / 3, 1].map((f) => ({
      vx: PAD_L + f * PLOT_W,
      label: fmtHHmm(x0 + f * (x1 - x0)),
      anchor: (f === 0 ? "start" : f === 1 ? "end" : "middle") as "start" | "middle" | "end",
    }));

    return {
      hasGpu,
      x0,
      x1,
      rssMax,
      sx,
      syPct,
      syRss,
      cpuPath,
      gpuPath,
      rssPath,
      cpuLast,
      gpuLast,
      rssLast,
      cpuLabelY,
      gpuLabelY,
      ticks,
    };
  }, [pts, minutes, now]);

  if (!model) {
    return (
      <p className="py-3 text-xs text-muted-foreground">
        資料蒐集中（每 30 秒一筆）— 圖表將於數分鐘內出現。歷史紀錄存於記憶體，重新部署後會重新累積。
      </p>
    );
  }

  const { hasGpu, x0, x1, rssMax, sx, syPct, syRss } = model;

  // rAF-throttled crosshair: at most one nearest-point search + setState per
  // frame; the latest pointer position wins.
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX;
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const vx = ((clientX - rect.left) / rect.width) * W;
      if (vx < PAD_L - 4 || vx > W - PAD_R + 4) {
        setHover(null);
        return;
      }
      const tx = x0 + ((vx - PAD_L) / PLOT_W) * (x1 - x0);
      let best = 0;
      let bestD = Infinity;
      for (let i = 0; i < pts.length; i++) {
        const d = Math.abs(pts[i].x - tx);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      setHover(best);
    });
  };

  const onLeave = () => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setHover(null);
  };

  const hp = hover != null && hover < pts.length ? pts[hover] : null;
  const hoverFrac = hp ? sx(hp.x) / W : 0;

  return (
    <div>
      {/* Legend — identity never rides on color alone; names + current values */}
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ background: CPU_COLOR }} />
          CPU 使用率
        </span>
        {hasGpu && (
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: GPU_COLOR }} />
            GPU
          </span>
        )}
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ background: RSS_COLOR }} />
          記憶體（RSS）
        </span>
      </div>

      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full"
          role="img"
          aria-label="節點使用率趨勢圖：CPU、GPU 與記憶體"
          onPointerMove={onMove}
          onPointerLeave={onLeave}
        >
          {/* percent pane grid: 0 / 50 / 100 */}
          {[0, 50, 100].map((v) => (
            <g key={v}>
              <line x1={PAD_L} x2={W - PAD_R} y1={syPct(v)} y2={syPct(v)} stroke={GRID_COLOR} strokeWidth="1" />
              <text
                x={PAD_L - 5}
                y={syPct(v) + 3}
                textAnchor="end"
                fontSize="9"
                fill={MUTED_INK}
                className="tabular-nums"
              >
                {v}
              </text>
            </g>
          ))}

          {/* RSS pane grid: ceiling + baseline */}
          <line x1={PAD_L} x2={W - PAD_R} y1={syRss(rssMax)} y2={syRss(rssMax)} stroke={GRID_COLOR} strokeWidth="1" />
          <line x1={PAD_L} x2={W - PAD_R} y1={syRss(0)} y2={syRss(0)} stroke={GRID_COLOR} strokeWidth="1" />
          <text x={PAD_L - 5} y={syRss(rssMax) + 3} textAnchor="end" fontSize="9" fill={MUTED_INK} className="tabular-nums">
            {rssMax < 1 ? rssMax.toFixed(1) : rssMax}
          </text>
          <text x={PAD_L - 5} y={syRss(0) + 3} textAnchor="end" fontSize="9" fill={MUTED_INK}>
            GB
          </text>

          {/* time ticks */}
          {model.ticks.map((t) => (
            <text
              key={t.vx}
              x={t.vx}
              y={H - 4}
              textAnchor={t.anchor}
              fontSize="9"
              fill={MUTED_INK}
              className="tabular-nums"
            >
              {t.label}
            </text>
          ))}

          {/* series */}
          <path d={model.cpuPath} fill="none" stroke={CPU_COLOR} strokeWidth="2" strokeLinejoin="round" />
          {hasGpu && (
            <path d={model.gpuPath} fill="none" stroke={GPU_COLOR} strokeWidth="2" strokeLinejoin="round" />
          )}
          <path d={model.rssPath} fill="none" stroke={RSS_COLOR} strokeWidth="1.5" strokeLinejoin="round" />

          {/* last-value labels (ink text + colored end-dot) */}
          {model.cpuLast && (
            <g>
              <circle cx={sx(model.cpuLast.x)} cy={syPct(model.cpuLast.v)} r="2.5" fill={CPU_COLOR} />
              <text x={W - PAD_R + 7} y={model.cpuLabelY + 3} fontSize="10" fill={VALUE_INK} className="tabular-nums">
                {model.cpuLast.v.toFixed(0)}%
              </text>
            </g>
          )}
          {model.gpuLast && (
            <g>
              <circle cx={sx(model.gpuLast.x)} cy={syPct(model.gpuLast.v)} r="2.5" fill={GPU_COLOR} />
              <text x={W - PAD_R + 7} y={model.gpuLabelY + 3} fontSize="10" fill={VALUE_INK} className="tabular-nums">
                {model.gpuLast.v.toFixed(0)}%
              </text>
            </g>
          )}
          {model.rssLast && (
            <g>
              <circle cx={sx(model.rssLast.x)} cy={syRss(model.rssLast.v)} r="2.5" fill={RSS_COLOR} />
              <text x={W - PAD_R + 7} y={syRss(model.rssLast.v) + 3} fontSize="10" fill={VALUE_INK} className="tabular-nums">
                {model.rssLast.v.toFixed(2)} GB
              </text>
            </g>
          )}

          {/* hover crosshair */}
          {hp && (
            <line x1={sx(hp.x)} x2={sx(hp.x)} y1={PCT_TOP} y2={syRss(0)} stroke={MUTED_INK} strokeWidth="1" strokeDasharray="3 3" />
          )}
        </svg>

        {/* hover tooltip */}
        {hp && (
          <div
            className="pointer-events-none absolute top-0 z-10 rounded border bg-white px-2.5 py-1.5 text-xs shadow-sm"
            style={{
              left: `${hoverFrac * 100}%`,
              transform: hoverFrac > 0.72 ? "translateX(-104%)" : hoverFrac < 0.12 ? "translateX(4%)" : "translateX(-50%)",
            }}
          >
            <p className="mb-0.5 font-medium tabular-nums" style={{ color: VALUE_INK }}>
              {fmtHHmm(hp.x)}
            </p>
            <p className="flex items-center gap-1.5 tabular-nums text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: CPU_COLOR }} />
              CPU {hp.cpu != null ? `${hp.cpu.toFixed(0)}%` : "—"}
            </p>
            {hasGpu && (
              <p className="flex items-center gap-1.5 tabular-nums text-muted-foreground">
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: GPU_COLOR }} />
                GPU {hp.gpu != null ? `${hp.gpu.toFixed(0)}%` : "—"}
              </p>
            )}
            <p className="flex items-center gap-1.5 tabular-nums text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: RSS_COLOR }} />
              記憶體 {hp.rss != null ? `${hp.rss.toFixed(2)} GB` : "—"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
});
