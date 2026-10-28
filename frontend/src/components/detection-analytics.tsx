"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { StatTile } from "@/components/stats-cards";
import { format, parseISO, addDays, subDays, startOfWeek, getDay } from "date-fns";
import { Activity, CalendarDays, Clock, Download, FileText, TrendingUp } from "lucide-react";

import {
  API_BASE,
  fetcher,
  cameraLabel,
  type Camera,
  type DetectionHeatmap,
} from "@/lib/api";

// ── Color scale ───────────────────────────────────────────────────────────────
// Single-hue sequential blue ramp (light→dark), validated against the white
// card surface: lightness strictly monotonic, adjacent CVD ΔE ≥ 17.8. Zero
// recedes into the surface; the table view + tooltips carry exact values.
const ZERO_COLOR = "#f1f5f9";
const RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];

const WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"]; // Monday-first

const RANGE_PRESETS = [
  { days: 30, label: "最近 30 天" },
  { days: 90, label: "最近 90 天" },
  { days: 183, label: "最近半年" },
  { days: 366, label: "最近一年" },
] as const;

/** Monday-first weekday index (0 = 週一 … 6 = 週日). */
function dowIndex(d: Date): number {
  return (getDay(d) + 6) % 7;
}

function hourRange(h: number): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(h)}:00–${pad((h + 1) % 24)}:00`;
}

function pct(n: number, total: number): string {
  return total > 0 ? `${((n / total) * 100).toFixed(1)}%` : "0%";
}

/**
 * Quantile-binned sequential scale (GitHub-style): nonzero values are split
 * into five levels at their 20/40/60/80th percentiles, so a handful of
 * outlier days can't wash every other day down to the lightest step.
 */
function makeScale(values: number[]): (v: number) => string {
  const nz = values.filter((v) => v > 0).sort((a, b) => a - b);
  if (nz.length === 0) return () => ZERO_COLOR;
  const q = (p: number) => {
    const pos = (nz.length - 1) * p;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    return nz[lo] + (nz[hi] - nz[lo]) * (pos - lo);
  };
  const thresholds = [0.2, 0.4, 0.6, 0.8].map(q);
  return (v: number) => {
    if (v <= 0) return ZERO_COLOR;
    const level = thresholds.filter((t) => v >= t).length;
    return RAMP[Math.min(level, RAMP.length - 1)];
  };
}

function niceCeil(v: number): number {
  if (v <= 5) return 5;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (m * p >= v) return m * p;
  }
  return 10 * p;
}

// ── Shared tooltip ────────────────────────────────────────────────────────────

interface Tip {
  x: number;
  y: number;
  value: string;
  label: string;
}

type OnTip = (tip: Tip | null) => void;

function ChartTooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null;
  const left =
    typeof window !== "undefined"
      ? Math.min(Math.max(tip.x, 80), window.innerWidth - 80)
      : tip.x;
  return (
    <div
      className="pointer-events-none fixed z-50 -translate-x-1/2 rounded-md border bg-white px-2.5 py-1.5 text-xs shadow-md"
      style={{ left, top: tip.y - 12, transform: "translate(-50%, -100%)" }}
    >
      <div className="font-semibold text-foreground">{tip.value}</div>
      <div className="mt-0.5 text-muted-foreground">{tip.label}</div>
    </div>
  );
}

function ScaleLegend() {
  return (
    <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
      <span className="mr-0.5">少</span>
      {[ZERO_COLOR, ...RAMP].map((c) => (
        <span
          key={c}
          className="inline-block h-2.5 w-2.5 rounded-[2px]"
          style={{ backgroundColor: c }}
        />
      ))}
      <span className="ml-0.5">多</span>
    </div>
  );
}

// ── Calendar heatmap (GitHub-style, weeks as columns) ─────────────────────────

const CAL = { cell: 12, gap: 2, left: 30, top: 20 };

function CalendarHeatmap({
  dayTotals,
  startDate,
  endDate,
  onTip,
}: {
  dayTotals: Map<string, number>;
  startDate: string;
  endDate: string;
  onTip: OnTip;
}) {
  const scale = useMemo(
    () => makeScale(Array.from(dayTotals.values())),
    [dayTotals]
  );

  const start = parseISO(startDate);
  const end = parseISO(endDate);
  const gridStart = startOfWeek(start, { weekStartsOn: 1 });

  const weeks: Date[][] = [];
  for (let cur = gridStart; cur <= end; cur = addDays(cur, 7)) {
    weeks.push(Array.from({ length: 7 }, (_, i) => addDays(cur, i)));
  }

  const pitch = CAL.cell + CAL.gap;
  const width = CAL.left + weeks.length * pitch + 4;
  const height = CAL.top + 7 * pitch + 4;

  // Month labels: one per month change, skipped if it would crowd the previous.
  const monthLabels: { x: number; text: string }[] = [];
  let prevMonth = -1;
  let prevLabelCol = -10;
  weeks.forEach((col, w) => {
    const firstInRange = col.find((d) => d >= start && d <= end);
    if (!firstInRange) return;
    const m = firstInRange.getMonth();
    if (m !== prevMonth) {
      if (w - prevLabelCol >= 3) {
        monthLabels.push({ x: CAL.left + w * pitch, text: `${m + 1}月` });
        prevLabelCol = w;
      }
      prevMonth = m;
    }
  });

  return (
    <svg width={width} height={height} role="img" aria-label="每日偵測筆數日曆熱力圖">
      {monthLabels.map((l) => (
        <text key={l.x} x={l.x} y={12} className="fill-muted-foreground text-[10px]">
          {l.text}
        </text>
      ))}
      {[0, 2, 4, 6].map((row) => (
        <text
          key={row}
          x={CAL.left - 8}
          y={CAL.top + row * pitch + CAL.cell - 2}
          textAnchor="end"
          className="fill-muted-foreground text-[10px]"
        >
          {WEEKDAYS[row]}
        </text>
      ))}
      {weeks.map((col, w) =>
        col.map((d, row) => {
          if (d < start || d > end) return null;
          const key = format(d, "yyyy-MM-dd");
          const n = dayTotals.get(key) ?? 0;
          return (
            <rect
              key={key}
              x={CAL.left + w * pitch}
              y={CAL.top + row * pitch}
              width={CAL.cell}
              height={CAL.cell}
              rx={2}
              fill={scale(n)}
              strokeWidth={1}
              className="stroke-transparent hover:stroke-slate-400"
              onPointerMove={(e) =>
                onTip({
                  x: e.clientX,
                  y: e.clientY,
                  value: `${n.toLocaleString()} 筆`,
                  label: `${format(d, "yyyy/MM/dd")}（週${WEEKDAYS[row]}）`,
                })
              }
              onPointerLeave={() => onTip(null)}
            />
          );
        })
      )}
    </svg>
  );
}

// ── Weekday × hour heatmap ────────────────────────────────────────────────────

const DH = { cellW: 24, cellH: 16, gap: 2, left: 38, top: 18 };

function DowHourHeatmap({
  matrix,
  total,
  onTip,
}: {
  matrix: number[][]; // [7][24], Monday-first
  total: number;
  onTip: OnTip;
}) {
  const scale = useMemo(() => makeScale(matrix.flat()), [matrix]);
  const pitchX = DH.cellW + DH.gap;
  const pitchY = DH.cellH + DH.gap;
  const width = DH.left + 24 * pitchX + 4;
  const height = DH.top + 7 * pitchY + 4;

  return (
    <svg width={width} height={height} role="img" aria-label="星期與時段分布熱力圖">
      {Array.from({ length: 8 }, (_, i) => i * 3).map((h) =>
        h < 24 ? (
          <text
            key={h}
            x={DH.left + h * pitchX}
            y={12}
            className="fill-muted-foreground text-[10px]"
          >
            {String(h).padStart(2, "0")}
          </text>
        ) : null
      )}
      {WEEKDAYS.map((w, row) => (
        <text
          key={w}
          x={DH.left - 8}
          y={DH.top + row * pitchY + DH.cellH - 4}
          textAnchor="end"
          className="fill-muted-foreground text-[10px]"
        >
          週{w}
        </text>
      ))}
      {matrix.map((rowVals, row) =>
        rowVals.map((n, h) => (
          <rect
            key={`${row}-${h}`}
            x={DH.left + h * pitchX}
            y={DH.top + row * pitchY}
            width={DH.cellW}
            height={DH.cellH}
            rx={2}
            fill={scale(n)}
            strokeWidth={1}
            className="stroke-transparent hover:stroke-slate-400"
            onPointerMove={(e) =>
              onTip({
                x: e.clientX,
                y: e.clientY,
                value: `${n.toLocaleString()} 筆（${pct(n, total)}）`,
                label: `週${WEEKDAYS[row]} ${hourRange(h)}`,
              })
            }
            onPointerLeave={() => onTip(null)}
          />
        ))
      )}
    </svg>
  );
}

// ── 24-hour distribution bar chart ────────────────────────────────────────────

function HourBarChart({
  hourTotals,
  total,
  onTip,
}: {
  hourTotals: number[];
  total: number;
  onTip: OnTip;
}) {
  const [hovered, setHovered] = useState<number | null>(null);
  const W = 720;
  const H = 190;
  const M = { l: 40, r: 8, t: 18, b: 22 };
  const plotW = W - M.l - M.r;
  const plotH = H - M.t - M.b;
  const slot = plotW / 24;
  const barW = Math.min(24, slot - 4);

  const max = Math.max(...hourTotals, 0);
  const yMax = niceCeil(max);
  const y = (v: number) => M.t + plotH - (v / yMax) * plotH;
  const peakHour = max > 0 ? hourTotals.indexOf(max) : -1;
  const ticks = Array.from(new Set([0, Math.round(yMax / 2), yMax]));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      role="img"
      aria-label="24 小時偵測分布長條圖"
    >
      {ticks.map((t) => (
        <g key={t}>
          <line
            x1={M.l}
            x2={W - M.r}
            y1={y(t)}
            y2={y(t)}
            stroke="#e2e8f0"
            strokeWidth={1}
          />
          <text
            x={M.l - 6}
            y={y(t) + 3}
            textAnchor="end"
            className="fill-muted-foreground text-[10px] tabular-nums"
          >
            {t.toLocaleString()}
          </text>
        </g>
      ))}
      {hourTotals.map((n, h) => {
        const x = M.l + h * slot + (slot - barW) / 2;
        const barTop = y(n);
        const hgt = M.t + plotH - barTop;
        const r = Math.min(4, hgt);
        return (
          <g key={h}>
            {n > 0 && (
              <path
                d={`M ${x} ${M.t + plotH} L ${x} ${barTop + r} Q ${x} ${barTop} ${x + r} ${barTop} L ${x + barW - r} ${barTop} Q ${x + barW} ${barTop} ${x + barW} ${barTop + r} L ${x + barW} ${M.t + plotH} Z`}
                fill={hovered === h ? "#5598e7" : "#2563eb"}
              />
            )}
            {/* Oversized transparent hit area so every hour is hoverable. */}
            <rect
              x={M.l + h * slot}
              y={M.t}
              width={slot}
              height={plotH}
              fill="transparent"
              onPointerMove={(e) => {
                setHovered(h);
                onTip({
                  x: e.clientX,
                  y: e.clientY,
                  value: `${n.toLocaleString()} 筆（${pct(n, total)}）`,
                  label: hourRange(h),
                });
              }}
              onPointerLeave={() => {
                setHovered(null);
                onTip(null);
              }}
            />
            {h === peakHour && (
              <text
                x={x + barW / 2}
                y={barTop - 5}
                textAnchor="middle"
                className="fill-foreground text-[11px] font-semibold"
              >
                {n.toLocaleString()}
              </text>
            )}
          </g>
        );
      })}
      <line
        x1={M.l}
        x2={W - M.r}
        y1={M.t + plotH}
        y2={M.t + plotH}
        stroke="#cbd5e1"
        strokeWidth={1}
      />
      {Array.from({ length: 8 }, (_, i) => i * 3).map((h) => (
        <text
          key={h}
          x={M.l + h * slot + slot / 2}
          y={H - 8}
          textAnchor="middle"
          className="fill-muted-foreground text-[10px] tabular-nums"
        >
          {String(h).padStart(2, "0")}
        </text>
      ))}
    </svg>
  );
}

// ── Hourly stats table (accessibility fallback + report copy source) ──────────

function HourTableHalf({
  hourTotals,
  total,
  offset,
}: {
  hourTotals: number[];
  total: number;
  offset: number;
}) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b text-muted-foreground">
          <th className="py-1.5 text-left font-medium">時段</th>
          <th className="py-1.5 text-right font-medium">筆數</th>
          <th className="py-1.5 text-right font-medium">佔比</th>
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: 12 }, (_, i) => {
          const h = offset + i;
          const n = hourTotals[h];
          return (
            <tr key={h} className="border-b border-border/60 last:border-0">
              <td className="py-1.5 font-mono text-muted-foreground">{hourRange(h)}</td>
              <td className="py-1.5 text-right font-mono font-medium">
                {n.toLocaleString()}
              </td>
              <td className="py-1.5 text-right font-mono text-muted-foreground">
                {pct(n, total)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function HourTable({ hourTotals, total }: { hourTotals: number[]; total: number }) {
  return (
    <div className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
      <HourTableHalf hourTotals={hourTotals} total={total} offset={0} />
      <HourTableHalf hourTotals={hourTotals} total={total} offset={12} />
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function DetectionAnalytics() {
  const [rangeDays, setRangeDays] = useState<number>(90);
  const [cameraSel, setCameraSel] = useState<string>("all");
  const [tip, setTip] = useState<Tip | null>(null);

  // One fetch covers the longest preset; range/camera filtering happens
  // client-side so switching filters never blanks the charts.
  const { data, isLoading } = useSWR<DetectionHeatmap>(
    `${API_BASE}/api/detections/heatmap?days=366`,
    fetcher,
    { refreshInterval: 60000, keepPreviousData: true }
  );
  const { data: cameras } = useSWR<Camera[]>(`${API_BASE}/api/cameras`, fetcher, {
    refreshInterval: 10000,
  });

  const stats = useMemo(() => {
    if (!data) return null;

    const end = parseISO(data.end);
    const startDate = format(subDays(end, rangeDays - 1), "yyyy-MM-dd");
    // Never extend past what the API actually returned.
    const start = startDate > data.start ? startDate : data.start;

    const cells = data.cells.filter(
      (c) =>
        c.date >= start && (cameraSel === "all" || c.camera_id === cameraSel)
    );

    const dayTotals = new Map<string, number>();
    const hourTotals = Array.from({ length: 24 }, () => 0);
    const dowHour = Array.from({ length: 7 }, () => Array.from({ length: 24 }, () => 0));
    const dowCache = new Map<string, number>();
    let total = 0;

    for (const c of cells) {
      total += c.count;
      dayTotals.set(c.date, (dayTotals.get(c.date) ?? 0) + c.count);
      hourTotals[c.hour] += c.count;
      let dow = dowCache.get(c.date);
      if (dow === undefined) {
        dow = dowIndex(parseISO(c.date));
        dowCache.set(c.date, dow);
      }
      dowHour[dow][c.hour] += c.count;
    }

    const dowTotals = dowHour.map((row) => row.reduce((a, b) => a + b, 0));
    const peakHour = hourTotals.indexOf(Math.max(...hourTotals));
    const peakDow = dowTotals.indexOf(Math.max(...dowTotals));
    let maxDay: { date: string; count: number } | null = null;
    for (const [date, count] of dayTotals) {
      if (!maxDay || count > maxDay.count) maxDay = { date, count };
    }

    return {
      start,
      end: data.end,
      cells,
      dayTotals,
      hourTotals,
      dowHour,
      dowTotals,
      total,
      peakHour,
      peakDow,
      maxDay,
    };
  }, [data, rangeDays, cameraSel]);

  // Camera options: registered cameras plus any ids that only exist in the
  // data (detections outlive camera deletion).
  const cameraOptions = useMemo(() => {
    const byId = new Map<string, string>();
    for (const cam of cameras ?? []) byId.set(cam.camera_id, cameraLabel(cam));
    for (const c of data?.cells ?? []) {
      if (!byId.has(c.camera_id)) byId.set(c.camera_id, c.camera_id);
    }
    return Array.from(byId, ([id, label]) => ({ id, label }));
  }, [cameras, data]);

  const camName =
    cameraSel === "all"
      ? "全部攝影機"
      : cameraOptions.find((c) => c.id === cameraSel)?.label ?? cameraSel;

  const exportCsv = () => {
    if (!stats) return;
    const nameById = new Map(cameraOptions.map((c) => [c.id, c.label]));
    // Camera names are user-defined — quote every field so commas/quotes
    // in them can't shift columns.
    const field = (v: string | number) => `"${String(v).replace(/"/g, '""')}"`;
    const rows = stats.cells.map((c) =>
      [
        c.date,
        `週${WEEKDAYS[dowIndex(parseISO(c.date))]}`,
        hourRange(c.hour),
        nameById.get(c.camera_id) ?? c.camera_id,
        c.count,
      ]
        .map(field)
        .join(",")
    );
    // BOM so Excel opens the Chinese headers correctly.
    const csv = "\uFEFF日期,星期,時段,攝影機,筆數\n" + rows.join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `detection-stats_${stats.start}_${stats.end}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const selectClass =
    "appearance-none rounded-md border bg-white py-1.5 pl-3 pr-8 text-xs font-medium text-foreground transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary/30";

  if (isLoading && !data) {
    return (
      <div className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="rounded-lg border bg-white px-5 py-4">
              <span className="inline-block h-4 w-20 animate-pulse rounded bg-muted" />
              <div className="mt-3">
                <span className="inline-block h-7 w-16 animate-pulse rounded bg-muted" />
              </div>
            </div>
          ))}
        </div>
        <div className="h-48 animate-pulse rounded-lg border bg-white" />
        <div className="h-64 animate-pulse rounded-lg border bg-white" />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Filter row — scopes every chart, stat, and table below it. */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="inline-flex items-center rounded-lg border bg-white p-0.5">
          {RANGE_PRESETS.map((p) => (
            <button
              key={p.days}
              onClick={() => setRangeDays(p.days)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                rangeDays === p.days
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
        <select
          value={cameraSel}
          onChange={(e) => setCameraSel(e.target.value)}
          className={selectClass}
        >
          <option value="all">全部攝影機</option>
          {cameraOptions.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label}
            </option>
          ))}
        </select>
        <button
          onClick={exportCsv}
          disabled={!stats || stats.total === 0}
          className="ml-auto flex items-center gap-1.5 rounded-md border bg-white px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted disabled:opacity-40"
        >
          <Download className="h-3.5 w-3.5" />
          匯出 CSV
        </button>
      </div>

      {stats && stats.total === 0 ? (
        <div className="flex items-center justify-center rounded-lg border border-dashed bg-white py-20 text-sm text-muted-foreground">
          此範圍內沒有偵測紀錄
        </div>
      ) : stats ? (
        <>
          {/* Summary tiles */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile
              title="期間累計偵測"
              value={`${stats.total.toLocaleString()} 筆`}
              sub={`${stats.start} ～ ${stats.end}`}
              icon={Activity}
              color="text-blue-500"
            />
            <StatTile
              title="尖峰時段"
              value={hourRange(stats.peakHour)}
              sub={`${stats.hourTotals[stats.peakHour].toLocaleString()} 筆，佔 ${pct(stats.hourTotals[stats.peakHour], stats.total)}`}
              icon={Clock}
              color="text-orange-500"
            />
            <StatTile
              title="最活躍星期"
              value={`週${WEEKDAYS[stats.peakDow]}`}
              sub={`${stats.dowTotals[stats.peakDow].toLocaleString()} 筆，佔 ${pct(stats.dowTotals[stats.peakDow], stats.total)}`}
              icon={CalendarDays}
              color="text-green-500"
            />
            <StatTile
              title="單日最高"
              value={`${(stats.maxDay?.count ?? 0).toLocaleString()} 筆`}
              sub={stats.maxDay ? format(parseISO(stats.maxDay.date), "yyyy/MM/dd") : "--"}
              icon={TrendingUp}
              color="text-violet-500"
            />
          </div>

          {/* Ready-to-paste report summary */}
          <div className="flex items-start gap-2.5 rounded-lg border bg-white px-5 py-3.5 text-xs leading-relaxed text-muted-foreground">
            <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            <p>
              <span className="font-medium text-foreground">報告摘要：</span>
              {camName}於 {stats.start} 至 {stats.end} 期間累計{" "}
              {stats.total.toLocaleString()} 筆偵測，尖峰時段為{" "}
              {hourRange(stats.peakHour)}（
              {stats.hourTotals[stats.peakHour].toLocaleString()} 筆，佔{" "}
              {pct(stats.hourTotals[stats.peakHour], stats.total)}），以週
              {WEEKDAYS[stats.peakDow]}最為活躍
              {stats.maxDay
                ? `；單日最高出現於 ${format(parseISO(stats.maxDay.date), "yyyy/MM/dd")}（${stats.maxDay.count.toLocaleString()} 筆）`
                : ""}
              。
            </p>
          </div>

          {/* Calendar heatmap */}
          <div className="rounded-lg border bg-white">
            <div className="flex items-center justify-between border-b px-5 py-3">
              <h3 className="text-sm font-semibold">每日偵測熱力圖</h3>
              <ScaleLegend />
            </div>
            <div className="overflow-x-auto p-5">
              <CalendarHeatmap
                dayTotals={stats.dayTotals}
                startDate={stats.start}
                endDate={stats.end}
                onTip={setTip}
              />
            </div>
          </div>

          {/* Weekday × hour heatmap */}
          <div className="rounded-lg border bg-white">
            <div className="flex items-center justify-between border-b px-5 py-3">
              <h3 className="text-sm font-semibold">星期 × 時段分布</h3>
              <ScaleLegend />
            </div>
            <div className="overflow-x-auto p-5">
              <DowHourHeatmap
                matrix={stats.dowHour}
                total={stats.total}
                onTip={setTip}
              />
            </div>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            {/* 24-hour distribution */}
            <div className="rounded-lg border bg-white">
              <div className="border-b px-5 py-3">
                <h3 className="text-sm font-semibold">24 小時累積分布</h3>
              </div>
              <div className="p-5">
                <HourBarChart
                  hourTotals={stats.hourTotals}
                  total={stats.total}
                  onTip={setTip}
                />
              </div>
            </div>

            {/* Hourly table */}
            <div className="rounded-lg border bg-white">
              <div className="border-b px-5 py-3">
                <h3 className="text-sm font-semibold">每小時統計表</h3>
              </div>
              <div className="p-5">
                <HourTable hourTotals={stats.hourTotals} total={stats.total} />
              </div>
            </div>
          </div>
        </>
      ) : null}

      <ChartTooltip tip={tip} />
    </div>
  );
}
