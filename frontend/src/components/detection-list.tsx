"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { format } from "date-fns";
import { ArrowDown, ArrowDownWideNarrow, ArrowUp, ChevronDown, ChevronRight, Video } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { DetectionThumbnail } from "@/components/detection-thumbnail";
import type { Camera, Detection } from "@/lib/api";

export type DetectionSort = "time" | "count";
export type DetectionSortOrder = "asc" | "desc";

const DETECTION_SORT_OPTIONS: { value: DetectionSort; label: string }[] = [
  { value: "time", label: "時間" },
  { value: "count", label: "猴子數" },
];

interface DetectionListPanelProps {
  detections: Detection[];
  isLoading: boolean;
  total: number;
  page: number;
  totalPages: number;
  pageSize: number;
  pageSizeOptions: number[];
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  sortBy: DetectionSort;
  onSortByChange: (sort: DetectionSort) => void;
  sortOrder: DetectionSortOrder;
  onSortOrderChange: () => void;
  selectedIdx: number;
  onSelectIdx: (idx: number) => void;
  cameras?: Camera[];
  cameraFilter: string;
  onCameraFilterChange: (value: string) => void;
  // Hidden for non-admins: the confidence % is not rendered at all.
  showConfidence?: boolean;
}

/** Compact page window: prev/curr/next around the current page, clamped. */
function pageWindow(current: number, total: number): number[] {
  const start = Math.max(1, Math.min(current - 1, total - 2));
  const end = Math.min(total, start + 2);
  const out: number[] = [];
  for (let p = start; p <= end; p++) out.push(p);
  return out;
}

/** Consecutive rows sharing a visit, in the order the list already has them.
 *
 *  A visit is not a second way to browse the log — it is what one stretch of
 *  animals in front of a camera IS, and the rows are the animals inside it. So
 *  it groups the list in place rather than living in its own tab: the rows keep
 *  their component, their selection index, their thumbnails and their pane on
 *  the right, and a visit is a header they collapse under.
 *
 *  Rows with no visit_id (recorded before visits existed) come through as
 *  single-row groups with no header, which is exactly how they looked before.
 */
interface VisitGroup {
  key: string;
  visitId: string | null;
  rows: { det: Detection; idx: number }[];
  peak: number | null;
}

function groupByVisit(detections: Detection[]): VisitGroup[] {
  const out: VisitGroup[] = [];
  detections.forEach((det, idx) => {
    const vid = det.visit_id ?? null;
    const last = out[out.length - 1];
    if (vid && last && last.visitId === vid) {
      last.rows.push({ det, idx });
      return;
    }
    out.push({
      key: vid ?? `row-${det.id}`,
      visitId: vid,
      rows: [{ det, idx }],
      peak: det.visit_peak ?? null,
    });
  });
  return out;
}

function confColor(conf: number) {
  if (conf >= 0.85) return "text-red-600";
  if (conf >= 0.65) return "text-orange-600";
  return "text-muted-foreground";
}

export function DetectionListPanel({
  detections,
  isLoading,
  total,
  page,
  totalPages,
  pageSize,
  pageSizeOptions,
  onPageChange,
  onPageSizeChange,
  sortBy,
  onSortByChange,
  sortOrder,
  onSortOrderChange,
  selectedIdx,
  onSelectIdx,
  cameras,
  cameraFilter,
  onCameraFilterChange,
  showConfidence = true,
}: DetectionListPanelProps) {
  // Collapsed by default would hide the rows people came for; open by default
  // keeps today's list visible and lets a busy visit be folded away.
  const [collapsedVisits, setCollapsedVisits] = useState<Set<string>>(new Set());
  const groups = useMemo(() => groupByVisit(detections), [detections]);
  const open = useMemo(() => {
    const s = new Set(groups.map((g) => g.visitId).filter(Boolean) as string[]);
    collapsedVisits.forEach((v) => s.delete(v));
    return s;
  }, [groups, collapsedVisits]);
  const toggle = (visitId: string) =>
    setCollapsedVisits((prev) => {
      const next = new Set(prev);
      if (next.has(visitId)) next.delete(visitId);
      else next.add(visitId);
      return next;
    });

  const showCameraFilter = !!cameras && cameras.length >= 1;
  const goto = (p: number) => {
    const clamped = Math.max(1, Math.min(totalPages, p));
    if (clamped !== page) onPageChange(clamped);
  };

  // Keep the selected row visible as ←/→ navigation moves it (including the
  // re-scroll that should happen after a page flip loads fresh items).
  const activeRef = useRef<HTMLButtonElement>(null);
  const sortMenuRef = useRef<HTMLDivElement>(null);
  const [sortMenuOpen, setSortMenuOpen] = useState(false);
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedIdx, detections]);
  useEffect(() => {
    if (!sortMenuOpen) return;

    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (!sortMenuRef.current?.contains(event.target as Node)) {
        setSortMenuOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSortMenuOpen(false);
    };

    document.addEventListener("pointerdown", closeOnOutsidePointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [sortMenuOpen]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-white">
      {/* Header: title + count */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="text-sm font-semibold">偵測紀錄</h3>
        <span className="text-xs text-muted-foreground">
          共 <span className="font-mono font-medium text-foreground">{total}</span> 筆
        </span>
      </div>

      {/* Toolbar: group-by camera (left) + page size (right), justify-between */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-muted/30 px-4 py-2.5">
        {showCameraFilter ? (
          <div className="relative">
            <Video className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <select
              value={cameraFilter}
              onChange={(e) => onCameraFilterChange(e.target.value)}
              className="appearance-none rounded-md border bg-white py-1.5 pl-7 pr-7 text-xs font-medium text-foreground transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
            >
              <option value="all">全部攝影機</option>
              {cameras!.map((cam) => (
                <option key={cam.camera_id} value={cam.camera_id}>
                  {cam.name || cam.camera_id}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-1.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          </div>
        ) : (
          <span />
        )}
        <div className="flex shrink-0 flex-nowrap items-center justify-end gap-2">
          <div ref={sortMenuRef} className="relative shrink-0">
            <button
              type="button"
              onClick={() => setSortMenuOpen((open) => !open)}
              aria-haspopup="menu"
              aria-expanded={sortMenuOpen}
              aria-controls="detection-sort-menu"
              aria-label={`排序依據：${DETECTION_SORT_OPTIONS.find((option) => option.value === sortBy)?.label}`}
              title={`排序依據：${DETECTION_SORT_OPTIONS.find((option) => option.value === sortBy)?.label}`}
              className="inline-flex h-8 w-8 items-center justify-center rounded-md border bg-white text-foreground transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
            >
              <ArrowDownWideNarrow className="h-4 w-4" aria-hidden="true" />
            </button>
            {sortMenuOpen && (
              <div
                id="detection-sort-menu"
                role="menu"
                aria-label="排序依據"
                className="absolute right-0 top-full z-30 mt-1 min-w-[116px] rounded-md border bg-white p-1 shadow-lg"
              >
                <div className="px-2 py-1 text-[11px] font-medium text-muted-foreground">
                  排序依據
                </div>
                {DETECTION_SORT_OPTIONS.map((option) => {
                  const selected = sortBy === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      role="menuitemradio"
                      aria-checked={selected}
                      onClick={() => {
                        onSortByChange(option.value);
                        setSortMenuOpen(false);
                      }}
                      className={`flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-xs transition-colors hover:bg-muted ${
                        selected ? "bg-primary/10 font-medium text-primary" : "text-foreground"
                      }`}
                    >
                      <span>{option.label}</span>
                      {selected && <span aria-hidden="true">✓</span>}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={onSortOrderChange}
            aria-label={sortOrder === "desc" ? "目前降冪，切換為升冪" : "目前升冪，切換為降冪"}
            title={sortOrder === "desc" ? "降冪，點擊改為升冪" : "升冪，點擊改為降冪"}
            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md border bg-white text-foreground transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
          >
            {sortOrder === "desc" ? (
              <ArrowDown className="h-4 w-4" aria-hidden="true" />
            ) : (
              <ArrowUp className="h-4 w-4" aria-hidden="true" />
            )}
          </button>
          <label className="flex shrink-0 items-center gap-1.5 whitespace-nowrap text-xs text-muted-foreground">
            每頁
            <span className="relative">
              <select
                value={pageSize}
                onChange={(e) => onPageSizeChange(Number(e.target.value))}
                className="appearance-none rounded-md border bg-white py-1.5 pl-2.5 pr-7 text-xs font-medium text-foreground transition-colors hover:bg-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
              >
                {pageSizeOptions.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-1.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            </span>
          </label>
        </div>
      </div>

      {/* Scrollable list — fixed on mobile, fills remaining height on lg. */}
      <ScrollArea className="h-[520px] min-h-0 lg:h-0 lg:flex-1">
        <div className="divide-y">
          {isLoading ? (
            Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="flex items-center gap-3 px-4 py-2.5">
                <div className="h-12 w-12 shrink-0 animate-pulse rounded-md bg-muted" />
                <div className="flex-1 space-y-1.5">
                  <div className="h-3 w-28 animate-pulse rounded bg-muted" />
                  <div className="h-3 w-20 animate-pulse rounded bg-muted" />
                </div>
              </div>
            ))
          ) : detections.length === 0 ? (
            <div className="py-20 text-center text-sm text-muted-foreground">
              尚無偵測紀錄。
            </div>
          ) : (
            groups.map((group) => {
              const collapsed = group.visitId !== null && !open.has(group.visitId);
              const holdsSelection = group.rows.some((r) => r.idx === selectedIdx);
              return (
              <div key={group.key}>
              {group.visitId && (
                <button
                  type="button"
                  onClick={() => toggle(group.visitId!)}
                  aria-expanded={!collapsed}
                  className={`flex w-full items-center gap-2 border-y bg-muted/40 px-4 py-1.5 text-left text-xs transition-colors hover:bg-muted/70 ${
                    holdsSelection ? "border-l-2 border-l-primary" : ""
                  }`}
                >
                  {collapsed ? (
                    <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  )}
                  <span className="font-mono text-[11px] text-muted-foreground">
                    {format(new Date(group.rows[group.rows.length - 1].det.timestamp), "MM-dd HH:mm")}
                  </span>
                  <span className="font-medium">
                    {group.peak !== null ? `${group.peak} 隻` : `${group.rows.length} 筆`}
                  </span>
                  {/* Only when they disagree: the count filed vs the count seen
                      at once. That gap IS the duplicate the log used to hide. */}
                  {group.peak !== null && group.rows.length > group.peak && (
                    <span
                      className="rounded bg-rose-50 px-1.5 py-0.5 text-[10px] font-medium text-rose-700"
                      title={`這次造訪記了 ${group.rows.length} 筆，但同時最多只有 ${group.peak} 隻 — 多出來的是同一隻被記了不只一次`}
                    >
                      {group.rows.length} 筆 / {group.peak} 隻
                    </span>
                  )}
                  <span className="ml-auto text-[10px] text-muted-foreground">
                    {group.rows.length} 筆
                  </span>
                </button>
              )}
              {!collapsed && group.rows.map(({ det, idx }) => {
              const active = idx === selectedIdx;
              return (
                <button
                  key={det.id}
                  ref={active ? activeRef : undefined}
                  onClick={() => onSelectIdx(idx)}
                  className={`flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                    active
                      ? "bg-primary/10"
                      : "hover:bg-muted/40"
                  }`}
                >
                  {/* Selection accent */}
                  <span
                    className={`-ml-4 mr-0 h-12 w-0.5 rounded-r ${
                      active ? "bg-primary" : "bg-transparent"
                    }`}
                  />
                  {/* Thumbnail */}
                  <div className="h-12 w-12 shrink-0 overflow-hidden rounded-md border bg-neutral-900">
                    <DetectionThumbnail
                      detection={det}
                      alt={`偵測 ${det.id}`}
                      loading="lazy"
                      className="h-full w-full object-cover"
                    />
                  </div>
                  {/* Meta */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-medium text-foreground">
                        {format(new Date(det.timestamp), "MM-dd HH:mm:ss")}
                      </span>
                      {showConfidence && (
                        <span
                          className={`ml-auto font-mono text-xs font-semibold ${confColor(det.confidence)}`}
                        >
                          {(det.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <div className="mt-1 flex items-center gap-1.5">
                      <span className="truncate rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                        {det.camera_id}
                      </span>
                      {det.track_id >= 0 && (
                        <span className="rounded bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-600">
                          #{det.track_id}
                        </span>
                      )}
                    </div>
                  </div>
                </button>
              );
            })}
              </div>
              );
            })
          )}
        </div>
      </ScrollArea>

      {/* Footer: pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between border-t px-3 py-2.5">
          <span className="text-[11px] text-muted-foreground">
            第 {page} / {totalPages} 頁
          </span>
          <Pagination className="mx-0 w-auto justify-end">
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious
                  href="#"
                  size="sm"
                  onClick={(e) => {
                    e.preventDefault();
                    goto(page - 1);
                  }}
                  aria-disabled={page <= 1}
                  className={page <= 1 ? "pointer-events-none opacity-40" : undefined}
                />
              </PaginationItem>
              {pageWindow(page, totalPages).map((p) => (
                <PaginationItem key={p}>
                  <PaginationLink
                    href="#"
                    size="icon"
                    isActive={p === page}
                    onClick={(e) => {
                      e.preventDefault();
                      goto(p);
                    }}
                  >
                    {p}
                  </PaginationLink>
                </PaginationItem>
              ))}
              <PaginationItem>
                <PaginationNext
                  href="#"
                  size="sm"
                  onClick={(e) => {
                    e.preventDefault();
                    goto(page + 1);
                  }}
                  aria-disabled={page >= totalPages}
                  className={
                    page >= totalPages ? "pointer-events-none opacity-40" : undefined
                  }
                />
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        </div>
      )}
    </div>
  );
}
