"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MessageBanner, useFlashMessage } from "@/components/status-message";
import useSWR, { mutate as globalMutate } from "swr";
import useSWRInfinite from "swr/infinite";
import { format } from "date-fns";
import { duplicateUploadText } from "@/lib/utils";
import {
  Download,
  Trash2,
  Loader2,
  CheckSquare,
  Square,
  XSquare,
  ImagePlus,
  Lock,
  Unlock,
  PlayCircle,
  Send,
} from "lucide-react";
import {
  API_BASE,
  fetcher,
  cameraLabel,
  downloadImagesUrl,
  downloadSelectedImages,
  deleteSelectedDetections,
  setDetectionLock,
  setDetectionsLockBulk,
  setDetectionsVisibilityBulk,
  addToTrainingDataset,
  type Camera,
  type Detection,
  type DetectionList,
} from "@/lib/api";
import { DetectionThumbnail } from "@/components/detection-thumbnail";
import { DetectionModal } from "@/components/detection-modal";

// One screenful-and-a-bit per request. Divides evenly by every grid column
// count below (3/4/5/6), so a fully-loaded page never ends in a ragged row.
const PAGE_SIZE = 60;

/** 影像管理 — detection-image grid (multi-select → 下載所選 / 刪除所選 /
 *  加入訓練資料集) + filter download. */
export function ImageManagement({ canTraining = false }: { canTraining?: boolean }) {
  // Infinite scroll over the offset-paged endpoint. `revalidateAll: false` +
  // `revalidateFirstPage: true` is what keeps the 10 s poll cheap once the
  // admin is deep in the list: the interval refetches page 0 only (where new
  // detections land), while `mutateDetections()` with no arguments — what the
  // lock/delete handlers call — still forces every loaded page to refetch.
  const {
    data: pages,
    error: pagesError,
    isLoading: detectionsLoading,
    size,
    setSize,
    mutate: mutateDetections,
  } = useSWRInfinite<DetectionList>(
    (index, previous) => {
      // A short page is the last page; stop asking for more.
      if (previous && previous.items.length < PAGE_SIZE) return null;
      return `${API_BASE}/api/detections?limit=${PAGE_SIZE}&offset=${index * PAGE_SIZE}`;
    },
    fetcher,
    { refreshInterval: 10000, revalidateAll: false, revalidateFirstPage: true }
  );
  const { data: cameras } = useSWR<Camera[]>(`${API_BASE}/api/cameras`, fetcher, {
    refreshInterval: 10000,
  });

  // Offsets shift under us: a detection arriving (or being deleted) while the
  // admin is scrolled deep moves every later row across a page boundary, so the
  // same id comes back on two pages. Keep the first copy — pages are
  // newest-first, so that is the one whose position matches what is on screen.
  const items = useMemo(() => {
    const seen = new Set<number>();
    const out: Detection[] = [];
    for (const page of pages ?? []) {
      for (const det of page?.items ?? []) {
        if (seen.has(det.id)) continue;
        seen.add(det.id);
        out.push(det);
      }
    }
    return out;
  }, [pages]);
  const total = pages?.[0]?.total ?? items.length;
  const lastPage = pages?.[pages.length - 1];
  // `data[size - 1]` is undefined while the batch for the newest page is still
  // in flight — SWR keeps the previous, shorter array until it resolves.
  const isLoadingMore = !!pages && size > 0 && pages[size - 1] === undefined;
  const hasMore =
    !!lastPage && lastPage.items.length === PAGE_SIZE && items.length < total;

  // Sentinel below the grid pulls the next page in as it nears the viewport.
  //
  // `pages.length` is in the deps on purpose, and the observer is rebuilt (not
  // just left running) on every page that lands. An IntersectionObserver only
  // reports *changes*: it fires once when you observe an element and then stays
  // silent for as long as that element remains intersecting. Once the list is
  // long enough that appending 60 tiles no longer pushes the sentinel back out
  // of the 600px margin, the sentinel never leaves the observed zone, no change
  // ever occurs, and auto-loading dies silently at whatever page it reached
  // (measured: it stalled at 7 pages and no amount of scrolling revived it).
  // Re-observing re-delivers the current state, which keeps the chain going and
  // stops on its own as soon as hasMore goes false.
  const pagesLoaded = pages?.length ?? 0;
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = sentinelRef.current;
    // Never auto-retry a failed page — that would spin the endpoint. The footer
    // offers an explicit 重試 instead.
    if (!el || !hasMore || isLoadingMore || pagesError) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) setSize((s) => s + 1);
      },
      { rootMargin: "600px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [hasMore, isLoadingMore, pagesError, pagesLoaded, setSize]);

  const [selected, setSelected] = useState<Set<number>>(new Set());
  // The record open in the review modal, held as an ID for the same reason
  // `anchorIdRef` is: the 10s poll re-fetches page 0, so a detection arriving
  // at the head shifts every index down by one. Stored as an index, the modal
  // would silently swap to the neighbouring record mid-review — which is the
  // exact judgement this pane exists to support.
  const [reviewId, setReviewId] = useState<number | null>(null);
  const [reviewDeleting, setReviewDeleting] = useState(false);
  const [reviewDeleteError, setReviewDeleteError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [locking, setLocking] = useState(false);
  const [lockBusyId, setLockBusyId] = useState<number | null>(null);
  const [downloadingSelected, setDownloadingSelected] = useState(false);
  const [addingToDataset, setAddingToDataset] = useState(false);
  const [publishingAll, setPublishingAll] = useState(false);
  const [cameraId, setCameraId] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const hasFilters = cameraId || dateFrom || dateTo;
  const [message, flash] = useFlashMessage();

  const toggleSelect = (id: number) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  // Anchor for shift-click range selection. Stored as an id (not index) so it
  // survives the 10s SWR refresh reordering/removing items.
  const anchorIdRef = useRef<number | null>(null);
  const handleCardClick = (id: number, shiftKey: boolean) => {
    if (shiftKey && anchorIdRef.current !== null) {
      const anchorIdx = items.findIndex((d) => d.id === anchorIdRef.current);
      const targetIdx = items.findIndex((d) => d.id === id);
      if (anchorIdx !== -1 && targetIdx !== -1) {
        const [from, to] =
          anchorIdx < targetIdx ? [anchorIdx, targetIdx] : [targetIdx, anchorIdx];
        const rangeIds = items.slice(from, to + 1).map((d) => d.id);
        setSelected((prev) => new Set([...prev, ...rangeIds]));
        return;
      }
    }
    toggleSelect(id);
    anchorIdRef.current = id;
  };
  // Locked rows are still selectable — 下載所選 and 加入訓練資料集 apply to them
  // fine. It is only delete they are held back from.
  const lockedIds = new Set(items.filter((d) => d.locked).map((d) => d.id));
  const selectedIds = Array.from(selected);
  const selectedLocked = selectedIds.filter((id) => lockedIds.has(id));
  const selectedUnlocked = selectedIds.filter((id) => !lockedIds.has(id));

  const selectAll = () => setSelected(new Set(items.map((d) => d.id)));
  const deselectAll = () => {
    setSelected(new Set());
    anchorIdRef.current = null;
  };

  // Image management uses an infinite grid rather than numbered pages. The
  // current page is therefore the set already rendered/loaded; do not make
  // Ctrl+A issue an unbounded request for every record in the database.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "a") return;
      if (reviewId !== null) return;
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable)
      ) {
        return;
      }
      event.preventDefault();
      setSelected(new Set(items.map((d) => d.id)));
      anchorIdRef.current = null;
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [items, reviewId]);

  const handlePublishAll = async () => {
    if (publishingAll) return;
    if (!window.confirm("確定要將全部偵測紀錄推送給使用者嗎？")) return;
    setPublishingAll(true);
    try {
      const result = await setDetectionsVisibilityBulk(true);
      flash(
        "success",
        result.updated > 0
          ? `已推送 ${result.updated} 筆偵測紀錄給使用者`
          : "全部偵測紀錄都已推送給使用者"
      );
      await mutateDetections();
      globalMutate((key) => typeof key === "string" && key.includes("/api/detections"));
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "推送給使用者失敗");
    } finally {
      setPublishingAll(false);
    }
  };

  const handleDownloadSelected = async () => {
    if (selected.size === 0) return;
    setDownloadingSelected(true);
    try {
      await downloadSelectedImages(Array.from(selected));
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "下載失敗");
    } finally {
      setDownloadingSelected(false);
    }
  };

  /** Upload the RAW snapshots of the given detections into the Ultralytics
   *  Platform training dataset. They arrive unlabeled = background negatives
   *  until boxes are drawn in the Platform UI. Shared by the toolbar button
   *  (selection) and the right-click context menu (single or selection). */
  const addIdsToDataset = async (ids: number[]) => {
    if (ids.length === 0 || addingToDataset) return;
    setAddingToDataset(true);
    try {
      const result = await addToTrainingDataset(ids);
      const failedNote =
        result.failed.length > 0 ? `，${result.failed.length} 張失敗` : "";
      flash(
        result.uploaded > 0 ? "success" : "error",
        `已上傳 ${result.uploaded} 張（未標註＝背景負樣本，正樣本請到 Platform 補框）${failedNote}${duplicateUploadText(result.duplicates)}`
      );
      // Only clear the selection when it was what we uploaded.
      if (result.uploaded > 0 && ids.every((id) => selected.has(id))) {
        setSelected(new Set());
      }
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "加入訓練資料集失敗");
    } finally {
      setAddingToDataset(false);
    }
  };

  const handleAddToDataset = () => addIdsToDataset(Array.from(selected));

  // Right-click context menu on a card. If the clicked card is part of a
  // multi-selection the action applies to the whole selection, otherwise to
  // just that card.
  const [menu, setMenu] = useState<{ x: number; y: number; id: number } | null>(null);
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenu(null);
    };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", onKey);
    };
  }, [menu]);
  const menuTargetIds =
    menu && selected.has(menu.id) && selected.size > 1
      ? Array.from(selected)
      : menu
        ? [menu.id]
        : [];
  // -1 both when nothing is open and when the open record has left the list
  // (deleted elsewhere) — either way the modal should not be showing.
  const reviewIdx = reviewId === null ? -1 : items.findIndex((d) => d.id === reviewId);
  // A stale inline error must not follow the modal to the next record.
  useEffect(() => {
    setReviewDeleteError(null);
  }, [reviewId]);
  // The record's clip. 檢視影片 is enabled exactly when there is one to play —
  // measured 2026-08-28, 865 of 866 rows have it, so this is not a dead item.
  const menuRow = menu ? items.find((d) => d.id === menu.id) : undefined;
  const menuClipFile = menuRow?.trail_path || null;

  /** Lock = 「這筆我看過，確定是真的，不要刪」。刪除是硬刪且無復原，所以這是
   *  審核過的紀錄唯一的保護。鎖定與可見性互不影響。 */
  const handleLockSelected = async (locked: boolean) => {
    const ids = locked ? selectedUnlocked : selectedLocked;
    if (ids.length === 0 || locking || deleting) return;
    setLocking(true);
    try {
      const result = await setDetectionsLockBulk(ids, locked);
      flash(
        "success",
        locked
          ? `已鎖定 ${result.updated} 筆，刪除時會保留`
          : `已解除 ${result.updated} 筆的鎖定`
      );
      mutateDetections();
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "更新鎖定失敗");
    } finally {
      setLocking(false);
    }
  };

  const handleToggleLock = async (id: number, locked: boolean) => {
    if (lockBusyId !== null || deleting) return;
    setLockBusyId(id);
    try {
      await setDetectionLock(id, locked);
      flash("success", locked ? `#${id} 已鎖定，刪除時會保留` : `#${id} 已解除鎖定`);
      mutateDetections();
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "更新鎖定失敗");
    } finally {
      setLockBusyId(null);
    }
  };

  /** Every successful delete refreshes the same two things. */
  const afterDelete = () => {
    mutateDetections();
    globalMutate((k) => typeof k === "string" && k.includes("/api/admin/storage-stats"));
  };

  /** Delete the record currently under review, from inside the modal.
   *
   *  Reuses the same endpoint as 刪除所選 — a second delete path would be a
   *  second place to forget the lock rule. On success it steps to the record
   *  that takes its place so triage continues; when it was the last one,
   *  `reviewId` goes null and the modal closes. The error is returned INTO the
   *  pane because the `flash` banner renders underneath the modal overlay. */
  const handleDeleteReviewed = async (det: Detection) => {
    if (reviewDeleting || det.locked) return;
    if (!window.confirm(`確定要刪除偵測 #${det.id} 及其圖片？此操作無法復原。`)) return;
    setReviewDeleting(true);
    setReviewDeleteError(null);
    try {
      const result = await deleteSelectedDetections([det.id]);
      if (result.deleted_detections === 0) {
        setReviewDeleteError(
          (result.refused_locked?.length ?? 0) > 0 ? "已鎖定，未刪除" : "未刪除任何紀錄"
        );
        return;
      }
      const nextId = items[reviewIdx + 1]?.id ?? null;
      setSelected((prev) => {
        const next = new Set(prev);
        next.delete(det.id);
        return next;
      });
      setReviewId(nextId);
      flash("success", `已刪除 #${det.id}、${result.deleted_files} 個檔案`);
      afterDelete();
    } catch (err: unknown) {
      setReviewDeleteError(err instanceof Error ? err.message : "刪除失敗");
    } finally {
      setReviewDeleting(false);
    }
  };

  const handleDeleteSelected = async () => {
    if (selected.size === 0) return;
    // Never delete while a lock write is in flight. `selectedUnlocked` is derived
    // from the last SWR snapshot, so mid-flight it still shows the rows the user
    // just locked as unlocked — sending them would hard-delete, with no undo,
    // exactly what they were protecting.
    if (locking || lockBusyId !== null) {
      flash("error", "鎖定尚未完成，請稍候再刪除");
      return;
    }
    // Locked rows never reach the request. The backend refuses them anyway and
    // reports them back as `refused_locked` — filtering here is so the confirm
    // dialog states the real number instead of one the server will contradict.
    if (selectedUnlocked.length === 0) {
      flash("error", `選取的 ${selected.size} 筆全部已鎖定，未刪除任何紀錄`);
      return;
    }
    const skipNote =
      selectedLocked.length > 0 ? `\n（另有 ${selectedLocked.length} 筆已鎖定，將保留）` : "";
    if (
      !window.confirm(
        `確定要刪除選取的 ${selectedUnlocked.length} 筆偵測紀錄及其圖片？此操作無法復原。${skipNote}`
      )
    )
      return;
    setDeleting(true);
    try {
      const result = await deleteSelectedDetections(selectedUnlocked);
      const refused = result.refused_locked?.length ?? 0;
      flash(
        "success",
        `已刪除 ${result.deleted_detections} 筆紀錄、${result.deleted_files} 個檔案` +
          (refused > 0 ? `；${refused} 筆已鎖定，未刪除` : "")
      );
      setSelected(new Set());
      afterDelete();
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "刪除失敗");
    } finally {
      setDeleting(false);
    }
  };

  const handleDownload = (filtered: boolean) => {
    const params = filtered
      ? {
          camera_id: cameraId || undefined,
          date_from: dateFrom || undefined,
          date_to: dateTo || undefined,
        }
      : undefined;
    const a = document.createElement("a");
    a.href = downloadImagesUrl(params);
    a.download = "";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  return (
    <div className="space-y-5">
      <MessageBanner message={message} />

      {/* Detection image grid */}
      <div className="rounded-lg border bg-white">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b px-5 py-3">
          <div>
            <h3 className="text-sm font-semibold">偵測影像</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {hasMore ? `已載入 ${items.length} / ${total}` : items.length} 筆偵測紀錄
              {lockedIds.size > 0 && `（${lockedIds.size} 筆已鎖定）`}
              ，點選縮圖以選取，Shift+點選可選取範圍，Ctrl/Cmd+A 選取目前已載入
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <button
              onClick={handlePublishAll}
              disabled={publishingAll || items.length === 0}
              className="flex items-center gap-1.5 rounded-md border border-primary/40 px-3 py-1.5 text-xs font-medium text-primary hover:bg-primary/5 disabled:opacity-40"
              title="將全部偵測紀錄設為使用者可見，不受目前選取範圍限制"
            >
              {publishingAll ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
              全部推送給使用者
            </button>
            <button
              onClick={selectAll}
              disabled={items.length === 0}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-30"
              title={hasMore ? `全選已載入的 ${items.length} 筆（往下捲動可載入更多）` : "全選"}
            >
              <CheckSquare className="h-3.5 w-3.5" />
              全選
            </button>
            <button
              onClick={deselectAll}
              disabled={selected.size === 0}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-30"
              title="取消全選"
            >
              <XSquare className="h-3.5 w-3.5" />
              取消
            </button>
          </div>
        </div>

        {/* Selection action toolbar — appears only when something is selected.
            top-[88px] = full header height (h-14 brand row + tab nav row); the
            background must be opaque or the grid shows through while stuck. */}
        {selected.size > 0 && (
          <div className="sticky top-[88px] z-20 flex flex-wrap items-center gap-2 border-b bg-[color-mix(in_oklab,var(--color-primary)_5%,white)] px-5 py-2.5 shadow-sm">
            <span className="text-xs font-medium">已選取 {selected.size} 筆</span>
            <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
              <button
                onClick={handleDownloadSelected}
                disabled={downloadingSelected}
                className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-40"
              >
                {downloadingSelected ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
                下載所選 ({selected.size})
              </button>
              {canTraining && (
                <button
                  onClick={handleAddToDataset}
                  disabled={addingToDataset}
                  title="上傳原始快照到 Ultralytics Platform 訓練資料集（未標註＝背景負樣本）"
                  className="flex items-center gap-1.5 rounded-md border border-primary/40 px-3 py-1.5 text-xs font-medium text-primary hover:bg-primary/5 disabled:opacity-40"
                >
                  {addingToDataset ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <ImagePlus className="h-3 w-3" />
                  )}
                  加入訓練資料集 ({selected.size})
                </button>
              )}
              {selectedUnlocked.length > 0 && (
                <button
                  onClick={() => handleLockSelected(true)}
                  disabled={locking || deleting}
                  title="鎖定後刪除會跳過這些紀錄（刪除是硬刪，無法復原）"
                  className="flex items-center gap-1.5 rounded-md border border-amber-300 px-3 py-1.5 text-xs font-medium text-amber-700 hover:bg-amber-50 disabled:opacity-40"
                >
                  {locking ? <Loader2 className="h-3 w-3 animate-spin" /> : <Lock className="h-3 w-3" />}
                  鎖定所選 ({selectedUnlocked.length})
                </button>
              )}
              {selectedLocked.length > 0 && (
                <button
                  onClick={() => handleLockSelected(false)}
                  disabled={locking || deleting}
                  title="解除鎖定後，這些紀錄就能被刪除"
                  className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium text-muted-foreground hover:bg-muted disabled:opacity-40"
                >
                  {locking ? <Loader2 className="h-3 w-3 animate-spin" /> : <Unlock className="h-3 w-3" />}
                  解除鎖定 ({selectedLocked.length})
                </button>
              )}
              <button
                onClick={handleDeleteSelected}
                disabled={deleting || locking || lockBusyId !== null || selectedUnlocked.length === 0}
                title={
                  selectedUnlocked.length === 0
                    ? "選取的紀錄全部已鎖定"
                    : selectedLocked.length > 0
                      ? `${selectedLocked.length} 筆已鎖定，將保留`
                      : undefined
                }
                className="flex items-center gap-1.5 rounded-md border border-red-200 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-40"
              >
                {deleting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                刪除所選
                {selectedLocked.length > 0 && selectedUnlocked.length > 0 &&
                  ` (${selectedUnlocked.length})`}
              </button>
              <button
                onClick={deselectAll}
                className="rounded px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                取消
              </button>
            </div>
          </div>
        )}

        <div className="p-4">
          {detectionsLoading ? (
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6">
              {Array.from({ length: 12 }).map((_, i) => (
                <div key={i} className="aspect-square animate-pulse rounded-lg bg-muted" />
              ))}
            </div>
          ) : items.length === 0 ? (
            <div className="py-12 text-center text-sm text-muted-foreground">尚無偵測紀錄</div>
          ) : (
            <div className="grid select-none grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6">
              {items.map((det) => (
                <DetectionCard
                  key={det.id}
                  detection={det}
                  isSelected={selected.has(det.id)}
                  lockBusy={lockBusyId === det.id}
                  onToggleLock={(locked) => handleToggleLock(det.id, locked)}
                  onToggle={(shiftKey) => handleCardClick(det.id, shiftKey)}
                  onContextMenu={(x, y) =>
                    setMenu({
                      // Clamp so the menu never overflows the viewport edge.
                      x: Math.min(x, window.innerWidth - 200),
                      y: Math.min(y, window.innerHeight - 120),
                      id: det.id,
                    })
                  }
                />
              ))}
            </div>
          )}

          {/* Scroll sentinel. The button is not decoration: it is the fallback
              for when the observer cannot fire (no viewport intersection after
              a filtered-down list, keyboard-only navigation). */}
          {items.length > 0 && (
            <div
              ref={sentinelRef}
              className="flex items-center justify-center pt-4 text-xs text-muted-foreground"
            >
              {isLoadingMore ? (
                <span className="flex items-center gap-1.5">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  載入更多…
                </span>
              ) : pagesError ? (
                <button
                  onClick={() => mutateDetections()}
                  className="rounded-md border border-red-200 px-3 py-1.5 text-red-600 hover:bg-red-50"
                >
                  載入失敗，點選重試
                </button>
              ) : hasMore ? (
                <button
                  onClick={() => setSize((s) => s + 1)}
                  className="rounded-md border px-3 py-1.5 hover:bg-muted hover:text-foreground"
                >
                  載入更多（還有 {total - items.length} 筆）
                </button>
              ) : (
                <span>已載入全部 {items.length} 筆</span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Reuses the 偵測檢閱 pane — clip playback, 標註↔影片 toggle, raw-frame
          download and ← / → all come from it. `page`/`totalPages` are 1 and
          `total` is the LOADED count, so the modal's "共 N 筆" matches what
          ← / → can actually reach rather than promising the 800+ still unpaged.
          `isAdmin` is intentionally not passed: DetectionLog does not pass it
          either, so the visibility override stays where it lives today — the
          inline review pane — and this modal behaves identically everywhere. */}
      <DetectionModal
        open={reviewIdx !== -1}
        onClose={() => setReviewId(null)}
        detections={items}
        selectedIdx={reviewIdx === -1 ? 0 : reviewIdx}
        onPrev={() => setReviewId(items[Math.max(0, reviewIdx - 1)]?.id ?? null)}
        onNext={() =>
          setReviewId(items[Math.min(items.length - 1, reviewIdx + 1)]?.id ?? null)
        }
        total={items.length}
        page={1}
        totalPages={1}
        canDownload
        showConfidence
        onDelete={handleDeleteReviewed}
        deleting={reviewDeleting}
        deleteError={reviewDeleteError}
      />

      {/* Right-click context menu */}
      {menu && (
        <div
          className="fixed z-50 min-w-[11rem] rounded-md border bg-white py-1 shadow-lg"
          style={{ left: menu.x, top: menu.y }}
          onClick={(e) => e.stopPropagation()}
          onContextMenu={(e) => e.preventDefault()}
        >
          <p className="px-3 py-1 text-[10px] text-muted-foreground">
            {menuTargetIds.length > 1 ? `已選取 ${menuTargetIds.length} 筆` : `偵測 #${menu.id}`}
          </p>
          {/* Deliberately NOT menuTargetIds: 新增到資料集 and 下載 act on the whole
              selection, but reviewing is a single-record act — open the card
              that was actually right-clicked. */}
          <button
            onClick={() => {
              setMenu(null);
              setReviewId(menu.id);
            }}
            disabled={!menuClipFile}
            title={menuClipFile ? "開啟偵測檢閱，可播放這筆的影片" : "這筆沒有影片"}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-muted disabled:opacity-40"
          >
            <PlayCircle className="h-3.5 w-3.5 text-primary" />
            檢視影片
          </button>
          {canTraining && (
            <button
              onClick={() => {
                setMenu(null);
                addIdsToDataset(menuTargetIds);
              }}
              disabled={addingToDataset}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-muted disabled:opacity-40"
            >
              <ImagePlus className="h-3.5 w-3.5 text-primary" />
              新增到資料集
              {menuTargetIds.length > 1 && ` (${menuTargetIds.length})`}
            </button>
          )}
          <button
            onClick={() => {
              setMenu(null);
              downloadSelectedImages(menuTargetIds).catch((err: unknown) =>
                flash("error", err instanceof Error ? err.message : "下載失敗")
              );
            }}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-muted"
          >
            <Download className="h-3.5 w-3.5" />
            下載
            {menuTargetIds.length > 1 && ` (${menuTargetIds.length})`}
          </button>
        </div>
      )}

      {/* Bulk download by filter */}
      <div className="rounded-lg border bg-white">
        <div className="border-b px-5 py-3">
          <h3 className="text-sm font-semibold">依條件下載</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            將原始快照打包為 ZIP 下載（不含標註框）。或於上方勾選縮圖後「下載所選」。
          </p>
        </div>
        <div className="p-5">
          <div className="flex flex-col gap-4">
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">攝影機</label>
                <select
                  value={cameraId}
                  onChange={(e) => setCameraId(e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                >
                  <option value="">全部攝影機</option>
                  {cameras?.map((cam) => (
                    <option key={cam.camera_id} value={cam.camera_id}>
                      {cameraLabel(cam)}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">起始日期</label>
                <input
                  type="date"
                  value={dateFrom}
                  onChange={(e) => setDateFrom(e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">結束日期</label>
                <input
                  type="date"
                  value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                />
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => handleDownload(false)}
                className="flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-xs font-medium text-white hover:bg-primary/90"
              >
                <Download className="h-3.5 w-3.5" />
                下載全部影像
              </button>
              {hasFilters && (
                <button
                  onClick={() => handleDownload(true)}
                  className="flex items-center gap-1.5 rounded-md border px-4 py-2 text-xs font-medium text-foreground hover:bg-muted"
                >
                  <Download className="h-3.5 w-3.5" />
                  下載篩選結果
                </button>
              )}
              {hasFilters && (
                <button
                  onClick={() => {
                    setCameraId("");
                    setDateFrom("");
                    setDateTo("");
                  }}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  清除篩選
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Detection image card ────────────────────────────────────────────────────

function DetectionCard({
  detection,
  isSelected,
  lockBusy,
  onToggle,
  onToggleLock,
  onContextMenu,
}: {
  detection: Detection;
  isSelected: boolean;
  lockBusy: boolean;
  onToggle: (shiftKey: boolean) => void;
  onToggleLock: (locked: boolean) => void;
  onContextMenu: (x: number, y: number) => void;
}) {
  // image_path is the clean, full-resolution source used by downloads and
  // training. Crops are optional after a rebuild.
  const locked = detection.locked ?? false;

  // The lock toggle is a sibling of the card button, not a child: a button
  // inside a button is invalid DOM. The wrapper carries `group` + the square so
  // both sit in the same box and hover still drives the overlays.
  return (
    <div className="group relative aspect-square">
      <button
        onClick={(e) => onToggle(e.shiftKey)}
        onContextMenu={(e) => {
          e.preventDefault();
          onContextMenu(e.clientX, e.clientY);
        }}
        className={`absolute inset-0 overflow-hidden rounded-lg border-2 transition-colors ${
          isSelected
            ? "border-primary ring-2 ring-primary/20"
            : locked
              ? "border-amber-400/80"
              : "border-transparent hover:border-muted-foreground/30"
        }`}
      >
        <DetectionThumbnail
          detection={detection}
          alt={`偵測 #${detection.track_id}`}
          className="h-full w-full object-cover"
          loading="lazy"
        />

        <div className="absolute left-1.5 top-1.5">
          {isSelected ? (
            <CheckSquare className="h-5 w-5 text-primary drop-shadow" />
          ) : (
            <Square className="h-5 w-5 text-white/70 opacity-0 drop-shadow group-hover:opacity-100" />
          )}
        </div>

        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent px-2 pb-1.5 pt-4">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-white">#{detection.track_id}</span>
            <span className="text-[10px] text-white/80">{(detection.confidence * 100).toFixed(0)}%</span>
          </div>
          <p className="text-[9px] text-white/60">{format(new Date(detection.timestamp), "MM/dd HH:mm")}</p>
        </div>
      </button>

      {/* Locked cards stay marked at rest — the whole point is that you can see
          which records are protected without hovering or selecting. */}
      <button
        onClick={() => onToggleLock(!locked)}
        disabled={lockBusy}
        title={
          locked
            ? "已鎖定：刪除會保留此筆。點選以解除鎖定"
            : "鎖定此筆，使其無法被刪除（刪除是硬刪，無法復原）"
        }
        aria-label={locked ? `解除鎖定偵測 #${detection.id}` : `鎖定偵測 #${detection.id}`}
        aria-pressed={locked}
        className={`absolute right-1.5 top-1.5 z-10 rounded p-1 shadow-sm transition-opacity disabled:opacity-40 ${
          locked
            ? "bg-amber-400 text-white opacity-100"
            : "bg-black/45 text-white opacity-0 focus-visible:opacity-100 group-hover:opacity-100"
        }`}
      >
        {lockBusy ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : locked ? (
          <Lock className="h-3.5 w-3.5" />
        ) : (
          <Unlock className="h-3.5 w-3.5" />
        )}
      </button>
    </div>
  );
}
