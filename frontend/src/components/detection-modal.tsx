"use client";

import { useEffect } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

import { DetectionReview } from "@/components/detection-review";
import type { Detection } from "@/lib/api";

interface DetectionModalProps {
  open: boolean;
  onClose: () => void;
  /** The list the modal navigates within — same array the table is showing. */
  detections: Detection[];
  selectedIdx: number;
  onPrev: () => void;
  onNext: () => void;
  total?: number;
  page?: number;
  totalPages?: number;
  canDownload?: boolean;
  showConfidence?: boolean;
  isAdmin?: boolean;
  /** Forwarded to the review pane; omitted = no delete action rendered. */
  onDelete?: (detection: Detection) => void;
  deleting?: boolean;
  deleteError?: string | null;
}

/**
 * Reuses the 偵測檢閱 pane inside a modal, so 檢視 stays on the page.
 *
 * Opening a new tab meant the snapshot's Content-Type decided whether the
 * browser rendered or downloaded the file, and it threw away everything the
 * review pane already does well: trail/annotated toggle, crop preview, raw
 * frame download, and ← / → navigation across the whole result set. Wrapping
 * the existing component is strictly better than a bespoke image viewer —
 * one implementation, one set of behaviours to maintain.
 */
export function DetectionModal({
  open,
  onClose,
  detections,
  selectedIdx,
  onPrev,
  onNext,
  total,
  page,
  totalPages,
  canDownload,
  showConfidence,
  isAdmin,
  onDelete,
  deleting,
  deleteError,
}: DetectionModalProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      // The review pane owns the arrows; the modal only owns Escape.
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);

  if (!open) return null;

  // Page-level spacing (for example ImageManagement's space-y-5) must not
  // apply margins to the fixed backdrop and leave part of the viewport exposed.
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm sm:p-8"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="偵測檢閱"
    >
      <div
        className="relative flex h-[88vh] w-full max-w-5xl flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          title="關閉（Esc）"
          className="absolute -top-10 right-0 rounded-full bg-black/50 p-2 text-white/80 transition hover:bg-black/70 hover:text-white"
        >
          <X className="h-5 w-5" />
        </button>
        <DetectionReview
          detections={detections}
          isLoading={false}
          total={total}
          page={page}
          totalPages={totalPages}
          selectedIdx={selectedIdx}
          onPrev={onPrev}
          onNext={onNext}
          canDownload={canDownload}
          showConfidence={showConfidence}
          isAdmin={isAdmin}
          onDelete={onDelete}
          deleting={deleting}
          deleteError={deleteError}
        />
      </div>
    </div>,
    document.body,
  );
}
