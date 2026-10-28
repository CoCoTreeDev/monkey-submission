"use client";

import { useState } from "react";

import { format } from "date-fns";
import { ImageIcon } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Detection } from "@/lib/api";
import { DetectionModal } from "@/components/detection-modal";

interface DetectionLogProps {
  detections: Detection[];
  isLoading: boolean;
  // Hidden for non-admins: the 信心度 column is not rendered at all.
  showConfidence?: boolean;
  /** Passed through to the review modal — hides download actions when off. */
  canDownload?: boolean;
}

function confBadge(conf: number) {
  const label = `${(conf * 100).toFixed(0)}%`;
  if (conf >= 0.85)
    return (
      <span className="rounded bg-red-50 px-1.5 py-0.5 text-[11px] font-medium text-red-600">
        {label}
      </span>
    );
  if (conf >= 0.65)
    return (
      <span className="rounded bg-orange-50 px-1.5 py-0.5 text-[11px] font-medium text-orange-600">
        {label}
      </span>
    );
  return (
    <span className="rounded bg-muted px-1.5 py-0.5 text-[11px] font-medium text-muted-foreground">
      {label}
    </span>
  );
}

export function DetectionLog({
  detections,
  isLoading,
  showConfidence = true,
  canDownload,
}: DetectionLogProps) {
  const [modalIdx, setModalIdx] = useState<number | null>(null);
  return (
    <>
      <DetectionModal
        open={modalIdx !== null}
        onClose={() => setModalIdx(null)}
        detections={detections}
        selectedIdx={modalIdx ?? 0}
        onPrev={() => setModalIdx((i) => Math.max(0, (i ?? 0) - 1))}
        onNext={() =>
          setModalIdx((i) => Math.min(detections.length - 1, (i ?? 0) + 1))
        }
        canDownload={canDownload}
        showConfidence={showConfidence}
      />
    <div className="rounded-lg border bg-white">
      <div className="border-b px-5 py-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">偵測紀錄</h3>
          {detections.length > 0 && (
            <span className="text-xs text-muted-foreground">{detections.length} 筆紀錄</span>
          )}
        </div>
      </div>

      <div className="px-4 pb-4">
        <ScrollArea className="h-[340px]">
          <table className="w-full border-collapse text-sm" style={{ tableLayout: "fixed" }}>
            {showConfidence ? (
              <colgroup>
                <col style={{ width: "35%" }} />
                <col style={{ width: "16%" }} />
                <col style={{ width: "18%" }} />
                <col style={{ width: "16%" }} />
                <col style={{ width: "15%" }} />
              </colgroup>
            ) : (
              <colgroup>
                <col style={{ width: "40%" }} />
                <col style={{ width: "18%" }} />
                <col style={{ width: "22%" }} />
                <col style={{ width: "20%" }} />
              </colgroup>
            )}
            <thead>
              <tr className="border-b text-left">
                <th className="py-2.5 text-xs font-medium text-muted-foreground">
                  時間
                </th>
                <th className="py-2.5 text-xs font-medium text-muted-foreground">
                  追蹤
                </th>
                <th className="py-2.5 text-xs font-medium text-muted-foreground">
                  攝影機
                </th>
                {showConfidence && (
                  <th className="py-2.5 text-xs font-medium text-muted-foreground">
                    信心度
                  </th>
                )}
                <th className="py-2.5 text-right text-xs font-medium text-muted-foreground">
                  快照
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td colSpan={showConfidence ? 5 : 4} className="py-2.5">
                      <div className="h-4 w-60 animate-pulse rounded bg-muted" />
                    </td>
                  </tr>
                ))
              ) : detections.length === 0 ? (
                <tr>
                  <td
                    colSpan={showConfidence ? 5 : 4}
                    className="py-16 text-center text-muted-foreground"
                  >
                    尚無偵測紀錄。
                  </td>
                </tr>
              ) : (
                detections.map((det, idx) => (
                  <tr
                    key={det.id}
                    className="border-b last:border-0 hover:bg-muted/30"
                  >
                    <td className="py-2 font-mono text-xs text-muted-foreground whitespace-nowrap">
                      {format(new Date(det.timestamp), "MM-dd HH:mm:ss")}
                    </td>
                    <td className="py-2">
                      {det.track_id >= 0 ? (
                        <span className="rounded bg-blue-50 px-1.5 py-0.5 text-[11px] font-medium text-blue-600">
                          #{det.track_id}
                        </span>
                      ) : (
                        <span className="text-xs text-muted-foreground/40">
                          --
                        </span>
                      )}
                    </td>
                    <td className="py-2">
                      <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
                        {det.camera_id}
                      </span>
                    </td>
                    {showConfidence && (
                      <td className="py-2">{confBadge(det.confidence)}</td>
                    )}
                    <td className="py-2 text-right">
                      {det.image_path ? (
                        <button
                          className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                          onClick={(e) => {
                            e.stopPropagation();
                            // In-page lightbox rather than a new tab: navigating
                            // away made the snapshot's Content-Type decide
                            // whether the browser rendered or downloaded it, and
                            // the user only ever wanted a closer look.
                            setModalIdx(idx);
                          }}
                        >
                          <ImageIcon className="h-3 w-3" />
                          檢視
                        </button>
                      ) : (
                        <span className="text-xs text-muted-foreground/40">
                          --
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </ScrollArea>
      </div>
    </div>
    </>
  );
}
