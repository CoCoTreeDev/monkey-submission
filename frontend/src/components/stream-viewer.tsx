"use client";

/* eslint-disable @next/next/no-img-element */

import { useState, useEffect, useCallback } from "react";
import { VideoOff, Maximize2, X } from "lucide-react";
import { streamUrl } from "@/lib/api";

interface StreamViewerProps {
  cameraId: string;
  cameraName?: string;
  isAlive?: boolean;
}

export function StreamViewer({ cameraId, cameraName, isAlive = true }: StreamViewerProps) {
  const [hasError, setHasError] = useState(false);
  const [modal, setModal] = useState(false);
  const [nonce, setNonce] = useState(0);

  // The backend MJPEG endpoint keeps replaying the camera's last frame even
  // when the RTSP source stalls, so a frozen/idle camera still SHOWS its
  // picture — the stream is always rendered; only a real transport error
  // falls back (and auto-retries). isAlive only tints the status badge.
  const base = streamUrl(cameraId);
  const url = `${base}${base.includes("?") ? "&" : "?"}r=${nonce}`;
  const showFallback = hasError;
  const statusColor = isAlive ? "bg-green-500" : "bg-gray-300";
  // 客戶回饋：夜間沒有光線拍不到清晰影像，畫面靜止是「無光源」而非故障
  const statusLabel = isAlive ? "運作中" : "無光源";

  const handleStreamError = useCallback(() => setHasError(true), []);

  useEffect(() => {
    if (!hasError) return;
    const t = setTimeout(() => {
      setNonce((n) => n + 1);
      setHasError(false);
    }, 8000);
    return () => clearTimeout(t);
  }, [hasError]);

  const closeModal = useCallback(() => setModal(false), []);
  useEffect(() => {
    if (!modal) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeModal();
    };
    document.addEventListener("keydown", handleKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.style.overflow = "";
    };
  }, [modal, closeModal]);

  return (
    <>
      {/* Fullscreen modal */}
      {modal && (
        <div
          className="fixed inset-0 z-50 flex flex-col bg-black"
          onClick={closeModal}
        >
          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-2.5">
              <div className={`h-2 w-2 rounded-full ${statusColor}`} />
              <span className="text-sm font-medium text-white">
                {cameraName || cameraId}
              </span>
              <span className="text-xs text-white/60">{statusLabel}</span>
            </div>
            <button
              onClick={closeModal}
              className="rounded-full bg-white/10 p-2 text-white/80 transition hover:bg-white/20 hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
          <div className="flex flex-1 items-center justify-center p-4" onClick={(e) => e.stopPropagation()}>
            <img
              src={url}
              alt={`來自 ${cameraName || cameraId} 的即時串流`}
              className="max-h-full max-w-full object-contain"
              onError={handleStreamError}
            />
          </div>
        </div>
      )}

      <div className="overflow-hidden rounded-lg border bg-white">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-4 py-2.5">
          <div className="flex items-center gap-2.5">
            <div className={`h-2 w-2 rounded-full ${statusColor}`} />
            <span className="text-sm font-medium">{cameraName || cameraId}</span>
            <span
              className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${
                isAlive
                  ? "bg-green-50 text-green-700"
                  : "bg-gray-100 text-gray-500"
              }`}
            >
              {statusLabel}
            </span>
            <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
              {cameraId}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              className="flex h-7 w-7 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-30"
              onClick={() => setModal(true)}
              disabled={showFallback}
              title="全螢幕檢視"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Stream */}
        {showFallback ? (
          <div className="flex flex-col items-center justify-center gap-3 bg-neutral-50 py-20">
            <VideoOff className="h-8 w-8 text-muted-foreground/40" />
            <p className="text-sm text-muted-foreground">畫面載入中，請稍候…</p>
            <button
              className="rounded border px-3 py-1.5 text-xs font-medium text-primary hover:bg-muted"
              onClick={() => {
                setNonce((n) => n + 1);
                setHasError(false);
              }}
            >
              重新整理
            </button>
          </div>
        ) : (
          <div className="bg-neutral-900">
            <img
              src={url}
              alt={`來自 ${cameraId} 的即時串流`}
              className="w-full object-contain"
              style={{ minHeight: 300 }}
              onError={handleStreamError}
            />
          </div>
        )}
      </div>
    </>
  );
}
