"use client";

/* eslint-disable @next/next/no-img-element */

import { useState, useEffect, useRef, useSyncExternalStore } from "react";
import { snapshotFallbackUrl, snapshotUrl } from "@/lib/api";

/* sessionStorage does not change under us mid-render, so there is nothing to
   subscribe to — only a server answer and a client answer. */
const _neverChanges = () => () => {};
const _onClient = () => true;
const _onServer = () => false;

interface SnapshotImgProps
  extends Omit<React.ImgHTMLAttributes<HTMLImageElement>, "src"> {
  filename: string;
  /** Optional second asset for rows whose primary media was reclaimed. */
  fallbackFilename?: string;
}

/**
 * Image component that loads from the backend snapshot API.
 *
 * The `<img>` is keyed on `filename` so that switching to a different detection
 * REMOUNTS the node — the previous picture clears immediately instead of
 * lingering until the new (potentially large trail GIF) finishes downloading.
 * Without this, a reused <img> keeps painting stale pixels and the review pane
 * appears to "not jump" for seconds on a slow backend. Keying on filename (not
 * src) means the 5s SWR poll and the error→fallback src swap do NOT remount.
 */
export function SnapshotImg({
  filename,
  fallbackFilename,
  alt,
  onError,
  onLoad,
  className,
  ...props
}: SnapshotImgProps) {
  const sourceKey = `${filename}\u0000${fallbackFilename ?? ""}`;
  const candidates = [
    { filename, static: false },
    ...(fallbackFilename && fallbackFilename !== filename
      ? [{ filename: fallbackFilename, static: false }]
      : []),
    { filename, static: true },
    ...(fallbackFilename && fallbackFilename !== filename
      ? [{ filename: fallbackFilename, static: true }]
      : []),
  ];
  const [loadState, setLoadState] = useState({ owner: sourceKey, loaded: false });
  const [sourceState, setSourceState] = useState({
    owner: sourceKey,
    index: 0,
    failed: false,
  });
  const imgRef = useRef<HTMLImageElement | null>(null);

  /* The snapshot URL carries the session token, and the token lives in
     sessionStorage — which does not exist while this renders on the server.
     Painting a src there hands the browser a URL with no token, it starts the
     request during hydration, the API answers 401, and every thumbnail on the
     page turns into a broken box. This is the one thing that legitimately
     differs between the server and client renders, which is what
     useSyncExternalStore's two snapshots are for. */
  const tokenReadable = useSyncExternalStore(_neverChanges, _onClient, _onServer);

  const current = sourceState.owner === sourceKey
    ? sourceState
    : { owner: sourceKey, index: 0, failed: false };
  const candidate = candidates[current.index] ?? candidates[0];
  const loaded = loadState.owner === sourceKey && loadState.loaded;
  // Props can change while a card component remains mounted during SWR
  // refresh. Treat the new prop as its own source synchronously, avoiding a
  // blanking effect that would cause an extra render for every thumbnail.
  const failed = current.failed;

  // Abandon any in-flight download when the source changes or the component
  // unmounts — clearing src makes the browser abort the request, so stepping
  // through detections can't leak multi-MB trail-GIF loads across records.
  // (The <img> is keyed on filename, so each filename gets its own DOM node;
  // capture it per-filename so cleanup targets the node being replaced.)
  //
  // Deferred, and only for a node that has actually left the document: React
  // re-runs effects on a STILL-MOUNTED node in development, and clearing a live
  // node's src there blanks it permanently, because React will not re-set a
  // prop that did not change. That is what turned every thumbnail into an empty
  // box. By the next microtask a genuinely discarded node is detached and one
  // that was only re-run is not, which tells the two apart without depending on
  // the order React tears effects down in.
  useEffect(() => {
    const img = imgRef.current;
    return () => {
      queueMicrotask(() => {
        if (img && !img.isConnected && !img.complete) img.src = "";
      });
    };
  }, [sourceKey]);

  const handleLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    setLoadState({ owner: sourceKey, loaded: true });
    setSourceState({ owner: sourceKey, index: current.index, failed: false });
    onLoad?.(e);
  };

  const handleError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    if (current.index < candidates.length - 1) {
      setSourceState({ owner: sourceKey, index: current.index + 1, failed: false });
      setLoadState({ owner: sourceKey, loaded: false });
      return;
    }
    setSourceState({ owner: sourceKey, index: current.index, failed: true });
    onError?.(e);
  };

  if (failed) {
    return (
      <div className={`flex items-center justify-center bg-muted text-xs text-muted-foreground ${className ?? ""}`}>
        無圖片
      </div>
    );
  }

  return (
    <img
      key={filename}
      ref={imgRef}
      src={candidate.static ? snapshotFallbackUrl(candidate.filename) : tokenReadable ? snapshotUrl(candidate.filename) : undefined}
      alt={alt ?? ""}
      onError={handleError}
      onLoad={handleLoad}
      className={`${className ?? ""} ${loaded || failed ? "opacity-100" : "opacity-0"} transition-opacity duration-200`.trim()}
      {...props}
    />
  );
}
