"use client";

import type { Detection } from "@/lib/api";
import { SnapshotImg } from "@/components/snapshot-img";

/**
 * Both surfaces use the crop as their primary card image so the monkey stays
 * visible at thumbnail size. The full raw frame remains available in the
 * review/download flow, and is only a fallback when a legacy row has no crop.
 */
export function DetectionThumbnail({
  detection,
  alt,
  className,
  loading,
}: {
  detection: Pick<Detection, "image_path" | "crop_path" | "annotated_path">;
  alt: string;
  className?: string;
  loading?: "eager" | "lazy";
}) {
  const filename = detection.crop_path || detection.image_path || detection.annotated_path;

  if (!filename) {
    return (
      <div
        className={`flex items-center justify-center bg-muted text-xs text-muted-foreground ${className ?? ""}`}
      >
        無圖片
      </div>
    );
  }

  return (
    <SnapshotImg
      filename={filename}
      fallbackFilename={detection.image_path || detection.annotated_path || undefined}
      alt={alt}
      className={className}
      loading={loading}
    />
  );
}
