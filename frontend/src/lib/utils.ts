import type { DuplicateNote } from "@/lib/api";
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function duplicateUploadText(duplicates: DuplicateNote[]): string {
  if (duplicates.length === 0) return "";
  const preview = duplicates
    .slice(0, 3)
    .map((d) => `${d.name} ≈ ${d.similar_to}（距離 ${d.distance}）`)
    .join("、");
  const more = duplicates.length > 3 ? ` 等 ${duplicates.length} 張` : "";
  return `；${duplicates.length} 張與現有影像高度相似${more}，主檔照收，訓練集會在去重時折疊${preview ? `：${preview}` : ""}`;
}
