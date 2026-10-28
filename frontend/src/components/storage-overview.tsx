"use client";

import { useState } from "react";
import { MessageBanner, useFlashMessage } from "@/components/status-message";
import useSWR, { mutate as globalMutate } from "swr";
import { HardDrive, FileImage, Database, Trash2, Loader2 } from "lucide-react";
import { API_BASE, fetcher, clearSnapshots, type StorageStats } from "@/lib/api";

/** 儲存空間概覽 — file count / size / detection count + 清除全部 (admin, Overview tab). */
export function StorageOverview() {
  const {
    data: stats,
    isLoading: statsLoading,
    mutate: mutateStats,
  } = useSWR<StorageStats>(`${API_BASE}/api/admin/storage-stats`, fetcher, {
    refreshInterval: 30000,
  });
  const [clearing, setClearing] = useState(false);
  const [message, flash] = useFlashMessage();

  const handleClearAll = async () => {
    // 已鎖定的紀錄會被保留 — 這裡先講明，否則刪完檔案數不會歸零，看起來像壞掉。
    if (
      !window.confirm(
        "確定要刪除所有快照檔案與偵測紀錄？此操作無法復原。\n（已鎖定的紀錄及其檔案會保留）"
      )
    )
      return;
    setClearing(true);
    try {
      const result = await clearSnapshots();
      const kept = result.kept_locked ?? 0;
      flash(
        "success",
        `已刪除 ${result.deleted_files} 個檔案、${result.deleted_detections} 筆偵測紀錄` +
          (kept > 0 ? `；${kept} 筆已鎖定，連同檔案一併保留` : "")
      );
      mutateStats();
      // Refresh detection lists/stats shown on other tabs.
      globalMutate((k) => typeof k === "string" && k.includes("/api/detections"));
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "清除失敗");
    } finally {
      setClearing(false);
    }
  };

  const cell = (icon: React.ReactNode, label: string, value: React.ReactNode) => (
    <div className="flex items-center gap-3 rounded-lg border px-4 py-3">
      {icon}
      <div>
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-lg font-semibold">
          {statsLoading ? (
            <span className="inline-block h-6 w-12 animate-pulse rounded bg-muted" />
          ) : (
            value
          )}
        </p>
      </div>
    </div>
  );

  return (
    <div className="rounded-lg border bg-white">
      <div className="flex items-center justify-between border-b px-5 py-3">
        <h3 className="text-sm font-semibold">儲存空間概覽</h3>
        <button
          onClick={handleClearAll}
          disabled={clearing || !stats || stats.total_files === 0}
          className="flex items-center gap-1.5 rounded-md border border-red-200 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-40"
        >
          {clearing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
          清除全部
        </button>
      </div>
      <MessageBanner message={message} className="mx-5 mt-3" />
      <div className="grid gap-4 p-5 sm:grid-cols-3">
        {cell(<FileImage className="h-5 w-5 text-blue-500" />, "檔案數量", (stats?.total_files ?? 0).toLocaleString())}
        {cell(<HardDrive className="h-5 w-5 text-violet-500" />, "儲存空間", stats?.total_size_human ?? "0 B")}
        {cell(<Database className="h-5 w-5 text-green-500" />, "偵測紀錄數", stats?.total_detections ?? 0)}
      </div>
    </div>
  );
}
