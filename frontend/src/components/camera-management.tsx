"use client";

import { useState } from "react";
import { MessageBanner, useFlashMessage } from "@/components/status-message";
import useSWR from "swr";
import { Loader2, Pencil, Trash2, Check, X } from "lucide-react";
import {
  API_BASE,
  fetcher,
  cameraLabel,
  cameraHasDistinctName,
  updateCameraAdmin,
  deleteCameraAdmin,
  type Camera,
} from "@/lib/api";

/** 攝影機管理 — edit name/RTSP + remove (Operations tab, admin). */
export function CameraManagement() {
  const { data: cameras, mutate: mutateCameras } = useSWR<Camera[]>(
    `${API_BASE}/api/cameras`,
    fetcher,
    { refreshInterval: 10000 }
  );

  const [editingCameraId, setEditingCameraId] = useState<string | null>(null);
  const [editCamName, setEditCamName] = useState("");
  const [editCamRtsp, setEditCamRtsp] = useState("");
  const [savingCameraId, setSavingCameraId] = useState<string | null>(null);
  const [deletingCameraId, setDeletingCameraId] = useState<string | null>(null);
  const [message, flash] = useFlashMessage();

  const startCameraEdit = (cam: Camera) => {
    setEditingCameraId(cam.camera_id);
    setEditCamName(cam.name || "");
    setEditCamRtsp(cam.rtsp_url);
  };

  const handleSaveCamera = async () => {
    if (!editingCameraId || !editCamRtsp.trim()) return;
    setSavingCameraId(editingCameraId);
    try {
      await updateCameraAdmin(editingCameraId, {
        name: editCamName.trim(),
        rtsp_url: editCamRtsp.trim(),
      });
      setEditingCameraId(null);
      flash("success", "攝影機設定已更新");
      mutateCameras();
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "更新失敗");
    } finally {
      setSavingCameraId(null);
    }
  };

  const handleDeleteCamera = async (cam: Camera) => {
    let preserved = 0;
    try {
      const list = await fetcher<{ total: number }>(
        `${API_BASE}/api/detections?camera_id=${encodeURIComponent(cam.camera_id)}&limit=1`
      );
      preserved = list.total;
    } catch {
      /* proceed with confirm even if count fetch fails */
    }

    const label = cameraLabel(cam);
    if (
      !window.confirm(
        `確定要移除攝影機「${label}」？\n\n串流與推論將停止，但既有 ${preserved.toLocaleString()} 筆偵測紀錄會保留。`
      )
    ) {
      return;
    }

    setDeletingCameraId(cam.camera_id);
    try {
      const result = await deleteCameraAdmin(cam.camera_id);
      flash(
        "success",
        `已移除「${result.camera_id}」，保留 ${result.detections_preserved.toLocaleString()} 筆偵測紀錄`
      );
      if (editingCameraId === cam.camera_id) setEditingCameraId(null);
      mutateCameras();
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "移除失敗");
    } finally {
      setDeletingCameraId(null);
    }
  };

  return (
    <div className="rounded-lg border bg-white">
      <div className="border-b px-5 py-3">
        <h3 className="text-sm font-semibold">攝影機管理</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          移除攝影機只停止串流與推論，既有偵測紀錄會保留。
        </p>
      </div>
      <MessageBanner message={message} className="mx-5 mt-3" />
      <div className="divide-y">
        {!cameras?.length ? (
          <p className="px-5 py-4 text-xs text-muted-foreground">尚未註冊攝影機</p>
        ) : (
          cameras.map((cam) => {
            const isEditing = editingCameraId === cam.camera_id;
            const isSaving = savingCameraId === cam.camera_id;
            const isDeleting = deletingCameraId === cam.camera_id;
            return (
              <div key={cam.camera_id} className="px-5 py-3">
                {isEditing ? (
                  <div className="space-y-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
                        {cam.camera_id}
                      </span>
                      <div
                        className={`h-2 w-2 rounded-full ${cam.is_alive ? "bg-green-500" : "bg-gray-300"}`}
                        title={cam.is_alive ? "運作中" : "無光源"}
                      />
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div className="space-y-1">
                        <label className="text-xs font-medium text-muted-foreground">顯示名稱</label>
                        <input
                          value={editCamName}
                          onChange={(e) => setEditCamName(e.target.value)}
                          placeholder={cam.camera_id}
                          className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                        />
                      </div>
                      <div className="space-y-1 sm:col-span-2">
                        <label className="text-xs font-medium text-muted-foreground">RTSP 網址</label>
                        <input
                          value={editCamRtsp}
                          onChange={(e) => setEditCamRtsp(e.target.value)}
                          className="w-full rounded-md border bg-background px-3 py-2 font-mono text-xs outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                          required
                        />
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleSaveCamera}
                        disabled={isSaving || !editCamRtsp.trim()}
                        className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-40"
                      >
                        {isSaving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
                        儲存
                      </button>
                      <button
                        type="button"
                        onClick={() => setEditingCameraId(null)}
                        disabled={isSaving}
                        className="inline-flex items-center gap-1 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-40"
                      >
                        <X className="h-3.5 w-3.5" />
                        取消
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <div
                          className={`h-2 w-2 shrink-0 rounded-full ${cam.is_alive ? "bg-green-500" : "bg-gray-300"}`}
                        />
                        <span className="text-sm font-medium">{cameraLabel(cam)}</span>
                        {cam.poc && (
                          <span className="rounded bg-amber-50 px-1.5 py-0.5 text-[11px] font-medium text-amber-700">
                            POC・管理員限定
                          </span>
                        )}
                        {cameraHasDistinctName(cam) && (
                          <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
                            {cam.camera_id}
                          </span>
                        )}
                        {!cam.is_alive && (
                          <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[11px] text-gray-500">無光源</span>
                        )}
                      </div>
                      <p className="truncate font-mono text-xs text-muted-foreground">{cam.rtsp_url}</p>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      <button
                        type="button"
                        onClick={() => startCameraEdit(cam)}
                        disabled={isDeleting}
                        title="編輯攝影機"
                        className="inline-flex h-8 w-8 items-center justify-center rounded-md border text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-40"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDeleteCamera(cam)}
                        disabled={isDeleting}
                        title="移除攝影機"
                        className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-red-200 text-red-600 hover:bg-red-50 disabled:opacity-40"
                      >
                        {isDeleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
