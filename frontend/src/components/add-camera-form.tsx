"use client";

import { useState } from "react";
import { Plus } from "lucide-react";
import { addCamera } from "@/lib/api";

interface AddCameraFormProps {
  onAdded: () => void;
}

export function AddCameraForm({ onAdded }: AddCameraFormProps) {
  const [cameraId, setCameraId] = useState("");
  const [rtspUrl, setRtspUrl] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cameraId.trim() || !rtspUrl.trim()) return;

    setLoading(true);
    setError("");

    try {
      await addCamera({
        camera_id: cameraId.trim(),
        rtsp_url: rtspUrl.trim(),
        name: name.trim(),
      });
      setCameraId("");
      setRtspUrl("");
      setName("");
      onAdded();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "新增攝影機失敗");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-lg border bg-white">
      <div className="border-b px-5 py-3">
        <h3 className="text-sm font-semibold">新增攝影機</h3>
      </div>
      <div className="p-5">
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                攝影機 ID
              </label>
              <input
                type="text"
                placeholder="cam1"
                value={cameraId}
                onChange={(e) => setCameraId(e.target.value)}
                className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                required
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                RTSP 網址
              </label>
              <input
                type="text"
                placeholder="rtsp://192.168.1.1/stream"
                value={rtspUrl}
                onChange={(e) => setRtspUrl(e.target.value)}
                className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                required
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                顯示名稱
              </label>
              <input
                type="text"
                placeholder="選填"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
              />
            </div>
          </div>
          {error && (
            <p className="text-xs text-red-600">{error}</p>
          )}
          <button
            type="submit"
            disabled={loading}
            className="flex w-fit items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-40"
          >
            <Plus className="h-3.5 w-3.5" />
            {loading ? "新增中..." : "新增攝影機"}
          </button>
        </form>
      </div>
    </div>
  );
}
