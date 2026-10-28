"use client";

import { useState } from "react";
import useSWR from "swr";
import { Plus, Send, Pencil, Trash2, Check, X, Loader2 } from "lucide-react";
import {
  API_BASE,
  fetcher,
  createWebhook,
  updateWebhook,
  deleteWebhook,
  testWebhook,
  type Webhook,
} from "@/lib/api";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";

export function WebhookSettings() {
  const {
    data: webhooks,
    isLoading,
    mutate,
  } = useSWR<Webhook[]>(`${API_BASE}/api/webhooks`, fetcher);

  // ── Add form state ──────────────────────────────────────────────────
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState("");

  // ── Inline edit state ───────────────────────────────────────────────
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState("");
  const [editUrl, setEditUrl] = useState("");
  const [saving, setSaving] = useState(false);

  // ── Per-row action state ────────────────────────────────────────────
  const [testingId, setTestingId] = useState<number | null>(null);
  const [togglingId, setTogglingId] = useState<number | null>(null);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const clearMessage = () => setTimeout(() => setMessage(null), 3000);

  // ── Handlers ────────────────────────────────────────────────────────

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    setAdding(true);
    setAddError("");
    try {
      await createWebhook({ name: name.trim(), url: url.trim() });
      setName("");
      setUrl("");
      mutate();
    } catch (err: unknown) {
      setAddError(err instanceof Error ? err.message : "新增 Webhook 失敗");
    } finally {
      setAdding(false);
    }
  };

  const handleToggle = async (wh: Webhook) => {
    setTogglingId(wh.id);
    try {
      await updateWebhook(wh.id, { enabled: !wh.enabled });
      mutate();
    } catch {
      setMessage({ type: "error", text: "切換 Webhook 狀態失敗" });
      clearMessage();
    } finally {
      setTogglingId(null);
    }
  };

  const startEdit = (wh: Webhook) => {
    setEditingId(wh.id);
    setEditName(wh.name);
    setEditUrl(wh.url);
  };

  const cancelEdit = () => {
    setEditingId(null);
  };

  const handleSave = async () => {
    if (editingId === null || !editUrl.trim()) return;
    setSaving(true);
    try {
      await updateWebhook(editingId, {
        name: editName.trim(),
        url: editUrl.trim(),
      });
      setEditingId(null);
      mutate();
    } catch (err: unknown) {
      setMessage({
        type: "error",
        text: err instanceof Error ? err.message : "儲存失敗",
      });
      clearMessage();
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async (wh: Webhook) => {
    setTestingId(wh.id);
    setMessage(null);
    try {
      const result = await testWebhook(wh.id);
      setMessage({
        type: "success",
        text: `測試已傳送至「${wh.name || wh.url}」(HTTP ${result.http_status})`,
      });
    } catch (err: unknown) {
      setMessage({
        type: "error",
        text: err instanceof Error ? err.message : "測試失敗",
      });
    } finally {
      setTestingId(null);
      clearMessage();
    }
  };

  const handleDelete = async (wh: Webhook) => {
    if (!window.confirm(`確定要刪除 Webhook「${wh.name || wh.url}」？`)) return;
    try {
      await deleteWebhook(wh.id);
      mutate();
    } catch {
      setMessage({ type: "error", text: "刪除 Webhook 失敗" });
      clearMessage();
    }
  };

  // ── Render ──────────────────────────────────────────────────────────

  return (
    <div className="space-y-5">
      {/* Add Webhook */}
      <div className="rounded-lg border bg-white">
        <div className="border-b px-5 py-3">
          <h3 className="text-sm font-semibold">新增 Webhook</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            新增 Discord、Slack 或自訂 Webhook 端點
          </p>
        </div>
        <div className="p-5">
          <form onSubmit={handleAdd} className="flex flex-col gap-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  名稱
                </label>
                <input
                  type="text"
                  placeholder="例如：Discord 警報"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  Webhook 網址
                </label>
                <input
                  type="url"
                  placeholder="https://discord.com/api/webhooks/..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-2 font-mono text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                  required
                />
              </div>
            </div>
            {addError && <p className="text-xs text-red-600">{addError}</p>}
            <button
              type="submit"
              disabled={adding}
              className="flex w-fit items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-40"
            >
              <Plus className="h-3.5 w-3.5" />
              {adding ? "新增中..." : "新增 Webhook"}
            </button>
          </form>
        </div>
      </div>

      {/* Status message */}
      {message && (
        <div
          className={`flex items-center gap-2 rounded-md px-3 py-2 text-xs ${
            message.type === "success"
              ? "bg-green-50 text-green-700"
              : "bg-red-50 text-red-700"
          }`}
        >
          {message.type === "success" ? (
            <Check className="h-3.5 w-3.5" />
          ) : (
            <X className="h-3.5 w-3.5" />
          )}
          {message.text}
        </div>
      )}

      {/* Webhooks Table */}
      <div className="rounded-lg border bg-white">
        <div className="border-b px-5 py-3">
          <h3 className="text-sm font-semibold">已設定的 Webhook</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            偵測警報會傳送至所有已啟用的 Webhook
          </p>
        </div>

        {isLoading ? (
          <div className="px-5 py-8 text-center text-sm text-muted-foreground">
            載入 Webhook 中...
          </div>
        ) : !webhooks || webhooks.length === 0 ? (
          <div className="px-5 py-8 text-center text-sm text-muted-foreground">
            尚未設定任何 Webhook，請在上方新增。
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-16 text-center">狀態</TableHead>
                <TableHead>名稱</TableHead>
                <TableHead>網址</TableHead>
                <TableHead className="w-36 text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {webhooks.map((wh) => (
                <TableRow key={wh.id}>
                  {/* Status toggle */}
                  <TableCell>
                    <button
                      onClick={() => handleToggle(wh)}
                      disabled={togglingId === wh.id}
                      className="flex w-full items-center justify-center"
                      title={
                        !wh.enabled
                          ? "已停用 — 點擊啟用"
                          : wh.last_error
                            ? `錯誤：${wh.last_error}`
                            : "運作中 — 點擊停用"
                      }
                    >
                      {togglingId === wh.id ? (
                        <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                      ) : (
                        <div
                          className={`h-2.5 w-2.5 rounded-full ${
                            !wh.enabled
                              ? "bg-gray-300"
                              : wh.last_error
                                ? "bg-red-500"
                                : "bg-green-500"
                          }`}
                        />
                      )}
                    </button>
                  </TableCell>

                  {/* Name */}
                  <TableCell>
                    {editingId === wh.id ? (
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        className="w-full rounded border bg-background px-2 py-1 text-sm outline-none focus:border-primary"
                      />
                    ) : (
                      <span className="text-sm font-medium">
                        {wh.name || (
                          <span className="text-muted-foreground italic">
                            未命名
                          </span>
                        )}
                      </span>
                    )}
                  </TableCell>

                  {/* URL */}
                  <TableCell>
                    {editingId === wh.id ? (
                      <input
                        type="url"
                        value={editUrl}
                        onChange={(e) => setEditUrl(e.target.value)}
                        className="w-full rounded border bg-background px-2 py-1 font-mono text-xs outline-none focus:border-primary"
                        required
                      />
                    ) : (
                      <span className="max-w-xs truncate font-mono text-xs text-muted-foreground block">
                        {wh.url}
                      </span>
                    )}
                  </TableCell>

                  {/* Actions */}
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-1">
                      {editingId === wh.id ? (
                        <>
                          <button
                            onClick={handleSave}
                            disabled={saving}
                            className="rounded p-1.5 text-green-600 hover:bg-green-50"
                            title="儲存"
                          >
                            {saving ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Check className="h-3.5 w-3.5" />
                            )}
                          </button>
                          <button
                            onClick={cancelEdit}
                            className="rounded p-1.5 text-muted-foreground hover:bg-muted"
                            title="取消"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            onClick={() => startEdit(wh)}
                            className="rounded p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                            title="編輯"
                          >
                            <Pencil className="h-3.5 w-3.5" />
                          </button>
                          <button
                            onClick={() => handleTest(wh)}
                            disabled={testingId === wh.id || !wh.enabled}
                            className="rounded p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-30"
                            title="傳送測試"
                          >
                            {testingId === wh.id ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Send className="h-3.5 w-3.5" />
                            )}
                          </button>
                          <button
                            onClick={() => handleDelete(wh)}
                            className="rounded p-1.5 text-muted-foreground hover:bg-red-50 hover:text-red-600"
                            title="刪除"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>

      {/* Payload preview */}
      <div className="rounded-lg border bg-white">
        <div className="border-b px-5 py-3">
          <h3 className="text-sm font-semibold">資料格式</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Discord 相容格式（multipart/form-data 附帶快照圖片）
          </p>
        </div>
        <div className="p-5">
          <pre className="overflow-x-auto rounded-md border bg-muted/50 p-3 font-mono text-[11px] text-muted-foreground">
{`// multipart/form-data:
//   payload_json: (下方 JSON)
//   files[0]:     snapshot.jpg (標註框快照)

{
  "content": "\ud83d\udc12 **偵測到猴子！**\\n攝影機：\`cam1\` | 信心度：\`92%\`",
  "embeds": [{
    "title": "猴子偵測警報",
    "color": 15736644,
    "fields": [
      { "name": "攝影機", "value": "cam1", "inline": true },
      { "name": "信心度", "value": "92.0%", "inline": true },
      { "name": "追蹤 ID", "value": "#1", "inline": true }
    ],
    "image": { "url": "attachment://snapshot.jpg" },
    "timestamp": "2026-02-12T19:30:00+00:00"
  }]
}`}
          </pre>
        </div>
      </div>
    </div>
  );
}
