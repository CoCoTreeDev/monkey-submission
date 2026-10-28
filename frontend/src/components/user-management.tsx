"use client";

import { useState } from "react";
import { MessageBanner, useFlashMessage } from "@/components/status-message";
import useSWR from "swr";
import {
  UserPlus,
  Pencil,
  Trash2,
  Check,
  X,
  Loader2,
  ShieldCheck,
  User as UserIcon,
} from "lucide-react";
import {
  API_BASE,
  fetcher,
  createUser,
  updateUser,
  deleteUser,
  type Camera,
  type FeatureDef,
  type Role,
  type User,
} from "@/lib/api";

/** Shared form state for adding or editing a user. */
interface FormState {
  name: string;
  code: string;
  role: Role;
  enabled: boolean;
  allowAll: boolean;
  allowed: Set<string>;
  flags: Record<string, boolean>;
}

function initialFlags(features: FeatureDef[], from?: Record<string, boolean>): Record<string, boolean> {
  const out: Record<string, boolean> = {};
  // Defaults mirror server enforcement: features default ON (_feature_enabled),
  // EXCEPT training — it can start paid GPU jobs / deploys, so it is opt-in.
  for (const f of features) out[f.key] = from?.[f.key] ?? f.key !== "training";
  return out;
}

function emptyForm(features: FeatureDef[], cameras: Camera[]): FormState {
  return {
    name: "",
    code: "",
    role: "user",
    enabled: true,
    // Default to explicit per-camera grants (all current cameras pre-selected) rather
    // than the "*" wildcard, so adding a camera later does NOT silently grant it.
    allowAll: false,
    allowed: new Set(cameras.map((c) => c.camera_id)),
    flags: initialFlags(features),
  };
}

function formFromUser(user: User, features: FeatureDef[]): FormState {
  const allowAll = user.allowed_cameras.includes("*");
  return {
    name: user.name,
    code: "",
    role: user.role,
    enabled: user.enabled,
    allowAll,
    allowed: new Set(user.allowed_cameras.filter((c) => c !== "*")),
    flags: initialFlags(features, user.feature_flags),
  };
}

function buildBody(form: FormState) {
  return {
    name: form.name.trim(),
    role: form.role,
    enabled: form.enabled,
    allowed_cameras: form.allowAll ? ["*"] : Array.from(form.allowed),
    feature_flags: form.flags,
  };
}

// ── Reusable form (add + edit) ───────────────────────────────────────────────

function UserForm({
  cameras,
  features,
  form,
  setForm,
  onSubmit,
  submitting,
  submitLabel,
  onCancel,
  requireCode,
}: {
  cameras: Camera[];
  features: FeatureDef[];
  form: FormState;
  setForm: (f: FormState) => void;
  onSubmit: () => void;
  submitting: boolean;
  submitLabel: string;
  onCancel?: () => void;
  requireCode: boolean;
}) {
  const toggleCamera = (id: string) => {
    const next = new Set(form.allowed);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setForm({ ...form, allowed: next });
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">顯示名稱</label>
          <input
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="例如：研究員 A"
            className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">
            存取碼{requireCode ? "" : "（留空則不變更）"}
          </label>
          <input
            type="text"
            value={form.code}
            onChange={(e) => setForm({ ...form, code: e.target.value })}
            placeholder={requireCode ? "登入用存取碼" : "••••••"}
            className="w-full rounded-md border bg-background px-3 py-2 font-mono text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
          />
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">角色</label>
          <select
            value={form.role}
            onChange={(e) => setForm({ ...form, role: e.target.value as Role })}
            className="rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
          >
            <option value="user">一般使用者</option>
            <option value="admin">管理員</option>
          </select>
        </div>
        <div className="space-y-1">
          <label aria-hidden className="invisible block select-none text-xs font-medium">
            啟用
          </label>
          <label className="flex items-center gap-2 py-2 text-sm">
            <input
              type="checkbox"
              checked={form.enabled}
              onChange={(e) => setForm({ ...form, enabled: e.target.checked })}
              className="h-4 w-4"
            />
            啟用帳號
          </label>
        </div>
      </div>

      {/* Camera access — hidden for admins (they always see everything) */}
      {form.role === "user" && (
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground">可存取攝影機</label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={form.allowAll}
              onChange={(e) => setForm({ ...form, allowAll: e.target.checked })}
              className="h-4 w-4"
            />
            全部攝影機（含日後新增）
          </label>
          <p className="text-xs text-muted-foreground">
            {form.allowAll
              ? "此帳號會自動取得日後新增攝影機的存取權。"
              : "僅授權下方勾選的攝影機；日後新增的攝影機不會自動授權。"}
          </p>
          {!form.allowAll && (
            <div className="grid grid-cols-2 gap-2 rounded-md border p-3 sm:grid-cols-3">
              {cameras.length === 0 ? (
                <p className="col-span-full text-xs text-muted-foreground">尚無攝影機</p>
              ) : (
                cameras.map((cam) => (
                  <label key={cam.camera_id} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={form.allowed.has(cam.camera_id)}
                      onChange={() => toggleCamera(cam.camera_id)}
                      className="h-4 w-4"
                    />
                    <span className="truncate">{cam.name || cam.camera_id}</span>
                  </label>
                ))
              )}
            </div>
          )}
        </div>
      )}

      {/* Feature toggles — only meaningful for users (admin bypasses all) */}
      {form.role === "user" && features.length > 0 && (
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground">功能開關</label>
          <div className="flex flex-wrap gap-4 rounded-md border p-3">
            {features.map((f) => (
              <label key={f.key} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={form.flags[f.key] ?? true}
                  onChange={(e) =>
                    setForm({ ...form, flags: { ...form.flags, [f.key]: e.target.checked } })
                  }
                  className="h-4 w-4"
                />
                {f.label}
              </label>
            ))}
          </div>
        </div>
      )}

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onSubmit}
          disabled={submitting || (requireCode && !form.code.trim())}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-40"
        >
          {submitting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
          {submitLabel}
        </button>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            disabled={submitting}
            className="inline-flex items-center gap-1.5 rounded-md border px-4 py-2 text-xs font-medium hover:bg-muted disabled:opacity-40"
          >
            <X className="h-3.5 w-3.5" />
            取消
          </button>
        )}
      </div>
    </div>
  );
}

// ── User management panel ────────────────────────────────────────────────────

export function UserManagement() {
  const { data: users, isLoading, mutate } = useSWR<User[]>(
    `${API_BASE}/api/admin/users`,
    fetcher,
    { refreshInterval: 30000 }
  );
  const { data: cameras } = useSWR<Camera[]>(`${API_BASE}/api/cameras`, fetcher);
  const { data: features } = useSWR<FeatureDef[]>(
    `${API_BASE}/api/admin/features`,
    fetcher
  );

  const cams = cameras ?? [];
  const feats = features ?? [];

  const [addForm, setAddForm] = useState<FormState | null>(null);
  const [addSubmitting, setAddSubmitting] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editForm, setEditForm] = useState<FormState | null>(null);
  const [editSubmitting, setEditSubmitting] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [message, flash] = useFlashMessage();

  const openAdd = () => setAddForm(emptyForm(feats, cams));
  const openEdit = (u: User) => {
    setEditingId(u.id);
    setEditForm(formFromUser(u, feats));
  };

  const handleAdd = async () => {
    if (!addForm) return;
    setAddSubmitting(true);
    try {
      await createUser({ ...buildBody(addForm), code: addForm.code.trim() });
      flash("success", "已新增帳號");
      setAddForm(null);
      mutate();
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "新增失敗");
    } finally {
      setAddSubmitting(false);
    }
  };

  const handleEdit = async () => {
    if (!editForm || editingId === null) return;
    setEditSubmitting(true);
    try {
      const body = buildBody(editForm);
      await updateUser(editingId, {
        ...body,
        ...(editForm.code.trim() ? { code: editForm.code.trim() } : {}),
      });
      flash("success", "已更新帳號");
      setEditingId(null);
      setEditForm(null);
      mutate();
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "更新失敗");
    } finally {
      setEditSubmitting(false);
    }
  };

  const handleDelete = async (u: User) => {
    if (!window.confirm(`確定要刪除帳號「${u.name || u.id}」？`)) return;
    setDeletingId(u.id);
    try {
      await deleteUser(u.id);
      flash("success", "已刪除帳號");
      if (editingId === u.id) setEditingId(null);
      mutate();
    } catch (err: unknown) {
      flash("error", err instanceof Error ? err.message : "刪除失敗");
    } finally {
      setDeletingId(null);
    }
  };

  const cameraSummary = (u: User): string => {
    if (u.role === "admin") return "全部（管理員）";
    if (u.allowed_cameras.includes("*")) return "全部攝影機";
    if (u.allowed_cameras.length === 0) return "無";
    return `${u.allowed_cameras.length} 台`;
  };

  return (
    <div className="rounded-lg border bg-white">
      <div className="flex items-center justify-between border-b px-5 py-3">
        <div>
          <h3 className="text-sm font-semibold">使用者管理</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            新增帳號、設定可存取的攝影機與功能開關（管理員不受限制）。
          </p>
        </div>
        {!addForm && (
          <button
            onClick={openAdd}
            className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primary/90"
          >
            <UserPlus className="h-3.5 w-3.5" />
            新增帳號
          </button>
        )}
      </div>

      <MessageBanner message={message} className="mx-5 mt-3" />

      {addForm && (
        <div className="border-b bg-muted/30 px-5 py-4">
          <h4 className="mb-3 text-xs font-semibold text-muted-foreground">新增帳號</h4>
          <UserForm
            cameras={cams}
            features={feats}
            form={addForm}
            setForm={setAddForm}
            onSubmit={handleAdd}
            submitting={addSubmitting}
            submitLabel="新增"
            onCancel={() => setAddForm(null)}
            requireCode
          />
        </div>
      )}

      <div className="divide-y">
        {isLoading ? (
          <p className="px-5 py-6 text-xs text-muted-foreground">載入中…</p>
        ) : !users?.length ? (
          <p className="px-5 py-6 text-xs text-muted-foreground">尚無帳號</p>
        ) : (
          users.map((u) => (
            <div key={u.id} className="px-5 py-3">
              {editingId === u.id && editForm ? (
                <div className="space-y-3">
                  <p className="text-xs font-semibold text-muted-foreground">
                    編輯「{u.name || u.id}」
                  </p>
                  <UserForm
                    cameras={cams}
                    features={feats}
                    form={editForm}
                    setForm={setEditForm}
                    onSubmit={handleEdit}
                    submitting={editSubmitting}
                    submitLabel="儲存"
                    onCancel={() => {
                      setEditingId(null);
                      setEditForm(null);
                    }}
                    requireCode={false}
                  />
                </div>
              ) : (
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-2.5">
                    {u.role === "admin" ? (
                      <ShieldCheck className="h-4 w-4 shrink-0 text-primary" />
                    ) : (
                      <UserIcon className="h-4 w-4 shrink-0 text-muted-foreground" />
                    )}
                    <span className="text-sm font-medium">{u.name || `#${u.id}`}</span>
                    <span className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">
                      {u.role === "admin" ? "管理員" : "使用者"}
                    </span>
                    {!u.enabled && (
                      <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[11px] text-gray-500">
                        已停用
                      </span>
                    )}
                    <span className="text-[11px] text-muted-foreground">
                      攝影機：{cameraSummary(u)}
                    </span>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <button
                      onClick={() => openEdit(u)}
                      title="編輯"
                      className="inline-flex h-8 w-8 items-center justify-center rounded-md border text-muted-foreground hover:bg-muted hover:text-foreground"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => handleDelete(u)}
                      disabled={deletingId === u.id}
                      title="刪除"
                      className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-red-200 text-red-600 hover:bg-red-50 disabled:opacity-40"
                    >
                      {deletingId === u.id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="h-3.5 w-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
