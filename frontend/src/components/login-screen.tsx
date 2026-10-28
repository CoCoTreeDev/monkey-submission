"use client";

import { useState } from "react";
import { Eye, LogIn, Loader2 } from "lucide-react";
import { login, type Role } from "@/lib/api";

/**
 * Full-screen login gate. A cookieless / fresh browser sees only this until a
 * valid access code is entered. `cocotree` → admin, `user` → general user.
 */
export function LoginScreen({ onLogin }: { onLogin: (role: Role) => void }) {
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!code.trim()) return;
    setLoading(true);
    setError("");
    try {
      const role = await login(code.trim());
      onLogin(role);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "登入失敗");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-6">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex flex-col items-center gap-2 text-center">
          <div className="flex items-center gap-2.5">
            <Eye className="h-6 w-6 text-primary" />
            <span className="text-lg font-semibold text-foreground">
              台灣獼猴監測平台
            </span>
          </div>
          <p className="text-sm text-muted-foreground">請輸入存取碼以進入系統</p>
        </div>
        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                存取碼
              </label>
              <input
                type="password"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="請輸入存取碼"
                className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
                autoFocus
                required
              />
            </div>
            {error && <p className="text-xs text-red-600">{error}</p>}
            <button
              type="submit"
              disabled={loading}
              className="flex w-full items-center justify-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary/90 disabled:opacity-40"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <LogIn className="h-4 w-4" />
              )}
              登入
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
