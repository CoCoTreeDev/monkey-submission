"use client";

import { Eye, LogOut } from "lucide-react";

export type TabId =
  | "overview"
  | "feeds"
  | "detections"
  | "analytics"
  | "operations"
  | "nodes"
  | "images"
  | "reid"
  | "training"
  | "users"
  | "webhooks";

interface Tab {
  id: TabId;
  label: string;
  adminOnly?: boolean;
}

const tabs: Tab[] = [
  { id: "overview", label: "總覽" },
  { id: "feeds", label: "即時影像" },
  { id: "detections", label: "偵測紀錄" },
  { id: "analytics", label: "統計分析" },
  { id: "operations", label: "操作管理", adminOnly: true },
  { id: "nodes", label: "節點", adminOnly: true },
  { id: "images", label: "影像管理", adminOnly: true },
  { id: "reid", label: "ReID 標記", adminOnly: true },
  // 訓練 is NOT adminOnly: visible for admins OR users with the `training`
  // feature flag (canTraining) — the flag is opt-in, so it defaults hidden.
  { id: "training", label: "訓練" },
  { id: "users", label: "使用者管理", adminOnly: true },
  { id: "webhooks", label: "Webhook 通知", adminOnly: true },
];

interface HeaderProps {
  camerasActive: number;
  isAdmin?: boolean;
  canFeeds?: boolean;
  canTraining?: boolean;
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  onLogout: () => void;
}

export function Header({
  camerasActive,
  isAdmin = false,
  canFeeds = true,
  canTraining = false,
  activeTab,
  onTabChange,
  onLogout,
}: HeaderProps) {
  const visibleTabs = tabs.filter(
    (tab) =>
      (isAdmin || !tab.adminOnly) &&
      (tab.id !== "feeds" || canFeeds) &&
      (tab.id !== "training" || canTraining)
  );

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-white">
      <div className="mx-auto max-w-7xl 2xl:max-w-[1800px] px-6">
        {/* Top row: brand + status */}
        <div className="flex h-14 items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Eye className="h-5 w-5 text-primary" />
            <span className="text-[15px] font-semibold text-foreground">
              台灣獼猴監測平台
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <div
                className={`h-2 w-2 rounded-full ${
                  camerasActive > 0 ? "bg-green-500" : "bg-gray-300"
                }`}
              />
              <span>{camerasActive} 台攝影機運作中</span>
            </div>
            <button
              onClick={onLogout}
              className="rounded p-1.5 text-muted-foreground transition-colors hover:text-foreground"
              title="登出"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Tab row */}
        <nav className="-mb-px flex min-w-0 gap-6 overflow-x-auto whitespace-nowrap">
          {visibleTabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`relative pb-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {tab.label}
              {activeTab === tab.id && (
                <span className="absolute bottom-0 left-0 right-0 h-[2px] rounded-full bg-primary" />
              )}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
