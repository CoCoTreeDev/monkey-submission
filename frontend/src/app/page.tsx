"use client";

import { useState, useEffect } from "react";
import useSWR from "swr";

import { Header, type TabId } from "@/components/header";
import { LoginScreen } from "@/components/login-screen";
import { StatsCards } from "@/components/stats-cards";
import { StreamViewer } from "@/components/stream-viewer";
import { DetectionLog } from "@/components/detection-log";
import {
  DetectionListPanel,
  type DetectionSort,
  type DetectionSortOrder,
} from "@/components/detection-list";
import { DetectionAnalytics } from "@/components/detection-analytics";
import { DetectionReview } from "@/components/detection-review";
import { AddCameraForm } from "@/components/add-camera-form";
import { WebhookSettings } from "@/components/webhook-config";
import { StorageOverview } from "@/components/storage-overview";
import { CameraManagement } from "@/components/camera-management";
import { InferencePerformance } from "@/components/inference-performance";
import { NodesPanel } from "@/components/nodes-panel";
import { ImageManagement } from "@/components/image-management";
import { ReIDLabeling } from "@/components/reid-labeling";
import { TrainingPanel } from "@/components/training-panel";
import { UserManagement } from "@/components/user-management";
import {
  API_BASE,
  fetcher,
  getSession,
  logout,
  type Camera,
  type DetectionList,
  type DetectionStats,
  type Me,
  type Role,
} from "@/lib/api";

const PAGE_SIZE_OPTIONS = [10, 20, 30, 50, 100];

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [pageSize, setPageSize] = useState(30);
  const [cameraFilter, setCameraFilter] = useState<string>("all");
  const [detectionSort, setDetectionSort] = useState<DetectionSort>("time");
  const [detectionSortOrder, setDetectionSortOrder] = useState<DetectionSortOrder>("desc");
  // Rows by default, and the visit view is ADMIN-ONLY. A detection row is one
  // thing the system actually recorded; a visit is an inference on top of it
  // (rows grouped by a 20 s gap, with an ESTIMATE of how many animals were
  // behind the track ids). Users get the record, not the interpretation —
  // Single source of truth: the selected index across the WHOLE filtered
  // result set. The page number and the in-page selection are both derived
  // from it, so the list and the review pane stay in sync without effects.
  const [selectedGlobalIdx, setSelectedGlobalIdx] = useState(0);
  // Polling can insert a newer row above the current one. Keep the selected
  // record by id as well as by position, otherwise the player silently moves
  // to the next row every five seconds while the list is refreshing.
  const [selectedDetectionId, setSelectedDetectionId] = useState<number | null>(null);

  // ── Auth gate ─────────────────────────────────────────────────────────────
  // A fresh/cookieless browser has no valid session → the login screen renders
  // and no data is fetched. `authReady` avoids a flash of login before the
  // client reads sessionStorage on mount.
  const [session, setSession] = useState<{ token: string; role: Role } | null>(
    null
  );
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      setSession(getSession());
      setAuthReady(true);
    });
    const onExpired = () => setSession(null);
    window.addEventListener("auth-expired", onExpired);
    return () => {
      cancelled = true;
      window.removeEventListener("auth-expired", onExpired);
    };
  }, []);

  const isLoggedIn = !!session;
  const isAdmin = session?.role === "admin";

  // Effective permissions for the current user (server is the real boundary; this
  // only gates what the UI shows). Admins bypass; unknown flags default to on.
  const { data: me } = useSWR<Me>(
    isLoggedIn ? `${API_BASE}/api/auth/me` : null,
    fetcher,
    { refreshInterval: 30000 }
  );
  const canFeeds = isAdmin || (me?.feature_flags?.live_feeds ?? true);
  const canDownload = isAdmin || (me?.feature_flags?.downloads ?? true);
  // training is OPT-IN (unlike other flags): a missing flag means no access.
  const canTraining = isAdmin || (me?.feature_flags?.training ?? false);

  const handleLogout = () => {
    logout();
    setSession(null);
    setActiveTab("overview");
  };

  const {
    data: stats,
    isLoading: statsLoading,
    mutate: mutateStats,
  } = useSWR<DetectionStats>(
    isLoggedIn ? `${API_BASE}/api/detections/stats` : null,
    fetcher,
    // Freeze the page while a record is selected. A new detection arriving at
    // the top otherwise shifts the offset under the selected id; the detail
    // pane briefly shows one record and seeks another during review.
    { refreshInterval: selectedDetectionId === null ? 5000 : 0 }
  );

  const {
    data: cameras,
    isLoading: camerasLoading,
    mutate: mutateCameras,
  } = useSWR<Camera[]>(isLoggedIn ? `${API_BASE}/api/cameras` : null, fetcher, {
    refreshInterval: 10000,
  });

  // If the filtered camera no longer exists, fall back to "all" (derived, no effect).
  const cameraStillExists =
    cameraFilter === "all" ||
    (cameras?.some((cam) => cam.camera_id === cameraFilter) ?? true);
  const effectiveCameraFilter = cameraStillExists ? cameraFilter : "all";

  const detectionPage = Math.floor(selectedGlobalIdx / pageSize) + 1;
  const detectionOffset = (detectionPage - 1) * pageSize;
  // "all" is a SELECT sentinel, not a camera. Resolve it once before building
  // the API query so the backend never receives a camera named "all".
  const activeCamera = effectiveCameraFilter === "all" ? null : effectiveCameraFilter;
  const cameraFilterQuery =
    activeCamera
      ? `&camera_id=${encodeURIComponent(activeCamera)}`
      : "";
  const {
    data: detections,
    isLoading: detectionsLoading,
    mutate: mutateDetections,
  } = useSWR<DetectionList>(
    isLoggedIn
      ? `${API_BASE}/api/detections?limit=${pageSize}&offset=${detectionOffset}&sort_by=${detectionSort}&sort_order=${detectionSortOrder}${cameraFilterQuery}`
      : null,
    fetcher,
    // Do not keep the previous page here. It belongs to a different selection
    // coordinate: showing it while a new page loads makes the player seek the
    // old row, then jump again when the requested page arrives.
    { refreshInterval: 5000 }
  );

  const detectionItems = detections?.items ?? [];
  const totalDetections = detections?.total ?? 0;
  const totalDetectionPages = Math.max(1, Math.ceil(totalDetections / pageSize));
  const selectedItemIdx = selectedDetectionId === null
    ? -1
    : detectionItems.findIndex((item) => item.id === selectedDetectionId);
  const selectedIdx = selectedItemIdx >= 0
    ? selectedItemIdx
    : Math.min(selectedGlobalIdx % pageSize, Math.max(detectionItems.length - 1, 0));
  // While the selected id remains on this page, it is the authoritative
  // position for navigation. This keeps polling from moving the player when a
  // newer detection is inserted above it, without an effect that causes a
  // second render just to repair an index.
  const selectedGlobalPosition = selectedItemIdx >= 0
    ? detectionOffset + selectedItemIdx
    : selectedGlobalIdx;

  const handleCameraFilterChange = (value: string) => {
    setCameraFilter(value);
    setSelectedGlobalIdx(0);
    setSelectedDetectionId(null);
  };
  const handlePageSizeChange = (value: number) => {
    setPageSize(value);
    setSelectedGlobalIdx(0);
    setSelectedDetectionId(null);
  };
  const handleDetectionSortChange = (value: DetectionSort) => {
    setDetectionSort(value);
    setSelectedGlobalIdx(0);
    setSelectedDetectionId(null);
  };
  const handleDetectionSortOrderChange = () => {
    setDetectionSortOrder((current) => (current === "desc" ? "asc" : "desc"));
    setSelectedGlobalIdx(0);
    setSelectedDetectionId(null);
  };
  // Pagination jump → land on the first item of the target page.
  const handlePageChange = (p: number) => {
    setSelectedGlobalIdx((p - 1) * pageSize);
    setSelectedDetectionId(null);
  };
  // In-page row click → absolute index across the whole result set.
  const handleSelectIdx = (localIdx: number) => {
    setSelectedGlobalIdx((detectionPage - 1) * pageSize + localIdx);
    setSelectedDetectionId(detectionItems[localIdx]?.id ?? null);
  };
  // Prev/next; the page auto-derives from the index, crossing boundaries.
  const goPrev = () => {
    const next = Math.max(0, selectedGlobalPosition - 1);
    setSelectedGlobalIdx(next);
    setSelectedDetectionId(Math.floor(next / pageSize) === detectionPage
      ? detectionItems[next % pageSize]?.id ?? null
      : null);
  };
  const goNext = () => {
    const next = Math.min(Math.max(totalDetections - 1, 0), selectedGlobalPosition + 1);
    setSelectedGlobalIdx(next);
    setSelectedDetectionId(Math.floor(next / pageSize) === detectionPage
      ? detectionItems[next % pageSize]?.id ?? null
      : null);
  };

  const handleCameraAdded = () => {
    mutateCameras();
    mutateStats();
    mutateDetections();
  };

  // ── Render gates ────────────────────────────────────────────────────────────
  if (!authReady) {
    return <div className="min-h-screen bg-background" />;
  }
  if (!session) {
    return <LoginScreen onLogin={() => setSession(getSession())} />;
  }

  return (
    <div className="min-h-screen bg-background">
      <Header
        camerasActive={stats?.cameras_active ?? 0}
        isAdmin={isAdmin}
        canFeeds={canFeeds}
        canTraining={canTraining}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onLogout={handleLogout}
      />

      <main className="mx-auto max-w-7xl 2xl:max-w-[1800px] px-6 py-6">
        {/* ── Overview ─────────────────────────────────────── */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            <StatsCards stats={stats} isLoading={statsLoading} />

            {isAdmin && <StorageOverview />}

            <DetectionLog
              detections={(detections?.items ?? []).slice(0, 10)}
              isLoading={detectionsLoading}
              showConfidence={isAdmin}
              canDownload={canDownload}
            />
          </div>
        )}

        {/* ── Live Feeds ───────────────────────────────────── */}
        {activeTab === "feeds" && canFeeds && (
          <div>
            {!camerasLoading && cameras && cameras.length > 0 ? (
              <div className="grid gap-5 lg:grid-cols-2">
                {cameras.map((cam) => (
                  <StreamViewer
                    key={cam.camera_id}
                    cameraId={cam.camera_id}
                    cameraName={cam.name || cam.camera_id}
                    isAlive={cam.is_alive ?? false}
                  />
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed bg-white py-24 text-sm text-muted-foreground">
                {camerasLoading ? (
                  "載入攝影機中..."
                ) : (
                  <>
                    <p>尚未註冊任何攝影機。</p>
                    {isAdmin && (
                      <button
                        className="mt-1 text-primary hover:underline"
                        onClick={() => setActiveTab("operations")}
                      >
                        前往操作管理新增攝影機
                      </button>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Detections ───────────────────────────────────── */}
        {/* Master-detail: list (selector + controls) on the left, detail
            review on the right (sticky). Stacks on small screens. */}

        {activeTab === "detections" && (
          <div className="grid items-stretch gap-5 lg:h-[calc(100dvh-8.75rem)] lg:grid-cols-[minmax(320px,360px)_minmax(0,1fr)]">
            <DetectionListPanel
              detections={detectionItems}
              isLoading={detectionsLoading}
              total={totalDetections}
              page={detectionPage}
              totalPages={totalDetectionPages}
              pageSize={pageSize}
              pageSizeOptions={PAGE_SIZE_OPTIONS}
              onPageChange={handlePageChange}
              onPageSizeChange={handlePageSizeChange}
              sortBy={detectionSort}
              onSortByChange={handleDetectionSortChange}
              sortOrder={detectionSortOrder}
              onSortOrderChange={handleDetectionSortOrderChange}
              selectedIdx={selectedIdx}
              onSelectIdx={handleSelectIdx}
              cameras={cameras}
              cameraFilter={effectiveCameraFilter}
              onCameraFilterChange={handleCameraFilterChange}
              showConfidence={isAdmin}
            />
            <div className="min-h-0 overflow-hidden lg:h-full">
              <DetectionReview
                detections={detectionItems}
                isLoading={detectionsLoading}
                total={totalDetections}
                page={detectionPage}
                pageSize={pageSize}
                totalPages={totalDetectionPages}
                selectedIdx={selectedIdx}
                onPrev={goPrev}
                onNext={goNext}
                canDownload={canDownload}
                showConfidence={isAdmin}
                isAdmin={isAdmin}
                onVisibilityChange={() => mutateDetections()}
              />
            </div>
          </div>
        )}

        {/* ── Analytics: time-of-day / calendar heatmaps ───── */}
        {activeTab === "analytics" && <DetectionAnalytics />}

        {/* ── Operations: cameras + inference (admin only) ─── */}
        {activeTab === "operations" && isAdmin && (
          <div className="space-y-5">
            <AddCameraForm onAdded={handleCameraAdded} />
            <CameraManagement />
            <InferencePerformance />
          </div>
        )}

        {/* ── Nodes: RunPod-style inference node dashboard (admin only) ── */}
        {activeTab === "nodes" && isAdmin && <NodesPanel />}

        {/* ── Images (admin only) ──────────────────────────── */}
        {activeTab === "images" && isAdmin && (
          <ImageManagement canTraining={canTraining} />
        )}

        {/* ── ReID human-in-the-loop labels (admin only) ── */}
        {activeTab === "reid" && isAdmin && <ReIDLabeling />}

        {/* ── Training (admin or `training` feature flag) ──── */}
        {activeTab === "training" && canTraining && <TrainingPanel />}

        {/* ── Users (admin only) ───────────────────────────── */}
        {activeTab === "users" && isAdmin && <UserManagement />}

        {/* ── Webhooks (admin only) ────────────────────────── */}
        {activeTab === "webhooks" && isAdmin && <WebhookSettings />}
      </main>
    </div>
  );
}
