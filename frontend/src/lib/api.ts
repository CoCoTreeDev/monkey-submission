/**
 * Backend API client and types for the Monkey Detection Platform.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────────────

export interface Detection {
  id: number;
  camera_id: string;
  timestamp: string;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_w: number;
  bbox_h: number;
  image_path: string;
  annotated_path: string;
  crop_path: string;
  track_id: number;
  trail_path: string;
  /** Manual gate override: null = follow the confidence rule. */
  user_visible?: boolean | null;
  /** What a non-admin actually sees (rule + override), computed server-side. */
  visible_to_users?: boolean;
  /** Delete guard set by a human who confirmed the record. Both delete paths
   *  refuse it. Independent of the visibility fields above. */
  locked?: boolean;
  /** Which visit this row belongs to — every row the producer emitted for one
   *  stretch of animals in front of the camera shares it. Null on rows written
   *  before visits were recorded. */
  visit_id?: string | null;
  /** Most animals followed at once in that visit, counted at emit time. */
  visit_peak?: number | null;
  /** Stable identity assigned by the Reference/review grouping. */
  animal_id?: string | null;
  /** How the animal identity was assigned (`rule` or `human`). */
  animal_source?: string | null;
  /** Where in the clip this animal is, in seconds from the clip's first frame,
   *  and how long the clip runs. Null on rows written before these existed. */
  clip_offset_seconds?: number | null;
  clip_end_offset_seconds?: number | null;
  clip_duration_seconds?: number | null;
  /** The stretches it was actually visible, `[[from, to], ...]` in the same
   *  seconds. An animal that walks out and comes back has more than one, which
   *  a single start→end bar would paint over as continuous presence. */
  clip_segments?: [number, number][] | null;
  /** Per-frame boxes from the persisted track. */
  track_boxes?: [number, number, number, number, number, number][];
  track_box_fps?: number;
}

export interface DetectionList {
  total: number;
  items: Detection[];
}

export type ReIDLabel = "same" | "different" | "uncertain";
export type ReIDTemporalRelation = "separate" | "overlap" | "unknown";
/** [frame index, x1, y1, x2, y2, confidence] in the trail video's source axis. */
export type ReIDTrackBox = [number, number, number, number, number, number];

export interface ReIDTrack {
  id: number;
  track_id: number;
  camera_id: string;
  timestamp: string;
  confidence: number;
  crop_path: string;
  image_path: string;
  annotated_path: string;
  trail_path: string;
  clip_duration_seconds: number | null;
  start_seconds: number | null;
  end_seconds: number | null;
  frame_count: number;
  track_boxes?: ReIDTrackBox[] | null;
  track_box_fps?: number | null;
}

export interface ReIDPair {
  pair_id: string;
  camera_id: string;
  trail_path: string;
  clip_duration_seconds: number | null;
  temporal_relation: ReIDTemporalRelation;
  gap_seconds: number | null;
  a: ReIDTrack;
  b: ReIDTrack;
  label: ReIDLabel | null;
  note: string;
  labeled_at: string | null;
  /** Deterministic queue shortcut; never written as human ground truth. */
  system_label: "different" | null;
  decision_source: "human" | "geometry_auto" | null;
  review_required: boolean;
  review_priority: number;
  review_reason: string;
}

export interface ReIDPairList {
  /** Number of pairs in the selected status filter within the review batch. */
  total: number;
  /** All valid candidates before the fixed review batch is selected. */
  candidate_total: number;
  /** Full candidate-pool counts, including pairs outside this round. */
  pending: number;
  labeled: number;
  /** Counts for the fixed batch shown in this review round. */
  batch_total: number;
  batch_pending: number;
  batch_labeled: number;
  /** Full-pool and fixed-round counts for system-only negative triage. */
  auto_resolved: number;
  batch_auto_resolved: number;
  batch_review_pending: number;
  pairs: ReIDPair[];
}

export interface DetectionStats {
  total_today: number;
  total_all: number;
  last_detection: string | null;
  cameras_active: number;
}

/** One (date, hour, camera) bucket from /api/detections/heatmap. */
export interface HeatmapCell {
  date: string; // YYYY-MM-DD, TW-local
  hour: number; // 0-23
  camera_id: string;
  count: number;
}

export interface DetectionHeatmap {
  start: string; // first date of the range (inclusive)
  end: string; // last date of the range (inclusive)
  total: number;
  cells: HeatmapCell[];
}

export interface Camera {
  id: number;
  camera_id: string;
  rtsp_url: string;
  name: string;
  created_at: string;
  poc?: boolean;
  // Runtime status from /api/cameras — is_alive = delivering non-black frames.
  is_alive?: boolean;
}

export interface CameraCreate {
  camera_id: string;
  rtsp_url: string;
  name?: string;
}

export interface CameraUpdate {
  name?: string;
  rtsp_url?: string;
}

export interface CameraDeleteResult {
  camera_id: string;
  detections_preserved: number;
}

export interface Webhook {
  id: number;
  name: string;
  url: string;
  enabled: boolean;
  last_error: string;
  created_at: string;
}

export interface WebhookCreate {
  name?: string;
  url: string;
  enabled?: boolean;
}

export interface WebhookUpdate {
  name?: string;
  url?: string;
  enabled?: boolean;
}

export interface StorageStats {
  total_files: number;
  total_size_bytes: number;
  total_size_human: string;
  total_detections: number;
}

export interface InferenceCameraStats {
  camera_id: string;
  stream_connected: boolean;
  frame_skip?: number;
  coverage_ceiling?: number;
  inference_ms_last: number;
  inference_ms_avg: number;
  throughput_fps: number;
  sample_fps: number;
  total_inferences: number;
  // Wasted work, so throughput can never hide it again: distinct frames
  // inferred, turns that found no new frame, inferences on an already-processed
  // frame (must stay 0), and the rolling share of GPU turns given back unused.
  frames_processed: number;
  frames_unchanged: number;
  duplicate_inferences: number;
  idle_ratio: number;
  // Idle gate — a feed carrying no image at all (lens-black at night) is
  // sampled at a heartbeat so its GPU turns go to cameras that can see. Live
  // verdict, the share of frames it dropped, and the pixel variance behind it.
  idle_gated?: boolean;
  gated_ratio?: number;
  frames_gated?: number;
  frame_stddev?: number;
  capture_fps?: number;
  stream_reconnects?: number;
  seconds_since_last: number | null;
  // Coverage — inferred frames / delivered frames. The camera can be delivering
  // 33 FPS while the GPU only looks at 11 of them; both numbers were already
  // reported and nobody divided them.
  coverage_ratio?: number;
  // Detector-THREAD liveness, which is not stream_connected: a worker thread
  // died on 2026-08-08 and the connected stream made it look merely slow.
  worker_alive?: boolean;
  worker_crashes?: number;
  worker_revivals?: number;
  /** Size this camera is actually inferred at — the floor, or its own frame
   *  size when that is larger. Showing only the floor is how a 1920x1080 feed
   *  being squeezed into 1600 stayed invisible. */
  imgsz?: number;
}

export interface InferenceStats {
  /** The FLOOR. Each camera reports the size it actually uses. */
  inference_imgsz: number;
  frame_skip: number;
  model_path: string;
  coverage_ceiling?: number; // 1/FRAME_SKIP — the most coverage the loop can reach
  cameras: InferenceCameraStats[];
}

// Phase 4.5: edge-node heartbeat (Mac mini utilization) for the admin UI.
export interface EdgeStatus {
  online: boolean;
  last_seen: string | null;
  seconds_since: number | null;
  heartbeat: {
    hostname: string;
    node_role?: string; // "api" = the cloud node's psutil self-sample (no GPU)
    cpu_percent: number | null;
    gpu_percent: number | null;
    memory_percent: number | null;
    memory_used_gb: number | null;
    memory_total_gb: number | null;
    process_rss_gb: number | null;
    cameras: InferenceCameraStats[];
  } | null;
}

// 節點 tab: per-node utilisation history for the line charts. The series is
// in-memory on the API node — it resets on every deploy (accepted for now).
export interface NodeHistoryPoint {
  t: string; // naive TW-local ISO — append +08:00 before Date parsing
  cpu: number | null;
  gpu: number | null;
  rss_gb: number | null;
}

export interface NodeHistory {
  node_id: string;
  points: NodeHistoryPoint[];
}

// Phase 2: per-user permission control.
export interface User {
  id: number;
  name: string;
  role: Role;
  enabled: boolean;
  allowed_cameras: string[]; // ["*"] = all cameras
  feature_flags: Record<string, boolean>;
  created_at: string;
}

export interface UserCreate {
  code: string;
  name?: string;
  role?: Role;
  enabled?: boolean;
  allowed_cameras?: string[];
  feature_flags?: Record<string, boolean>;
}

export interface UserUpdate {
  code?: string;
  name?: string;
  role?: Role;
  enabled?: boolean;
  allowed_cameras?: string[];
  feature_flags?: Record<string, boolean>;
}

/** The current user's own identity + effective permissions (from /api/auth/me). */
export interface Me {
  id: number;
  name: string;
  role: Role;
  allowed_cameras: string[];
  feature_flags: Record<string, boolean>;
}

export interface FeatureDef {
  key: string;
  label: string;
}

// ── Training (Ultralytics Platform control plane) ────────────────────────────
// The Platform is the single source of truth for datasets/models/training;
// these endpoints proxy it. Visible to admins or users with the `training`
// feature flag (opt-in — unlike other flags it defaults OFF).

export type TrainingStatus = "training" | "completed" | "failed" | "idle";

/** 本站評測 — latest eval_ratchet.py scores for one split, measured on the
 *  edge runner at deployment settings (imgsz 1600, fp16, MPS). */
export interface EvalScore {
  precision: number;
  recall: number;
  f1: number;
  passed: boolean; // ratchet exit 0 (F1 within tolerance of baseline)
  baseline_f1: number;
  created_at: string | null;
}

export interface TrainingModel {
  id: string;
  name: string;
  status: TrainingStatus;
  metrics: {
    precision: number | null;
    recall: number | null;
    map50: number | null;
  };
  createdAt: string | null;
  /** Latest 本站評測 per split; absent until model-evaluate.yml reports. */
  eval?: { test?: EvalScore; val?: EvalScore } | null;
}

/** Live progress passthrough from GET /api/admin/training/status/{id} —
 *  {job:{status,progress:{currentEpoch,totalEpochs,percentage,startedAt}}}. */
export interface TrainingStatusPayload {
  job?: {
    status?: string;
    progress?: {
      currentEpoch?: number;
      totalEpochs?: number;
      percentage?: number;
      startedAt?: string;
    };
  };
  [key: string]: unknown;
}

export interface TrainingStartBody {
  name: string;
  base?: string;
  epochs?: number;
  imgsz?: number;
  gpu?: string;
  /** Which Platform dataset to train on. Omitted = the backend master. */
  dataset?: string | null;
}

/** One incoming image the master already has a near-copy of. Informational —
 *  duplicates still land in the master (which archives everything); they are
 *  what the derived training set collapses away. */
export interface DuplicateNote {
  name: string;
  similar_to: string;
  distance: number; // dHash Hamming bits, 0 = pixel-identical after re-encode
}

export interface AddToDatasetResult {
  uploaded: number;
  failed: { id: number; reason: string }[];
  duplicates: DuplicateNote[];
}

// 去重與訓練集 — master + derived. The platform dataset is the MASTER (nothing
// deleted, test/val pinned); training runs on a DERIVED dataset whose TRAIN
// split has near-duplicates collapsed to one representative each.

export interface DerivedDataset {
  slug: string;
  name: string;
  dataset_id: string;
  kept: number;
  dropped: number;
  negatives_before: number;
  negatives_after: number;
  val_images: number;
  created_at: string | null;
}

export interface RegenerateResult {
  dataset_slug: string;
  dataset_id: string;
  name: string;
  kept: number;
  dropped: number;
  negatives_before: number;
  negatives_after: number;
  train_before: number;
  missing: number;
  val_images: number;
  expected_images: number;
  images_landed: number;
}

/** Regeneration is a minutes-long background job — POST starts it, GET polls. */
export interface RegenerateState {
  state: "idle" | "running" | "done" | "failed";
  step: string;
  done: number;
  total: number;
  error: string;
  result: RegenerateResult | null;
  started_at: string | null;
  finished_at: string | null;
  /** Newest derived dataset on record (present on GET only). */
  latest?: DerivedDataset | null;
}

// 資料集 browser — one platform image with boxes in RELATIVE (0-1) top-left
// coords, ready for an SVG overlay regardless of thumbnail scale.
export interface DatasetBox {
  x: number;
  y: number;
  w: number;
  h: number;
  cls: number;
}

export interface DatasetImage {
  id: string;
  /** Per-image key for ALL write operations (labels / split / delete). */
  hash: string;
  thumbnail_url: string; // signed, expiring — expect onError fallbacks
  /** Full-resolution signed URL (for the annotation editor); may be absent. */
  full_url?: string | null;
  width: number;
  height: number;
  split: string;
  boxes: DatasetBox[];
}

export interface DatasetImagesPage {
  total: number;
  has_more: boolean;
  counts: Record<string, number>; // all / train / val / test
  images: DatasetImage[];
}

export interface UploadFilesResult {
  uploaded: number;
  failed: { name: string; reason: string }[];
  duplicates: DuplicateNote[];
}

// ── Auth session ─────────────────────────────────────────────────────────────
//
// The whole app is gated behind login. A signed token (issued by /api/auth/login)
// lives in sessionStorage and is the single source of truth for identity + role.
// Every authenticated request carries it — via an Authorization header (fetch)
// or a ?token= query param (for <img>/MJPEG tags that cannot set headers).

export type Role = "admin" | "user";

const TOKEN_KEY = "auth_token";

interface TokenPayload {
  sub?: number;
  role?: Role;
  exp?: number;
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(TOKEN_KEY);
}

function decodeToken(token: string | null): TokenPayload | null {
  if (!token) return null;
  try {
    let b64 = token.split(".")[0].replace(/-/g, "+").replace(/_/g, "/");
    b64 += "=".repeat((4 - (b64.length % 4)) % 4);
    return JSON.parse(atob(b64)) as TokenPayload;
  } catch {
    return null;
  }
}

/** The current valid session, or null if unauthenticated / expired. */
export function getSession(): { token: string; role: Role } | null {
  const token = getToken();
  const payload = decodeToken(token);
  if (!token || !payload || (payload.exp ?? 0) * 1000 <= Date.now()) {
    if (token) clearSession();
    return null;
  }
  return { token, role: payload.role === "admin" ? "admin" : "user" };
}

export function clearSession(): void {
  if (typeof window !== "undefined") window.sessionStorage.removeItem(TOKEN_KEY);
}

/** Broadcast a forced logout (expired/invalid token) so the app returns to login. */
function onUnauthorized(): void {
  clearSession();
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event("auth-expired"));
  }
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const token = getToken();
  return { ...(extra ?? {}), ...(token ? { Authorization: `Bearer ${token}` } : {}) };
}

/**
 * Turn a FastAPI error body into a display string. `detail` may be a plain
 * string (HTTPException) OR an array of {loc,msg,type} (422 validation) — the
 * latter must never be rendered raw or the UI shows "[object Object]".
 */
function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const msg = detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : null))
        .filter(Boolean)
        .join("；");
      if (msg) return msg;
    }
  }
  return fallback;
}

/** fetch() that attaches the auth token and treats 401 as a session expiry. */
async function authFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const res = await fetch(url, {
    ...init,
    headers: authHeaders(init.headers as Record<string, string> | undefined),
  });
  if (res.status === 401) {
    onUnauthorized();
    throw new Error("Unauthorized");
  }
  return res;
}

/** Append the token as a query param — for URLs used as <img>/stream src. */
function withToken(url: string): string {
  const token = getToken();
  if (!token) return url;
  return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
}

export async function login(code: string): Promise<Role> {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(errorMessage(err, "登入失敗"));
  }
  const data = await res.json();
  if (typeof window !== "undefined") {
    window.sessionStorage.setItem(TOKEN_KEY, data.token);
  }
  return data.role === "admin" ? "admin" : "user";
}

export function logout(): void {
  clearSession();
}

// ── Fetchers (for SWR) ──────────────────────────────────────────────────────

export const fetcher = <T>(url: string): Promise<T> =>
  authFetch(url).then((res) => {
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json() as Promise<T>;
  });

// ── UI helpers ───────────────────────────────────────────────────────────────

/** Display name: collapse to one value when name equals camera_id. */
export function cameraLabel(cam: Pick<Camera, "camera_id" | "name">): string {
  const name = cam.name?.trim();
  if (name && name !== cam.camera_id) return name;
  return cam.camera_id;
}

export function cameraHasDistinctName(cam: Pick<Camera, "camera_id" | "name">): boolean {
  const name = cam.name?.trim();
  return Boolean(name && name !== cam.camera_id);
}

// ── API helpers ─────────────────────────────────────────────────────────────

async function apiRequest(
  path: string,
  init: RequestInit = {},
  fallback?: string,
): Promise<Response> {
  const res = await authFetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(errorMessage(err, fallback ?? `API error: ${res.status}`));
  }
  return res;
}

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  fallback?: string,
): Promise<T> {
  return (await apiRequest(path, init, fallback)).json();
}

/** Override (true/false) or reset (null) whether users can see a detection. */
export async function setDetectionVisibility(
  id: number,
  visible: boolean | null,
): Promise<{ id: number; user_visible: boolean | null; visible_to_users: boolean }> {
  return requestJson(`/api/admin/detections/${id}/visibility`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ visible }),
  }, "更新可見性失敗");
}

/** Make all currently hidden detections visible to non-admin users. */
export async function setDetectionsVisibilityBulk(
  visible: boolean,
): Promise<{ updated: number; visible: boolean | null; detection_ids: number[] }> {
  return requestJson(`/api/admin/detections/visibility-bulk`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ visible, only_hidden: true }),
  }, "批次更新可見性失敗");
}

/** Protect one detection from both delete paths, or release it. */
export async function setDetectionLock(
  id: number,
  locked: boolean,
): Promise<{ id: number; locked: boolean }> {
  return requestJson(`/api/admin/detections/${id}/lock`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ locked }),
  }, "更新鎖定失敗");
}

/** Lock/unlock many detections in one request. The endpoint also accepts
 *  camera/date/confidence filters (that is how the pre-cutoff backfill runs,
 *  reaching records this tab cannot display); the UI only needs the id form. */
export async function setDetectionsLockBulk(
  ids: number[],
  locked: boolean,
): Promise<{ updated: number; locked: boolean; detection_ids: number[] }> {
  return requestJson(`/api/admin/detections/lock-bulk`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ detection_ids: ids, locked }),
  }, "更新鎖定失敗");
}

/** Store one ReID research verdict. This never assigns animal_id or edits a detection. */
export async function labelReIDPair(
  detectionAId: number,
  detectionBId: number,
  label: ReIDLabel,
  note: string,
): Promise<{
  detection_a_id: number;
  detection_b_id: number;
  label: ReIDLabel;
  note: string;
  updated_at: string;
}> {
  return requestJson(`/api/admin/reid/labels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      detection_a_id: detectionAId,
      detection_b_id: detectionBId,
      label,
      note,
    }),
  }, "儲存 ReID 標記失敗");
}

export async function addCamera(body: CameraCreate): Promise<Camera> {
  return requestJson(`/api/cameras`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function streamUrl(cameraId: string): string {
  return withToken(`${API_BASE}/api/stream/${cameraId}/mjpeg`);
}

export function snapshotUrl(filename: string): string {
  // Preserve path separators. Encoding the slash as `%2F` is handled
  // differently by browser/CDN combinations and can miss FastAPI's path
  // converter.
  const encoded = filename.split("/").map((part) => encodeURIComponent(part)).join("/");
  return withToken(`${API_BASE}/api/snapshots/${encoded}`);
}

/** Fallback URL served from the frontend's own static files (public/snapshots/). */
export function snapshotFallbackUrl(filename: string): string {
  const encoded = filename.split("/").map((part) => encodeURIComponent(part)).join("/");
  return `/snapshots/${encoded}`;
}

export function detectionFramesUrl(detectionId: number): string {
  return withToken(`${API_BASE}/api/detections/${detectionId}/frames`);
}

export async function createWebhook(body: WebhookCreate): Promise<Webhook> {
  return requestJson(`/api/webhooks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function updateWebhook(
  id: number,
  body: WebhookUpdate
): Promise<Webhook> {
  return requestJson(`/api/webhooks/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function deleteWebhook(id: number): Promise<void> {
  await apiRequest(`/api/webhooks/${id}`, {
    method: "DELETE",
  });
}

export async function testWebhook(
  id: number
): Promise<{ status: string; http_status: number }> {
  return requestJson(`/api/webhooks/${id}/test`, {
    method: "POST",
  });
}

// ── Admin ───────────────────────────────────────────────────────────────────

export function downloadImagesUrl(params?: {
  camera_id?: string;
  date_from?: string;
  date_to?: string;
}): string {
  const query = new URLSearchParams();
  const token = getToken();
  if (token) query.set("token", token);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v) query.set(k, v);
    });
  }
  return `${API_BASE}/api/admin/download-images?${query.toString()}`;
}

/** Download the original snapshots for the selected detections as a ZIP.
 *  POST (not the ?token= GET link) — id lists are arbitrary/large; carries the
 *  Bearer token via authFetch, then saves the blob via a temporary object URL. */
export async function downloadSelectedImages(ids: number[]): Promise<void> {
  const res = await apiRequest(`/api/admin/download-selected`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  }, "下載失敗");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "primate-watch-selected.zip";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function deleteSelectedDetections(
  ids: number[]
): Promise<{
  deleted_detections: number;
  deleted_files: number;
  /** Locked ids the server refused. Present even when the client already
   *  filtered them out — it is the authoritative answer, not the optimistic one. */
  refused_locked?: number[];
}> {
  return requestJson(`/api/admin/delete-selected`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  }, "刪除失敗");
}

export async function clearSnapshots(): Promise<{
  deleted_files: number;
  deleted_detections: number;
  /** Locked records the sweep refused, along with their blobs. */
  kept_locked?: number;
}> {
  return requestJson(`/api/admin/clear-snapshots`, {
    method: "DELETE",
  }, "清除失敗");
}

export async function updateCameraAdmin(
  cameraId: string,
  body: CameraUpdate
): Promise<Camera> {
  return requestJson(
    `/api/admin/cameras/${encodeURIComponent(cameraId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    "更新攝影機失敗",
  );
}

export async function deleteCameraAdmin(
  cameraId: string
): Promise<CameraDeleteResult> {
  return requestJson(
    `/api/admin/cameras/${encodeURIComponent(cameraId)}`,
    {
      method: "DELETE",
    },
    "移除攝影機失敗",
  );
}

/** Forget an inference node. A live node re-registers on its next heartbeat. */
export async function deleteNode(hostname: string): Promise<void> {
  await apiRequest(
    `/api/admin/nodes/${encodeURIComponent(hostname)}`,
    { method: "DELETE" },
    "移除節點失敗",
  );
}

// ── Admin: user management (Phase 2) ─────────────────────────────────────────

export async function createUser(body: UserCreate): Promise<User> {
  return requestJson(`/api/admin/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, "新增帳號失敗");
}

export async function updateUser(id: number, body: UserUpdate): Promise<User> {
  return requestJson(`/api/admin/users/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, "更新帳號失敗");
}

export async function deleteUser(id: number): Promise<void> {
  await apiRequest(`/api/admin/users/${id}`, {
    method: "DELETE",
  }, "刪除帳號失敗");
}

// ── Admin: model training (Ultralytics Platform) ─────────────────────────────

/** Upload the RAW snapshots of the selected detections into the training
 *  dataset. Images arrive UNLABELED on the Platform (= background negatives
 *  until boxes are drawn in the Platform UI). */
export async function addToTrainingDataset(
  detectionIds: number[]
): Promise<AddToDatasetResult> {
  return requestJson(`/api/admin/training/add-to-dataset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ detection_ids: detectionIds }),
  }, "加入訓練資料集失敗");
}

/** Start a cloud training run — platform response passthrough (modelId etc.). */
export async function startTraining(
  body: TrainingStartBody
): Promise<Record<string, unknown>> {
  return requestJson(`/api/admin/training/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, "啟動訓練失敗");
}

/** Dispatch the accuracy-gated promotion workflow (棘輪雙檢) for a model.
 *
 *  `force` ships a model the ratchet rejected: the gate still runs and is still
 *  reported, only the abort is skipped, and the baseline is never refreshed. An
 *  eval that failed to run at all still blocks — force overrides a low score,
 *  not a missing measurement. */
export async function promoteModel(
  modelId: string,
  force = false
): Promise<{ message: string; model_id: string; force?: boolean }> {
  return requestJson(`/api/admin/training/promote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, force }),
  }, "觸發上板流程失敗");
}

/** Dispatch the edge-runner evaluation workflow (本站評測 — scores only,
 *  never touches the deployed model). Results appear on the models list. */
export async function evaluateModel(
  modelId: string
): Promise<{ message: string; model_id: string }> {
  return requestJson(`/api/admin/training/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  }, "觸發本站評測失敗");
}

/** Every dataset on the Platform, newest first. */
export interface PlatformDataset {
  id: string;
  name: string;
  slug: string;
  images: number;
  updatedAt: string | null;
  is_default: boolean;
}

export async function getPlatformDatasets(): Promise<PlatformDataset[]> {
  return requestJson(`/api/admin/training/datasets`, {}, "載入資料集清單失敗");
}

/** One page of a dataset (thumbnails + relative-coord boxes). `dataset` is the
 *  slug being browsed; every write below takes the same one, so an edit can
 *  never land in a dataset other than the one on screen. */
export async function getDatasetImages(
  split: string | null,
  offset: number,
  limit: number,
  dataset?: string | null
): Promise<DatasetImagesPage> {
  const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
  if (split && split !== "all") query.set("split", split);
  if (dataset) query.set("dataset", dataset);
  return requestJson(
    `/api/admin/training/dataset-images?${query.toString()}`,
    {},
    "載入資料集失敗",
  );
}

/** Replace one image's annotation boxes on the Platform (source of truth).
 *  Boxes are in the dashboard's top-left relative shape; the backend converts
 *  back to the Platform's YOLO center format. */
export async function updateImageLabels(
  hash: string,
  boxes: DatasetBox[],
  dataset?: string | null
): Promise<{ saved: boolean; label_count: number }> {
  return requestJson(
    `/api/admin/training/dataset-images/${encodeURIComponent(hash)}/labels`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes, dataset }),
    },
    "儲存標註失敗",
  );
}

/** Move images between splits. Each item carries the image's CURRENT split
 *  (the Platform keys source copies by hash+split); `split` is the target.
 *  Moving into test/val changes the ratchet benchmark — warn first. */
export async function moveImagesSplit(
  images: { hash: string; split: string }[],
  split: string,
  dataset?: string | null
): Promise<{ moved: number; skipped: number; target_split: string }> {
  return requestJson(`/api/admin/training/dataset-images/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images, split, dataset }),
  }, "移動切分失敗");
}

/** Delete images from the Platform dataset (bulk, by hash). Destructive. */
export async function deleteDatasetImages(
  hashes: string[],
  dataset?: string | null
): Promise<{ deleted: number }> {
  return requestJson(`/api/admin/training/dataset-images/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hashes, dataset }),
  }, "刪除影像失敗");
}

/** Start rebuilding the deduplicated 訓練集 from the master. */
export async function regenerateDerivedDataset(): Promise<RegenerateState> {
  return requestJson(`/api/admin/training/regenerate-dataset`, {
    method: "POST",
  }, "重新生成訓練集失敗");
}

/** Start a cloud training run against the newest deduplicated 訓練集. */
export async function trainOnDerived(
  body: TrainingStartBody
): Promise<Record<string, unknown>> {
  return requestJson(`/api/admin/training/train-on-derived`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, "啟動訓練失敗");
}

/** Drag-and-drop upload: push local JPG/PNG files into the training dataset.
 *  They arrive UNLABELED (= background negatives until boxes are drawn). */
export async function uploadTrainingFiles(files: File[]): Promise<UploadFilesResult> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return requestJson(`/api/admin/training/upload-files`, {
    method: "POST",
    body: form,
  }, "上傳圖片失敗");
}
