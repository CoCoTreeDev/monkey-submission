"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MessageBanner, useFlashMessage } from "@/components/status-message";
import useSWR from "swr";
import { format } from "date-fns";
import { duplicateUploadText } from "@/lib/utils";
import {
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Copy,
  Database,
  FolderInput,
  Gauge,
  ImageOff,
  Loader2,
  Play,
  RefreshCw,
  Rocket,
  Save,
  ShieldAlert,
  Trash2,
  Undo2,
  UploadCloud,
  X,
  XCircle,
} from "lucide-react";
import {
  API_BASE,
  fetcher,
  deleteDatasetImages,
  evaluateModel,
  getDatasetImages,
  getPlatformDatasets,
  type PlatformDataset,
  moveImagesSplit,
  promoteModel,
  regenerateDerivedDataset,
  startTraining,
  trainOnDerived,
  updateImageLabels,
  uploadTrainingFiles,
  type DatasetBox,
  type DatasetImage,
  type EvalScore,
  type RegenerateState,
  type TrainingModel,
  type TrainingStatusPayload,
} from "@/lib/api";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";

/** 訓練 — model-lifecycle control plane backed by the Ultralytics Platform.
 *  The Platform is the single source of truth; this tab lists its models,
 *  starts cloud training runs, shows live training progress (1 s polling for
 *  running models), browses the dataset with rendered annotations, compares
 *  versions against OUR gate scores (本站評測, from the edge runner), and
 *  dispatches evaluation / accuracy-gated promotion workflows. */
export function TrainingPanel() {
  const {
    data: models,
    isLoading,
    mutate,
  } = useSWR<TrainingModel[]>(`${API_BASE}/api/admin/training/models`, fetcher, {
    refreshInterval: 10000,
  });

  // Which dataset a new run trains on: the master, or the deduplicated 訓練集.
  const [startTarget, setStartTarget] = useState<"master" | "derived" | null>(null);
  const [promoteTarget, setPromoteTarget] = useState<{
    model: TrainingModel;
    force: boolean;
  } | null>(null);
  const [message, flash] = useFlashMessage(6000);

  // 評測中… tracking: modelId → snapshot of its eval payload at dispatch time.
  // The row keeps showing the spinner until the models list (10 s poll) comes
  // back with a DIFFERENT eval payload — no timezone math, just change.
  const [evaluating, setEvaluating] = useState<Record<string, string>>({});
  const evaluatingModels = new Set(
    Object.entries(evaluating)
      .filter(([id, snapshot]) => {
        const m = models?.find((x) => x.id === id);
        return !!m && JSON.stringify(m.eval ?? null) === snapshot;
      })
      .map(([id]) => id)
  );

  const handleEvaluate = async (m: TrainingModel) => {
    setEvaluating((prev) => ({ ...prev, [m.id]: JSON.stringify(m.eval ?? null) }));
    try {
      const res = await evaluateModel(m.id);
      flash("success", res.message || "已觸發本站評測");
    } catch (err: unknown) {
      setEvaluating((prev) => {
        const next = { ...prev };
        delete next[m.id];
        return next;
      });
      flash("error", err instanceof Error ? err.message : "觸發本站評測失敗");
    }
  };

  // At most 2 concurrent 1 s pollers — remaining training rows fall back to 10 s.
  const liveIds = (models ?? [])
    .filter((m) => m.status === "training")
    .slice(0, 2)
    .map((m) => m.id);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <BrainCircuit className="h-4 w-4 text-muted-foreground" />
        <h2 className="text-sm font-semibold">模型訓練</h2>
        {models && (
          <span className="text-xs text-muted-foreground">
            {models.length} 個模型 · 資料來源 Ultralytics Platform
          </span>
        )}
        <button
          onClick={() => setStartTarget("master")}
          className="ml-auto flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-xs font-medium text-white hover:bg-primary/90"
        >
          <Play className="h-3.5 w-3.5" />
          開始訓練
        </button>
      </div>

      <MessageBanner message={message} />

      {/* Model list */}
      <div className="rounded-lg border bg-white">
        <div className="border-b px-5 py-3">
          <h3 className="text-sm font-semibold">模型列表</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            「評測」在邊緣節點以部署設定（imgsz 1600、fp16、MPS）跑棘輪取得本站
            F1；「上板」跑棘輪雙檢（test+val），通過才會自動部署；「強制上板」照跑照記錄但不擋，
            且不會更動 baseline
          </p>
        </div>
        {isLoading && !models ? (
          <div className="flex items-center gap-2 p-5 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            載入中…
          </div>
        ) : !models?.length ? (
          <div className="p-8 text-center text-sm text-muted-foreground">
            專案內尚無模型 — 點「開始訓練」啟動第一次雲端訓練
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>名稱</TableHead>
                  <TableHead>狀態</TableHead>
                  <TableHead className="text-right">Precision</TableHead>
                  <TableHead className="text-right">Recall</TableHead>
                  <TableHead className="text-right">mAP50</TableHead>
                  <TableHead>本站評測 F1</TableHead>
                  <TableHead>建立時間</TableHead>
                  <TableHead className="text-right">動作</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((m) => (
                  <TableRow key={m.id}>
                    <TableCell>
                      <span className="font-medium">{m.name || "（未命名）"}</span>
                      <span className="ml-2 font-mono text-[10px] text-muted-foreground">
                        {m.id.slice(0, 8)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <StatusCell model={m} live={liveIds.includes(m.id)} />
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs">
                      {formatMetric(m.metrics.precision)}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs">
                      {formatMetric(m.metrics.recall)}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs">
                      {formatMetric(m.metrics.map50)}
                    </TableCell>
                    <TableCell>
                      <EvalCell model={m} isEvaluating={evaluatingModels.has(m.id)} />
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {formatCreatedAt(m.createdAt)}
                    </TableCell>
                    <TableCell className="text-right">
                      {m.status === "completed" && (
                        <div className="flex items-center justify-end gap-1.5">
                          {evaluatingModels.has(m.id) ? (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 text-xs text-muted-foreground">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              評測中…
                            </span>
                          ) : (
                            <button
                              onClick={() => handleEvaluate(m)}
                              className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs font-medium text-foreground hover:bg-muted"
                              title="在邊緣節點跑棘輪評測（test+val），只記分數、不動線上模型"
                            >
                              <Gauge className="h-3 w-3" />
                              評測
                            </button>
                          )}
                          <button
                            onClick={() => setPromoteTarget({ model: m, force: false })}
                            className="inline-flex items-center gap-1 rounded-md border border-primary/40 px-2.5 py-1 text-xs font-medium text-primary hover:bg-primary/5"
                            title="觸發棘輪精度雙檢，通過後自動部署"
                          >
                            <Rocket className="h-3 w-3" />
                            上板
                          </button>
                          <button
                            onClick={() => setPromoteTarget({ model: m, force: true })}
                            className="inline-flex items-center gap-1 rounded-md border border-destructive/40 px-2.5 py-1 text-xs font-medium text-destructive hover:bg-destructive/5"
                            title="棘輪沒過也照樣部署。評測照跑照記錄，但不擋；baseline 不會更動"
                          >
                            <ShieldAlert className="h-3 w-3" />
                            強制上板
                          </button>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      {/* Version comparison — OUR gate scores, not the platform's metrics */}
      {models && <ComparisonStrip models={models} />}

      {/* Dataset browser */}
      <DatasetBrowser />

      {/* 去重與訓練集 — the derived, deduplicated dataset training runs on */}
      <DerivedDatasetCard onTrain={() => setStartTarget("derived")} />

      {startTarget && (
        <StartTrainingDialog
          target={startTarget}
          onClose={() => setStartTarget(null)}
          onStarted={(modelId) => {
            setStartTarget(null);
            flash(
              "success",
              modelId
                ? `訓練已啟動（模型 ID：${modelId}）— 進度會顯示在列表中`
                : "訓練已啟動 — 進度會顯示在列表中"
            );
            mutate();
          }}
        />
      )}

      {promoteTarget && (
        <PromoteConfirmDialog
          model={promoteTarget.model}
          force={promoteTarget.force}
          onClose={() => setPromoteTarget(null)}
          onDone={(text, ok) => {
            setPromoteTarget(null);
            flash(ok ? "success" : "error", text);
          }}
        />
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatMetric(v: number | null): string {
  return v == null ? "—" : v.toFixed(3);
}

function formatCreatedAt(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? "—" : format(d, "yyyy/MM/dd HH:mm");
}

function formatElapsed(startedAt: string): string | null {
  const t = new Date(startedAt).getTime();
  if (isNaN(t)) return null;
  let s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  const h = Math.floor(s / 3600);
  s -= h * 3600;
  const m = Math.floor(s / 60);
  s -= m * 60;
  if (h > 0) return `${h} 時 ${m} 分`;
  if (m > 0) return `${m} 分 ${s} 秒`;
  return `${s} 秒`;
}

/** Pull a human progress string out of the platform's training-status
 *  passthrough, whatever shape it takes. Null = no usable progress info.
 *  Fallback only — the structured path reads data.job.progress directly. */
function extractProgress(data: Record<string, unknown> | undefined): string | null {
  if (!data) return null;
  const nested = data.training ?? data.job;
  const d = (
    nested && typeof nested === "object" ? nested : data
  ) as Record<string, unknown>;
  const progress = d.progress ?? d.percent;
  if (typeof progress === "number") {
    const pct = progress <= 1 ? progress * 100 : progress;
    return `${Math.round(Math.min(100, Math.max(0, pct)))}%`;
  }
  const epoch = d.epoch ?? d.currentEpoch ?? d.current_epoch;
  const epochs = d.epochs ?? d.totalEpochs ?? d.total_epochs;
  if (typeof epoch === "number" && typeof epochs === "number" && epochs > 0) {
    return `${epoch}/${epochs} epochs`;
  }
  return null;
}

const STATUS_STYLE: Record<TrainingModel["status"], string> = {
  training: "bg-blue-50 text-blue-700",
  completed: "bg-green-50 text-green-700",
  failed: "bg-red-50 text-red-600",
  idle: "bg-gray-100 text-gray-500",
};

const STATUS_LABEL: Record<TrainingModel["status"], string> = {
  training: "訓練中",
  completed: "已完成",
  failed: "失敗",
  idle: "閒置",
};

/** Status badge + live progress bar. Training rows poll the status endpoint —
 *  the first 2 at 1 Hz (`live`), the rest at 10 s; the backend's 1 s TTL cache
 *  keeps N viewers from multiplying platform calls. Other rows don't poll. */
function StatusCell({ model, live }: { model: TrainingModel; live: boolean }) {
  const isTraining = model.status === "training";
  const { data } = useSWR<TrainingStatusPayload>(
    isTraining ? `${API_BASE}/api/admin/training/status/${model.id}` : null,
    fetcher,
    { refreshInterval: live ? 1000 : 10000 }
  );
  const prog = isTraining ? data?.job?.progress : undefined;
  let pct: number | null = null;
  if (typeof prog?.percentage === "number") {
    pct = prog.percentage;
  } else if (
    typeof prog?.currentEpoch === "number" &&
    typeof prog?.totalEpochs === "number" &&
    prog.totalEpochs > 0
  ) {
    pct = (prog.currentEpoch / prog.totalEpochs) * 100;
  }
  if (pct != null) pct = Math.min(100, Math.max(0, pct));
  const fallback =
    isTraining && pct == null
      ? extractProgress(data as Record<string, unknown> | undefined)
      : null;
  const elapsed = prog?.startedAt ? formatElapsed(prog.startedAt) : null;

  return (
    <div className="space-y-1">
      <span
        className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ${STATUS_STYLE[model.status]}`}
      >
        {isTraining && (
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500" />
        )}
        {STATUS_LABEL[model.status]}
        {fallback && <span className="font-mono tabular-nums">{fallback}</span>}
      </span>
      {isTraining && pct != null && (
        <div className="w-40 space-y-0.5">
          <div className="h-1.5 overflow-hidden rounded-full bg-neutral-200">
            <div
              className="h-full rounded-full bg-blue-500 transition-[width] duration-700"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="flex justify-between font-mono text-[10px] tabular-nums text-muted-foreground">
            <span>
              {typeof prog?.currentEpoch === "number" &&
              typeof prog?.totalEpochs === "number"
                ? `${prog.currentEpoch}/${prog.totalEpochs} epochs`
                : ""}
            </span>
            <span>{pct.toFixed(1)}%</span>
          </div>
          {elapsed && (
            <p className="text-[10px] text-muted-foreground">經過 {elapsed}</p>
          )}
        </div>
      )}
    </div>
  );
}

// ── 本站評測 cell + version comparison ───────────────────────────────────────

function EvalScoreLine({ label, score }: { label: string; score?: EvalScore }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="w-7 text-[10px] text-muted-foreground">{label}</span>
      {score ? (
        <>
          <span className="font-mono text-xs tabular-nums">{score.f1.toFixed(3)}</span>
          <span
            className={`inline-flex items-center gap-0.5 rounded-full px-1.5 py-px text-[9px] font-medium ${
              score.passed ? "bg-green-50 text-green-700" : "bg-red-50 text-red-600"
            }`}
          >
            {score.passed ? (
              <CheckCircle2 className="h-2.5 w-2.5" />
            ) : (
              <XCircle className="h-2.5 w-2.5" />
            )}
            {score.passed ? "通過" : "未通過"}
          </span>
        </>
      ) : (
        <span className="text-xs text-muted-foreground">—</span>
      )}
    </div>
  );
}

function EvalCell({
  model,
  isEvaluating,
}: {
  model: TrainingModel;
  isEvaluating: boolean;
}) {
  if (isEvaluating && !model.eval) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        評測中…
      </span>
    );
  }
  if (!model.eval) return <span className="text-xs text-muted-foreground">—</span>;
  return (
    <div className="space-y-0.5">
      <EvalScoreLine label="test" score={model.eval.test} />
      <EvalScoreLine label="val" score={model.eval.val} />
    </div>
  );
}

/** Horizontal bar per evaluated model (test F1) with the ratchet baseline as
 *  a reference line — different versions become visually comparable. The axis
 *  is zoomed to the score range (F1s cluster near 0.9+), stated explicitly. */
function ComparisonStrip({ models }: { models: TrainingModel[] }) {
  const rows = models
    .filter((m): m is TrainingModel & { eval: { test: EvalScore } } =>
      Boolean(m.eval?.test)
    )
    .map((m) => ({
      id: m.id,
      name: m.name || m.id.slice(0, 8),
      f1: m.eval.test.f1,
      baseline: m.eval.test.baseline_f1 > 0 ? m.eval.test.baseline_f1 : null,
      passed: m.eval.test.passed,
    }));
  if (rows.length === 0) return null;

  const values = rows.flatMap((r) => [r.f1, ...(r.baseline != null ? [r.baseline] : [])]);
  const lo = Math.max(0, Math.min(...values) - 0.02);
  const hi = Math.min(1, Math.max(...values) + 0.005);
  const span = Math.max(hi - lo, 0.001);
  const pos = (v: number) => Math.min(100, Math.max(0, ((v - lo) / span) * 100));
  const baseline = rows.find((r) => r.baseline != null)?.baseline ?? null;

  return (
    <div className="rounded-lg border bg-white">
      <div className="flex flex-wrap items-center gap-2 border-b px-5 py-3">
        <h3 className="text-sm font-semibold">版本比較（本站 test F1）</h3>
        <span className="text-xs text-muted-foreground">
          軸範圍 {lo.toFixed(3)}–{hi.toFixed(3)}（非從 0 起算）
        </span>
        {baseline != null && (
          <span className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="inline-block h-3 w-0.5 bg-amber-500" />
            棘輪基準線 {baseline.toFixed(4)}
          </span>
        )}
      </div>
      <div className="space-y-2.5 p-5">
        {rows.map((r) => (
          <div key={r.id} className="flex items-center gap-3">
            <span className="w-36 truncate text-xs" title={r.name}>
              {r.name}
            </span>
            <div className="relative h-5 flex-1 overflow-hidden rounded bg-neutral-100">
              <div
                className={`h-full rounded transition-[width] ${
                  r.passed ? "bg-green-500/70" : "bg-red-400/70"
                }`}
                style={{ width: `${pos(r.f1)}%` }}
              />
              {r.baseline != null && (
                <div
                  className="absolute bottom-0 top-0 w-0.5 bg-amber-500"
                  style={{ left: `${pos(r.baseline)}%` }}
                  title={`棘輪基準線 ${r.baseline.toFixed(4)}`}
                />
              )}
            </div>
            <span className="w-14 text-right font-mono text-xs tabular-nums">
              {r.f1.toFixed(4)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── 資料集 browser ───────────────────────────────────────────────────────────

const SPLITS = ["all", "train", "val", "test"] as const;
type SplitFilter = (typeof SPLITS)[number];
const SPLIT_LABEL: Record<SplitFilter, string> = {
  all: "全部",
  train: "train",
  val: "val",
  test: "test",
};
const PAGE_SIZE = 24;
const UPLOAD_MAX = 50;

type UploadItem = {
  name: string;
  status: "uploading" | "done" | "failed";
  reason?: string;
};

function DatasetBrowser() {
  // Which dataset is on screen. ULTRALYTICS_DATASET pins ONE master, which was
  // right when the project had one dataset; it now has six benchmarks plus
  // derived bundles, and the pin still named the July set — 7th newest — so the
  // browser showed data the shipping model was not trained on. Empty string =
  // the backend default, used until the list arrives.
  const [dsList, setDsList] = useState<PlatformDataset[]>([]);
  const [dataset, setDataset] = useState<string>("");
  const [split, setSplit] = useState<SplitFilter>("all");
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [counts, setCounts] = useState<Record<string, number> | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [viewing, setViewing] = useState<DatasetImage | null>(null);
  // Result banner for editing operations (save / move / delete).
  const [opNote, flashOp] = useFlashMessage(5000);
  // Monotonic request id — a stale page response must never clobber state.
  const reqRef = useRef(0);

  const load = useCallback(async (s: SplitFilter, offset: number, ds: string) => {
    const req = ++reqRef.current;
    setLoading(true);
    setError("");
    try {
      const page = await getDatasetImages(s === "all" ? null : s, offset, PAGE_SIZE, ds || null);
      if (reqRef.current !== req) return;
      setCounts(page.counts);
      setHasMore(page.has_more);
      setImages((prev) => (offset === 0 ? page.images : [...prev, ...page.images]));
    } catch (err: unknown) {
      if (reqRef.current === req) {
        setError(err instanceof Error ? err.message : "載入資料集失敗");
      }
    } finally {
      if (reqRef.current === req) setLoading(false);
    }
  }, []);

  // Default to the most recently updated dataset rather than a hardcoded name:
  // pinning a different name would only move the staleness somewhere else.
  useEffect(() => {
    let cancelled = false;
    getPlatformDatasets()
      .then((rows) => {
        if (cancelled || rows.length === 0) return;
        setDsList(rows);
        setDataset((cur) => cur || rows[0].slug);
      })
      .catch(() => {
        /* the browser still works on the backend default */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setImages([]);
    load(split, 0, dataset);
  }, [split, dataset, load]);

  // Local state sync after Platform writes — grid updates without a reload
  // (the server-side cache is invalidated, so the next page load is fresh).
  const applyBoxes = (hash: string, boxes: DatasetBox[]) => {
    setImages((prev) => prev.map((i) => (i.hash === hash ? { ...i, boxes } : i)));
    setViewing((v) => (v && v.hash === hash ? { ...v, boxes } : v));
  };
  const applySplit = (hash: string, from: string, to: string) => {
    setImages((prev) =>
      split === "all"
        ? prev.map((i) => (i.hash === hash ? { ...i, split: to } : i))
        : prev.filter((i) => i.hash !== hash)
    );
    setCounts((prev) =>
      prev
        ? {
            ...prev,
            [from]: Math.max(0, (prev[from] ?? 0) - 1),
            [to]: (prev[to] ?? 0) + 1,
          }
        : prev
    );
    setViewing((v) => (v && v.hash === hash ? { ...v, split: to } : v));
  };
  const applyDelete = (img: DatasetImage) => {
    setImages((prev) => prev.filter((i) => i.hash !== img.hash));
    setCounts((prev) =>
      prev
        ? {
            ...prev,
            all: Math.max(0, (prev.all ?? 0) - 1),
            [img.split]: Math.max(0, (prev[img.split] ?? 0) - 1),
          }
        : prev
    );
    setViewing((v) => (v && v.hash === img.hash ? null : v));
  };

  // Right-click context menu on grid thumbnails: 移至 split / 刪除影像.
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; img: DatasetImage } | null>(null);
  useEffect(() => {
    if (!ctxMenu) return;
    const close = () => setCtxMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setCtxMenu(null);
    };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", onKey);
    };
  }, [ctxMenu]);

  const requestMove = async (img: DatasetImage, target: string) => {
    setCtxMenu(null);
    if (
      (target === "test" || target === "val") &&
      !window.confirm(
        `移至 ${target} 會改變棘輪評測集，基準線需要重定錨。確定要移動嗎？`
      )
    ) {
      return;
    }
    try {
      await moveImagesSplit([{ hash: img.hash, split: img.split }], target, dataset);
      applySplit(img.hash, img.split, target);
      flashOp("success", `已移至 ${target} — 已同步至 Platform`);
    } catch (err: unknown) {
      flashOp("error", err instanceof Error ? err.message : "移動切分失敗");
    }
  };

  const requestDelete = async (img: DatasetImage) => {
    setCtxMenu(null);
    if (!window.confirm("確定要刪除這張影像？會從 Platform 資料集移除原圖與標註。")) return;
    if (!window.confirm("再次確認：刪除後無法復原。")) return;
    try {
      await deleteDatasetImages([img.hash], dataset);
      applyDelete(img);
      flashOp("success", "已刪除影像 — 已同步至 Platform");
    } catch (err: unknown) {
      flashOp("error", err instanceof Error ? err.message : "刪除影像失敗");
    }
  };

  // Drag & drop upload
  const [dragOver, setDragOver] = useState(false);
  const [uploadItems, setUploadItems] = useState<UploadItem[]>([]);
  const [uploadNote, setUploadNote] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadingCount = uploadItems.filter((u) => u.status === "uploading").length;

  const handleFiles = async (list: FileList | File[]) => {
    if (uploadingCount > 0) return; // one batch at a time
    const files = Array.from(list).filter(
      (f) =>
        ["image/jpeg", "image/png"].includes(f.type.toLowerCase()) ||
        /\.(jpe?g|png)$/i.test(f.name)
    );
    setUploadNote(null);
    if (files.length === 0) {
      setUploadNote({ type: "error", text: "僅支援 JPG / PNG 圖片" });
      return;
    }
    if (files.length > UPLOAD_MAX) {
      setUploadNote({ type: "error", text: `一次最多上傳 ${UPLOAD_MAX} 張圖片` });
      return;
    }
    setUploadItems(files.map((f) => ({ name: f.name, status: "uploading" as const })));
    try {
      const result = await uploadTrainingFiles(files);
      const failedByName = new Map(result.failed.map((f) => [f.name, f.reason]));
      setUploadItems(
        files.map((f) =>
          failedByName.has(f.name)
            ? {
                name: f.name,
                status: "failed" as const,
                reason: failedByName.get(f.name),
              }
            : { name: f.name, status: "done" as const }
        )
      );
      setUploadNote({
        type: result.uploaded > 0 ? "success" : "error",
        text: `已上傳 ${result.uploaded} 張（未標註＝背景負樣本，正樣本請到 Platform 補框）${
          result.failed.length > 0 ? `，${result.failed.length} 張失敗` : ""
        }${duplicateUploadText(result.duplicates)}`,
      });
    } catch (err: unknown) {
      const reason = err instanceof Error ? err.message : "上傳圖片失敗";
      setUploadItems(
        files.map((f) => ({ name: f.name, status: "failed" as const, reason }))
      );
      setUploadNote({ type: "error", text: reason });
    }
  };

  return (
    <div className="rounded-lg border bg-white">
      <div className="flex flex-wrap items-center gap-2 border-b px-5 py-3">
        <Database className="h-4 w-4 text-muted-foreground" />
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold">資料集</h3>
            {/* Which dataset is being browsed AND edited — the two are the same
                by construction, so an edit cannot land somewhere unseen. */}
            {dsList.length > 0 && (
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="rounded-md border bg-white px-2 py-1 text-xs font-medium"
                title="切換要瀏覽與編輯的資料集"
              >
                {dsList.map((d) => (
                  <option key={d.id} value={d.slug}>
                    {d.slug} ({d.images})
                  </option>
                ))}
              </select>
            )}
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Ultralytics Platform 資料集 — 綠框為已標註的獼猴；點縮圖可編輯標註，右鍵可移動切分／刪除
          </p>
        </div>
        {/* Split filter chips */}
        <div className="ml-auto flex items-center gap-1">
          {SPLITS.map((s) => (
            <button
              key={s}
              onClick={() => setSplit(s)}
              className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                split === s
                  ? "bg-primary text-white"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              }`}
            >
              {SPLIT_LABEL[s]}
              {counts && (
                <span className="ml-1 font-mono tabular-nums opacity-70">
                  {counts[s] ?? 0}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Drop zone + per-file upload state */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`m-4 mb-0 rounded-md border-2 border-dashed px-4 py-3 transition-colors ${
          dragOver ? "border-primary bg-primary/5" : "border-muted"
        }`}
      >
        <div className="flex flex-wrap items-center gap-2">
          <UploadCloud className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            拖曳 JPG / PNG 到此上傳至訓練資料集（一次最多 {UPLOAD_MAX} 張，未標註＝背景負樣本）
          </span>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadingCount > 0}
            className="ml-auto rounded-md border px-2.5 py-1 text-xs font-medium hover:bg-muted disabled:opacity-40"
          >
            選擇檔案
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </div>
        {uploadNote && (
          <p
            className={`mt-2 text-xs ${
              uploadNote.type === "success" ? "text-green-700" : "text-red-600"
            }`}
          >
            {uploadNote.text}
          </p>
        )}
        {uploadItems.length > 0 && (
          <ul className="mt-2 max-h-32 space-y-0.5 overflow-y-auto">
            {uploadItems.map((u) => (
              <li key={u.name} className="flex items-center gap-1.5 text-[11px]">
                {u.status === "uploading" && (
                  <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                )}
                {u.status === "done" && (
                  <CheckCircle2 className="h-3 w-3 text-green-600" />
                )}
                {u.status === "failed" && <XCircle className="h-3 w-3 text-red-500" />}
                <span className="truncate">{u.name}</span>
                {u.reason && <span className="text-red-500">{u.reason}</span>}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Thumbnail grid */}
      <div className="p-4">
        {error && (
          <div className="mb-3 rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">
            {error}
          </div>
        )}
        <MessageBanner message={opNote} className="mb-3" />
        {images.length === 0 && loading ? (
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
            {Array.from({ length: 12 }).map((_, i) => (
              <div key={i} className="aspect-video animate-pulse rounded-md bg-muted" />
            ))}
          </div>
        ) : images.length === 0 ? (
          <div className="py-10 text-center text-sm text-muted-foreground">
            {error ? "無法載入資料集影像" : "此切分尚無影像"}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
              {images.map((img) => (
                <DatasetThumb
                  key={img.id}
                  img={img}
                  onClick={() => setViewing(img)}
                  onContextMenu={(x, y) =>
                    setCtxMenu({
                      x: Math.min(x, window.innerWidth - 200),
                      y: Math.min(y, window.innerHeight - 180),
                      img,
                    })
                  }
                />
              ))}
            </div>
            {hasMore && (
              <div className="mt-4 text-center">
                <button
                  onClick={() => load(split, images.length, dataset)}
                  disabled={loading}
                  className="inline-flex items-center gap-1.5 rounded-md border px-4 py-2 text-xs font-medium hover:bg-muted disabled:opacity-40"
                >
                  {loading && <Loader2 className="h-3 w-3 animate-spin" />}
                  {loading ? "載入中…" : "載入更多"}
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Right-click context menu */}
      {ctxMenu && (
        <div
          className="fixed z-50 min-w-[11rem] rounded-md border bg-white py-1 shadow-lg"
          style={{ left: ctxMenu.x, top: ctxMenu.y }}
          onClick={(e) => e.stopPropagation()}
          onContextMenu={(e) => e.preventDefault()}
        >
          <p className="px-3 py-1 text-[10px] text-muted-foreground">
            {ctxMenu.img.split} · {ctxMenu.img.boxes.length} 個標註
          </p>
          {(["train", "val", "test"] as const)
            .filter((s) => s !== ctxMenu.img.split)
            .map((s) => (
              <button
                key={s}
                onClick={() => requestMove(ctxMenu.img, s)}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-muted"
                title={
                  s === "test" || s === "val"
                    ? "會改變棘輪評測集，基準線需要重定錨"
                    : undefined
                }
              >
                <FolderInput className="h-3.5 w-3.5" />
                移至 {s}
                {(s === "test" || s === "val") && (
                  <span className="ml-auto text-[9px] text-amber-600">影響棘輪</span>
                )}
              </button>
            ))}
          <div className="my-1 border-t" />
          <button
            onClick={() => requestDelete(ctxMenu.img)}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs text-red-600 hover:bg-red-50"
          >
            <Trash2 className="h-3.5 w-3.5" />
            刪除影像
          </button>
        </div>
      )}

      {viewing && (
        <DatasetImageEditor
          img={viewing}
          dataset={dataset}
          onClose={() => setViewing(null)}
          onSaved={(hash, boxes) => {
            applyBoxes(hash, boxes);
            flashOp("success", "標註已同步至 Platform");
          }}
          onSplitChanged={(hash, from, to) => {
            applySplit(hash, from, to);
            flashOp("success", `已移至 ${to} — 已同步至 Platform`);
          }}
        />
      )}
    </div>
  );
}

/** SVG annotation overlay — relative 0-1 coords scale with any thumbnail size.
 *  Green stroke, no fill; label text omitted (thumbnails are too small). */
function BoxOverlay({ boxes }: { boxes: DatasetBox[] }) {
  if (boxes.length === 0) return null;
  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      className="pointer-events-none absolute inset-0 h-full w-full"
    >
      {boxes.map((b, i) => (
        <rect
          key={i}
          x={b.x * 100}
          y={b.y * 100}
          width={b.w * 100}
          height={b.h * 100}
          fill="none"
          stroke="#22c55e"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}

function DatasetThumb({
  img,
  onClick,
  onContextMenu,
}: {
  img: DatasetImage;
  onClick: () => void;
  onContextMenu: (x: number, y: number) => void;
}) {
  const [broken, setBroken] = useState(false);
  return (
    <button
      onClick={onClick}
      onContextMenu={(e) => {
        e.preventDefault();
        onContextMenu(e.clientX, e.clientY);
      }}
      className="group relative w-full overflow-hidden rounded-md border bg-neutral-100 transition-colors hover:border-primary/50"
      style={{
        aspectRatio:
          img.width > 0 && img.height > 0 ? `${img.width} / ${img.height}` : "16 / 9",
      }}
      title={`${img.split} · ${img.boxes.length} 個標註`}
    >
      {broken || !img.thumbnail_url ? (
        // Expired signed URL — placeholder keeps the grid intact
        <div className="flex h-full w-full flex-col items-center justify-center gap-1 text-muted-foreground">
          <ImageOff className="h-4 w-4" />
          <span className="text-[9px]">縮圖已過期</span>
        </div>
      ) : (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={img.thumbnail_url}
            alt=""
            loading="lazy"
            onError={() => setBroken(true)}
            className="absolute inset-0 h-full w-full object-cover"
          />
          <BoxOverlay boxes={img.boxes} />
        </>
      )}
      <span className="absolute left-1 top-1 rounded bg-black/60 px-1 py-px text-[9px] font-medium text-white">
        {img.split}
      </span>
      {img.boxes.length > 0 && (
        <span className="absolute right-1 top-1 rounded bg-green-600/90 px-1 py-px font-mono text-[9px] text-white">
          {img.boxes.length}
        </span>
      )}
    </button>
  );
}

/** Annotation editor — the enlarged view is editable and syncs back to the
 *  Ultralytics Platform (which stays the single source of truth):
 *  drag on empty area draws a new box (single class: 0 猴子), click selects,
 *  Delete/刪除框 removes, 儲存 PUTs the labels, 還原 discards. The split
 *  dropdown moves the image, with an amber confirm when the move would land
 *  in test/val (it changes the ratchet benchmark). */
function DatasetImageEditor({
  img,
  dataset,
  onClose,
  onSaved,
  onSplitChanged,
}: {
  img: DatasetImage;
  // The dataset this image was browsed from. Passed explicitly rather than
  // resolved server-side, so a save always writes back to the dataset the
  // editor was opened from.
  dataset: string;
  onClose: () => void;
  onSaved: (hash: string, boxes: DatasetBox[]) => void;
  onSplitChanged: (hash: string, from: string, to: string) => void;
}) {
  const [broken, setBroken] = useState(false);
  const [boxes, setBoxes] = useState<DatasetBox[]>(img.boxes);
  const [original, setOriginal] = useState<DatasetBox[]>(img.boxes);
  const [selected, setSelected] = useState<number | null>(null);
  const [drawing, setDrawing] = useState<{
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null>(null);
  const [saving, setSaving] = useState(false);
  const [split, setSplit] = useState(img.split);
  const [pendingSplit, setPendingSplit] = useState<string | null>(null);
  const [movingSplit, setMovingSplit] = useState(false);
  const [note, setNote] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const dirty = JSON.stringify(boxes) !== JSON.stringify(original);
  const imageUrl = img.full_url || img.thumbnail_url;

  // Keyboard: Delete/Backspace removes the selected box; Escape deselects,
  // then closes. Skipped while a form control has focus (the split dropdown).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
      if ((e.key === "Delete" || e.key === "Backspace") && selected != null) {
        e.preventDefault();
        setBoxes((prev) => prev.filter((_, i) => i !== selected));
        setSelected(null);
      } else if (e.key === "Escape") {
        if (selected != null) setSelected(null);
        else onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, onClose]);

  const relPoint = (e: React.PointerEvent) => {
    const rect = svgRef.current!.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
    };
  };

  const commitDraw = () => {
    if (!drawing) return;
    const x = Math.min(drawing.x0, drawing.x1);
    const y = Math.min(drawing.y0, drawing.y1);
    const w = Math.abs(drawing.x1 - drawing.x0);
    const h = Math.abs(drawing.y1 - drawing.y0);
    setDrawing(null);
    // Ignore accidental clicks — a real box needs some area (~0.5% per axis).
    if (w < 0.005 || h < 0.005) return;
    setBoxes((prev) => [...prev, { x, y, w, h, cls: 0 }]);
    setSelected(boxes.length);
  };

  const handleSave = async () => {
    setSaving(true);
    setNote(null);
    try {
      await updateImageLabels(img.hash, boxes, dataset);
      setOriginal(boxes);
      onSaved(img.hash, boxes);
      setNote({ type: "success", text: "已同步至 Platform" });
    } catch (err: unknown) {
      setNote({
        type: "error",
        text: err instanceof Error ? err.message : "儲存標註失敗",
      });
    } finally {
      setSaving(false);
    }
  };

  const doMoveSplit = async (target: string) => {
    setMovingSplit(true);
    setNote(null);
    try {
      await moveImagesSplit([{ hash: img.hash, split }], target, dataset);
      onSplitChanged(img.hash, split, target);
      setSplit(target);
      setNote({ type: "success", text: `已移至 ${target} — 已同步至 Platform` });
    } catch (err: unknown) {
      setNote({
        type: "error",
        text: err instanceof Error ? err.message : "移動切分失敗",
      });
    } finally {
      setMovingSplit(false);
      setPendingSplit(null);
    }
  };

  const requestSplit = (target: string) => {
    if (target === split || movingSplit) return;
    if (target === "test" || target === "val") setPendingSplit(target);
    else doMoveSplit(target);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl overflow-hidden rounded-lg border bg-white shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex flex-wrap items-center gap-2 border-b px-5 py-3">
          <span className="text-sm font-semibold">標註編輯</span>
          <select
            value={split}
            onChange={(e) => requestSplit(e.target.value)}
            disabled={movingSplit}
            className="rounded-md border bg-background px-2 py-1 text-xs outline-none focus:border-primary disabled:opacity-50"
            title="移動影像到別的切分"
          >
            {["train", "val", "test"].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          {movingSplit && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />}
          <span className="text-xs text-muted-foreground">
            {img.width}×{img.height} · {boxes.length} 個標註
          </span>
          {!img.full_url && (
            <span className="rounded bg-amber-50 px-1.5 py-0.5 text-[10px] text-amber-700">
              縮圖解析度
            </span>
          )}
          <button
            onClick={onClose}
            className="ml-auto rounded p-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Amber confirm — moving into test/val changes the ratchet benchmark */}
        {pendingSplit && (
          <div className="flex flex-wrap items-center gap-2 border-b border-amber-200 bg-amber-50 px-5 py-2.5 text-xs text-amber-800">
            <span>
              移至 <span className="font-medium">{pendingSplit}</span>{" "}
              會改變棘輪評測集，<span className="font-medium">基準線需要重定錨</span>。確定要移動嗎？
            </span>
            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={() => doMoveSplit(pendingSplit)}
                disabled={movingSplit}
                className="rounded-md bg-amber-600 px-2.5 py-1 font-medium text-white hover:bg-amber-700 disabled:opacity-40"
              >
                確定移動
              </button>
              <button
                onClick={() => setPendingSplit(null)}
                className="rounded-md border border-amber-300 px-2.5 py-1 hover:bg-amber-100"
              >
                取消
              </button>
            </div>
          </div>
        )}

        <div
          className="relative w-full select-none bg-black"
          style={{
            aspectRatio:
              img.width > 0 && img.height > 0
                ? `${img.width} / ${img.height}`
                : "16 / 9",
          }}
        >
          {broken || !imageUrl ? (
            <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 text-white/60">
              <ImageOff className="h-6 w-6" />
              <span className="text-xs">圖片網址已過期 — 請重新整理列表</span>
            </div>
          ) : (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imageUrl}
                alt=""
                draggable={false}
                onError={() => setBroken(true)}
                className="pointer-events-none absolute inset-0 h-full w-full object-fill"
              />
              {/* Editing surface: drag on empty area = new box; click box = select */}
              <svg
                ref={svgRef}
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
                onPointerDown={(e) => {
                  e.currentTarget.setPointerCapture(e.pointerId);
                  const p = relPoint(e);
                  setSelected(null);
                  setDrawing({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
                }}
                onPointerMove={(e) => {
                  if (!drawing) return;
                  const p = relPoint(e);
                  setDrawing((d) => (d ? { ...d, x1: p.x, y1: p.y } : d));
                }}
                onPointerUp={commitDraw}
                onPointerCancel={() => setDrawing(null)}
              >
                {boxes.map((b, i) => (
                  <rect
                    key={i}
                    x={b.x * 100}
                    y={b.y * 100}
                    width={b.w * 100}
                    height={b.h * 100}
                    fill="transparent"
                    stroke={i === selected ? "#f59e0b" : "#22c55e"}
                    strokeWidth={i === selected ? 2.5 : 1.5}
                    vectorEffect="non-scaling-stroke"
                    style={{ pointerEvents: "all", cursor: "pointer" }}
                    onPointerDown={(e) => {
                      e.stopPropagation();
                      setSelected(i);
                    }}
                  />
                ))}
                {drawing && (
                  <rect
                    x={Math.min(drawing.x0, drawing.x1) * 100}
                    y={Math.min(drawing.y0, drawing.y1) * 100}
                    width={Math.abs(drawing.x1 - drawing.x0) * 100}
                    height={Math.abs(drawing.y1 - drawing.y0) * 100}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
              </svg>
            </>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 px-5 py-3">
          <p className="text-[11px] text-muted-foreground">
            在空白處拖曳可新增框（類別：猴子）；點框選取後按 Delete 或「刪除框」移除
          </p>
          {note && (
            <span
              className={`text-xs ${
                note.type === "success" ? "text-green-700" : "text-red-600"
              }`}
            >
              {note.text}
            </span>
          )}
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => {
                if (selected != null) {
                  setBoxes((prev) => prev.filter((_, i) => i !== selected));
                  setSelected(null);
                }
              }}
              disabled={selected == null}
              className="flex items-center gap-1.5 rounded-md border border-red-200 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-40"
            >
              <Trash2 className="h-3 w-3" />
              刪除框
            </button>
            <button
              onClick={() => {
                setBoxes(original);
                setSelected(null);
              }}
              disabled={!dirty || saving}
              className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-40"
            >
              <Undo2 className="h-3 w-3" />
              還原
            </button>
            <button
              onClick={handleSave}
              disabled={!dirty || saving}
              className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-40"
            >
              {saving ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Save className="h-3 w-3" />
              )}
              {saving ? "儲存中…" : "儲存"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── 去重與訓練集 (master + derived) ───────────────────────────────────────────

/** The deduplicated dataset training actually runs on.
 *
 *  The Platform dataset above is the MASTER — everything is added there and
 *  nothing is ever deleted, so test/val stay pinned and the ratchet keeps
 *  sitting the same exam. This card regenerates a DERIVED copy whose train
 *  split has perceptual near-duplicates collapsed to one representative each.
 *  Regeneration is a minutes-long backend job, so the card polls its progress
 *  every 2 s while it runs and idles otherwise. */
function DerivedDatasetCard({ onTrain }: { onTrain: () => void }) {
  const { data, mutate } = useSWR<RegenerateState>(
    `${API_BASE}/api/admin/training/regenerate-dataset`,
    fetcher,
    { refreshInterval: (d) => (d?.state === "running" ? 2000 : 0) }
  );
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");

  const running = data?.state === "running" || starting;
  const latest = data?.latest ?? null;
  const result = data?.state === "done" ? data.result : null;

  const handleRegenerate = async () => {
    setStarting(true);
    setError("");
    try {
      const state = await regenerateDerivedDataset();
      await mutate(state, { revalidate: false });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "重新生成訓練集失敗");
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="rounded-lg border bg-white">
      <div className="flex flex-wrap items-center gap-2 border-b px-5 py-3">
        <Copy className="h-4 w-4 text-muted-foreground" />
        <div>
          <h3 className="text-sm font-semibold">去重與訓練集</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            主檔保留全部影像，訓練只用去重後的子集 — 重複影像會稀釋訓練訊號（實測：少
            25% 資料，新猴子召回 +5%）
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={handleRegenerate}
            disabled={running}
            className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-40"
            title="從主檔重新拉一份、只對 train 做感知去重，並在 Platform 建立新的訓練集"
          >
            {running ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            {running ? "生成中…" : "重新生成訓練集"}
          </button>
          <button
            onClick={onTrain}
            disabled={!latest || running}
            className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-40"
            title={latest ? `以 ${latest.slug} 啟動雲端訓練` : "先生成訓練集"}
          >
            <Play className="h-3.5 w-3.5" />
            用訓練集開始訓練
          </button>
        </div>
      </div>

      <div className="space-y-3 p-5">
        {error && <p className="text-xs text-red-600">{error}</p>}
        {data?.state === "failed" && data.error && (
          <div className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">
            生成失敗：{data.error}
          </div>
        )}

        {running && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="h-3 w-3 animate-spin" />
              <span>{data?.step || "準備中"}</span>
              {(data?.total ?? 0) > 0 && (
                <span className="font-mono tabular-nums">
                  {data?.done ?? 0}/{data?.total}
                </span>
              )}
            </div>
            {(data?.total ?? 0) > 0 && (
              <div className="h-1.5 overflow-hidden rounded-full bg-neutral-200">
                <div
                  className="h-full rounded-full bg-blue-500 transition-[width] duration-500"
                  style={{
                    width: `${Math.min(100, ((data?.done ?? 0) / (data?.total || 1)) * 100)}%`,
                  }}
                />
              </div>
            )}
            <p className="text-[11px] text-muted-foreground">
              需要幾分鐘（下載主檔影像 → 計算感知雜湊 → 上傳 → 等平台匯入），可以離開這個分頁
            </p>
          </div>
        )}

        {result && (
          <div className="rounded-md bg-green-50 px-3 py-2 text-xs leading-relaxed text-green-800">
            已生成 <span className="font-mono">{result.dataset_slug}</span> — train{" "}
            {result.train_before} 張 → <span className="font-medium">保留 {result.kept} 張</span>
            、移除 {result.dropped} 張重複；負樣本 {result.negatives_before} →{" "}
            {result.negatives_after}；val {result.val_images} 張未動
            {result.images_landed < result.expected_images && (
              <span className="ml-1 text-amber-700">
                （平台仍在匯入：{result.images_landed}/{result.expected_images}）
              </span>
            )}
          </div>
        )}

        {latest ? (
          <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1 text-xs">
            <span className="font-mono text-[11px]">{latest.slug}</span>
            <span>
              保留 <span className="font-mono tabular-nums font-medium">{latest.kept}</span> 張 /
              移除 <span className="font-mono tabular-nums">{latest.dropped}</span> 張
            </span>
            <span className="text-muted-foreground">
              負樣本 {latest.negatives_before} → {latest.negatives_after} · val{" "}
              {latest.val_images} 張
            </span>
            <span className="text-muted-foreground">
              建立時間 {formatCreatedAt(latest.created_at)}
            </span>
          </div>
        ) : (
          !running && (
            <p className="text-xs text-muted-foreground">
              尚未生成訓練集 — 按「重新生成訓練集」從主檔建立一份去重後的訓練資料集
            </p>
          )
        )}
      </div>
    </div>
  );
}

// ── Start-training dialog ────────────────────────────────────────────────────

const DEFAULTS = { base: "yolo26n.pt", epochs: 1000, imgsz: 1920, gpu: "rtx-6000-ada" };

function StartTrainingDialog({
  target,
  onClose,
  onStarted,
}: {
  target: "master" | "derived";
  onClose: () => void;
  onStarted: (modelId: string | null) => void;
}) {
  const [name, setName] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  // Which dataset the run trains on. The shipping model was trained on
  // `monkey-cocotree-allin-0816` (1220 images = train/val of all six datasets
  // merged) while this dialog silently sent the 1095-image July master, so a
  // retrain from the UI would have used LESS data than production.
  //
  // Defaulting to the LARGEST dataset was worse than the hardcoded name it
  // replaced: this account's biggest sets are `see world` (31,496 images) and
  // `VOC` (21,500), neither of which is a macaque, so the dialog opened with
  // stock data preselected and the project's own 1,655-image set had to be
  // chosen by hand every time. One run did reach a wrong dataset
  // (`exp-3` on `cats-batch-annotation-big`, precision 0.0001, recall 0.25).
  // The backend already marks the project dataset with `is_default`
  // (= config.ULTRALYTICS_DATASET); size was never the right proxy for it.
  const [dsList, setDsList] = useState<PlatformDataset[]>([]);
  const [dataset, setDataset] = useState<string>("");
  const [base, setBase] = useState(DEFAULTS.base);
  const [epochs, setEpochs] = useState(String(DEFAULTS.epochs));
  const [imgsz, setImgsz] = useState(String(DEFAULTS.imgsz));
  const [gpu, setGpu] = useState(DEFAULTS.gpu);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (target === "derived") return;   // that path picks its own derived set
    let cancelled = false;
    getPlatformDatasets()
      .then((rows) => {
        if (cancelled || rows.length === 0) return;
        setDsList(rows);
        setDataset(
          (cur) => cur || (rows.find((r) => r.is_default) ?? rows[0]).slug
        );
      })
      .catch(() => {
        /* falls back to the backend master */
      });
    return () => {
      cancelled = true;
    };
  }, [target]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setSubmitting(true);
    setError("");
    try {
      const body = {
        name: name.trim(),
        base: base.trim() || DEFAULTS.base,
        epochs: parseInt(epochs, 10) || DEFAULTS.epochs,
        imgsz: parseInt(imgsz, 10) || DEFAULTS.imgsz,
        gpu: gpu.trim() || DEFAULTS.gpu,
        ...(target === "derived" || !dataset ? {} : { dataset }),
      };
      const res = target === "derived" ? await trainOnDerived(body) : await startTraining(body);
      const modelId = res.modelId ?? res.model_id ?? res.id;
      onStarted(typeof modelId === "string" ? modelId : null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "啟動訓練失敗");
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-md rounded-lg border bg-white shadow-lg">
        <div className="flex items-center justify-between border-b px-5 py-3">
          <h3 className="text-sm font-semibold">
            {target === "derived" ? "用訓練集開始訓練" : "開始訓練"}
          </h3>
          <button
            onClick={onClose}
            className="rounded p-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4 p-5">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              名稱 <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              placeholder="例如：exp-monkey-aug"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              autoFocus
              className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary/20"
            />
          </div>

          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            {showAdvanced ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
            進階設定
          </button>

          {showAdvanced && (
            <div className="grid gap-3 rounded-md border bg-neutral-50 p-3 sm:grid-cols-2">
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  基礎模型
                </label>
                <input
                  type="text"
                  value={base}
                  onChange={(e) => setBase(e.target.value)}
                  className="w-full rounded-md border bg-white px-2.5 py-1.5 font-mono text-xs outline-none focus:border-primary"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  Epochs
                </label>
                <input
                  type="number"
                  min={1}
                  value={epochs}
                  onChange={(e) => setEpochs(e.target.value)}
                  className="w-full rounded-md border bg-white px-2.5 py-1.5 font-mono text-xs outline-none focus:border-primary"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  影像尺寸 (imgsz)
                </label>
                <input
                  type="number"
                  min={320}
                  step={32}
                  value={imgsz}
                  onChange={(e) => setImgsz(e.target.value)}
                  className="w-full rounded-md border bg-white px-2.5 py-1.5 font-mono text-xs outline-none focus:border-primary"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  GPU 類型
                </label>
                <input
                  type="text"
                  value={gpu}
                  onChange={(e) => setGpu(e.target.value)}
                  className="w-full rounded-md border bg-white px-2.5 py-1.5 font-mono text-xs outline-none focus:border-primary"
                />
              </div>
            </div>
          )}

          {/* Which data the run sees — the single most consequential choice in
              this dialog, so it is not hidden behind 進階設定. */}
          {target !== "derived" && dsList.length > 0 && (
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">
                訓練資料集
              </label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full rounded-md border px-3 py-2 text-sm"
              >
                {dsList.map((d) => (
                  <option key={d.id} value={d.slug}>
                    {d.slug} — {d.images} 張
                  </option>
                ))}
              </select>
            </div>
          )}

          <p className="text-xs text-muted-foreground">
            {target === "derived"
              ? "訓練會使用最新的去重訓練集，並在 Ultralytics Platform 雲端 GPU 上執行。"
              : "訓練在 Ultralytics Platform 雲端 GPU 上執行，會消耗 GPU 額度。"}
          </p>

          {error && <p className="text-xs text-red-600">{error}</p>}

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border px-4 py-2 text-xs font-medium text-foreground hover:bg-muted"
            >
              取消
            </button>
            <button
              type="submit"
              disabled={submitting || !name.trim()}
              className="flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-40"
            >
              {submitting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              {submitting ? "啟動中…" : "開始訓練"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ── Promote confirm dialog ───────────────────────────────────────────────────

function PromoteConfirmDialog({
  model,
  force,
  onClose,
  onDone,
}: {
  model: TrainingModel;
  force: boolean;
  onClose: () => void;
  onDone: (message: string, ok: boolean) => void;
}) {
  const [submitting, setSubmitting] = useState(false);
  // "Passing" = at least one split's latest 本站評測 cleared the ratchet.
  const hasPassingEval = Boolean(model.eval?.test?.passed || model.eval?.val?.passed);

  const handleConfirm = async () => {
    setSubmitting(true);
    try {
      const res = await promoteModel(model.id, force);
      onDone(
        res.message ||
          (force
            ? "已觸發強制上板 — 棘輪不通過也會部署"
            : "已觸發上板流程（棘輪雙檢通過才會部署）"),
        true
      );
    } catch (err: unknown) {
      onDone(
        err instanceof Error
          ? err.message
          : force
            ? "觸發強制上板失敗"
            : "觸發上板流程失敗",
        false
      );
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-md rounded-lg border bg-white shadow-lg">
        <div className="flex items-center justify-between border-b px-5 py-3">
          <h3 className="text-sm font-semibold">
            {force ? "強制上板確認" : "上板部署確認"}
          </h3>
          <button
            onClick={onClose}
            className="rounded p-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-3 p-5">
          <p className="text-sm">
            確定要{force ? "強制上板" : "上板"}模型「
            <span className="font-medium">{model.name || model.id}</span>」？
          </p>
          {!hasPassingEval && !force && (
            <div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2.5 text-xs leading-relaxed text-amber-800">
              <span className="font-medium">尚未通過本站評測</span>
              ，上板流程仍會先跑棘輪雙檢 — 未通過就不會部署。建議先按「評測」確認分數。
            </div>
          )}
          {force ? (
            <div className="space-y-2">
              <div className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2.5 text-xs leading-relaxed text-destructive">
                棘輪
                <span className="font-medium">沒通過也會部署到線上</span>
                。評測還是會照跑、照記錄，只是不擋。
              </div>
              <div className="rounded-md bg-muted px-3 py-2.5 text-xs leading-relaxed text-muted-foreground">
                <span className="font-medium">baseline 不會更動</span>
                ，地板維持在專案真正達到過的分數 — 所以之後的評測會持續對線上模型報
                FAIL，這是刻意的。
                <br />
                評測本身跑不完（例如基準設定不符）
                <span className="font-medium">仍然會擋下來</span>
                ：強制的是「分數不夠」，不是「沒量到」。
                <br />
                部署後請到 ml/PROMOTIONS.md 補一列，寫清楚放棄了什麼。
              </div>
            </div>
          ) : (
            <div className="rounded-md bg-amber-50 px-3 py-2.5 text-xs leading-relaxed text-amber-800">
              會在邊緣節點跑棘輪精度雙檢（test + val 兩個資料切分），
              <span className="font-medium">兩者都通過才會自動部署</span>
              ；未通過則不會更動線上模型。
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            最終驗證結果請至 GitHub Actions 的「Model Promote (Ratchet Gate)」
            工作流程查看。
          </p>
          <div className="flex justify-end gap-2 pt-1">
            <button
              onClick={onClose}
              className="rounded-md border px-4 py-2 text-xs font-medium text-foreground hover:bg-muted"
            >
              取消
            </button>
            <button
              onClick={handleConfirm}
              disabled={submitting}
              className={
                force
                  ? "flex items-center gap-1.5 rounded-md bg-destructive px-4 py-2 text-xs font-medium text-white hover:bg-destructive/90 disabled:opacity-40"
                  : "flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-40"
              }
            >
              {submitting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : force ? (
                <ShieldAlert className="h-3.5 w-3.5" />
              ) : (
                <Rocket className="h-3.5 w-3.5" />
              )}
              {submitting ? "觸發中…" : force ? "確認強制上板" : "確認上板"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
