# Live 影像與 trail 契約

這份文件記錄 live 影像與 trail 影片的實作邊界。Overlay 預設關閉，開啟前仍要在 shadow environment 做四路 Edge benchmark。

## Trail：保留什麼

「完整影格」分成兩個層次：

- RTSP source 不是無損錄影。`CameraStream` 的 bounded queue 滿時會丟掉最舊影格；`FRAME_SKIP` 也會決定 detector 消費哪些影格。因此系統不能宣稱保存每一個攝影機送來的 source frame。
- trail 會從 detector 已消費、仍在 `_FrameBuffer` 的影格中選最多 `TRAIL_GIF_MAX_FRAMES`（目前 150）張。這些 selected frames 全部送進 MP4；固定 24 FPS 的空槽只重複上一張影格，不會再解碼或複製新的 source image。
- `TRAIL_VIDEO_MAX_FRAMES`（目前 600）是 encoded timeline 上限。若長時間 gap 會超過上限，所有 selected frames 仍保留，但均勻壓進 bounded timeline，這段影片不再代表 wall-clock 的 1:1 速度。
- raw-frame ZIP 是乾淨的 labeling artefact，受 `RAW_FRAMES_MAX_MB`（目前 128 MiB）限制；超過 byte ceiling 時仍會取樣。MP4 完整不代表 raw ZIP 一定完整。

## Trail：24 FPS 與 seek 對齊

`TRAIL_VIDEO_FPS=24` 是 MP4 的固定編碼速率。selected frame 依來源 capture timestamp 放到 output slot：

- gap 中的 slot 重複最近一張已取得的幀，避免 RTSP burst 把數秒 wall-clock 壓成幾秒的加速影片；
- 同一個 slot mapping 也產生 detection 的 `clip_offset_seconds`、`clip_end_offset_seconds` 與 `clip_segments`；
- wall-clock 軸保存每個 encoded slot 所代表的 source timestamp，不能再用另一條 regular-clock binary search 推算 offset；
- 即時擷取正常會提供 wall-clock timestamp。若某次 trail 完全沒有可用 time base，位置不能猜測為影片開頭；既有 finaliser 會把它視為不完整 artefact 而不寫入 detection row，但仍不會讓 detector thread 結束。

影片產製或上傳失敗會記錄並讓該次 artefact／detection 按既有 fail-open policy 處理，不得讓 detector thread 結束。

## Live overlay：實作設計與降級

採用 Edge 直接產生 annotated MJPEG frame 的方案，但傳輸上使用 bounded latest-only mailbox：

1. detector 在推論完成的同一 source frame 上取 target-class boxes、`frame_seq` 與 `capture_ts`；
2. detector 只做 mailbox offer，不做 JPEG 或 HTTP，編碼與登入／上傳在獨立背景 thread 執行；
3. 每路只保留一個待送 frame，sender 慢時丟棄中間 frame，不累積 queue；
4. cloud 只把最新 annotated JPEG 保存在記憶體，過期（預設 2 秒）就從 overlay endpoint 回退到 raw MJPEG；不寫 DB、不寫 Blob；
5. Live tab 的「原始／推論框」切換使用相同登入 token、camera scope 與 `live_feeds` feature gate；POC camera 仍拒絕 overlay。

因為框是在 Edge 對同一張 JPEG 畫上去，瀏覽器收到的框與影像是同一個 `frame_seq`，不需要把框猜回另一條 cloud raw stream。`frame_seq` 是順序主判準，Edge host 的 NTP 微調不會誤丟新幀。

## 可重現驗收

沒有 Edge/RTSP 的本機環境可以跑 deterministic contract：

```bash
uv run --project backend --with pytest python -m pytest -q tests/test_trail_video_fps.py
uv run --project backend python tests/test_detection_events.py
uv run --project backend python tests/test_trail_video.py
uv run --project backend python tests/test_worker_resilience.py
```

前端驗證：

```bash
cd frontend
npx tsc --noEmit
npm run lint
npm run build
```

已記錄的本機驗證訊號包括：timeline／overlay regression checks、H.264 + faststart、固定 `24/1` frame rate、selected frame count、overlay JPEG 尺寸、sequence ordering、expiry、latest-only mailbox 與 sender 失敗不阻塞 detector。20 個模擬 burst／gap 的 selected frames 實測輸出 58 個 encoded frames、`2.417s`、`24/1`，20 個 source frames 全部仍在 output sequence 中。

## Edge rollout gate

既有四路 baseline（overlay 關閉）是 aggregate 45.7 distinct frames/s、約 11.4/camera、zero duplicate、zero idle，推論約 21.9 ms。這不是 overlay 開啟後的成績；開啟 `LIVE_OVERLAY_ENABLED=true` 前，必須在 shadow environment 重跑同一組四路測試，至少記錄 capture／queue drop／detector consumed／overlay sent、端到端延遲、CPU/GPU、RSS 與網路流量。任何 overlay sender 失敗都應只留下 raw MJPEG 可播放。
