# CoCoTree 獼猴監測平台

[English](README.md) · **繁體中文**

這是以 FastAPI、SQLite、YOLO 與 ByteTrack 建立的台灣獼猴監測原型，
搭配 Next.js／React 繁體中文儀表板。攝影機 worker 偵測並追蹤獼猴，
將事件紀錄、影像、影片與軌跡時間軸連在一起，方便回看與人工審查。

## 已有功能

- 攝影機管理、即時影像與單一攝影機內的追蹤。
- 事件瀏覽、儲存影片播放、偵測審查與顯示權限管理。
- 本機媒體儲存，以及可選用的 Azure Blob Storage 設定。
- 模型訓練／評估串接與回歸檢查腳本。
- 本機驗證工具及選用的 edge 服務範例。

內附模型有兩個類別：`monkey=0`、`human=1`。監測流程只取 class 0，
第二類並不代表系統會追蹤人、統計人數或發送人員告警。
Track ID 屬於各攝影機的追蹤過程，不能用來確認跨攝影機的同一隻動物；
保存的造訪紀錄也不是族群數量。

5G／UGen300 部署、警戒區規則與 LINE 群組路由是競賽的後續開發項目。
現有 webhook 串接與這些規劃分開。

## 本機執行

需要 Python 3.12 以上、`uv`、Node.js 20.9 以上與 `npm`。
以下指令都從專案根目錄執行。

```bash
make install
```

第一次正式使用前，建立本機設定。以下指令會產生固定保存的簽章／雜湊密鑰
及隨機初始登入碼，並拒絕覆寫既有 `.env`：

```bash
python3 - <<'PYCONFIG'
from pathlib import Path
import os
import secrets

path = Path("backend/.env")
text = Path("backend/.env.example").read_text()
values = {
    "ADMIN_PASSWORD_SALT": secrets.token_hex(32),
    "SEED_ADMIN_CODE": secrets.token_urlsafe(18),
    "SEED_USER_CODE": secrets.token_urlsafe(18),
}
lines = [line for line in text.splitlines()
         if not any(line.startswith(key + "=") for key in values)]
lines += [f"{key}={value}" for key, value in values.items()]
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write("\n".join(lines) + "\n")
print("Created backend/.env. Read its SEED_* values locally to sign in.")
PYCONFIG
```

建立使用者後請保留 `ADMIN_PASSWORD_SALT`：系統用它簽署 session，
也用它雜湊已儲存的登入碼。Seed 登入碼只在使用者資料表為空時使用。
預設的 `demo-admin`／`demo-user` 與示範 salt 僅供本機示範，
正式使用前必須自行設定。

```bash
make dev
```

儀表板位於 `http://localhost:3000`，API 位於 `http://localhost:8000`，
互動式 API 文件位於 `http://localhost:8000/docs`。
用本機 `.env` 中的管理者登入碼登入後，再加入有權存取的攝影機。
此專案不附攝影機憑證或私人現場資料庫。

也可以分別執行 `make backend` 與 `make frontend`。
後端預設使用本機 SQLite 與 `backend/snapshots/`；若儀表板與 API
分開提供服務，請設定 `NEXT_PUBLIC_API_URL`。
環境變數以 [backend/config.py](backend/config.py) 為準，
常用設定列在 [backend/.env.example](backend/.env.example)。

## 模型與評估

`backend/best.pt` 是內附的獼猴偵測模型；缺少權重時後端會報錯。
訓練與資料集指令需要自行設定 Platform 帳號、API key 及有權使用的資料集。
原始現場影像不隨此專案提供；中繼資料格式請見[資料集說明](ml/datasets/README.md)。

目前模型曾以例外方式接受：test F1 為 **0.823166**，低於 **0.8254**
的閘門下限；validation F1 為 **0.888889**。原本較強的 baseline 維持不變。
這些是已記錄的 benchmark 結果，不代表此專案重新完成評測，也不保證現場準確率。
詳見[模型與評估說明](ml/PROMOTIONS.md)。

人工審查與模型評估是開發流程；執行中的偵測器不會自動從新造訪學習。
候選模型替換前應檢查固定的 test、validation split 與困難回歸案例。

## 驗證與選用部署

```bash
make ci
```

[本機測試工具](infra/ci/README.md)會執行後端檢查、TypeScript、lint
與前端 production build，不會部署服務。

此專案沒有啟用 GitHub Actions 部署或模型上板自動化。
[Workflow 範例](infra/workflow-examples/README.md)放在 Actions 目錄之外，
需要自行設定 runner、憑證與服務後才能使用。
[Edge 服務範例](infra/edge/README.md)使用本機程式碼啟動，
不會自動同步或重設 checkout。

## 檔案配置

| 目錄 | 內容 |
|---|---|
| `backend/` | API、攝影機 worker、偵測、追蹤與儲存 |
| `frontend/` | 儀表板與審查介面 |
| `tests/` | 單元檢查、回歸案例與離線分析工具 |
| `ml/` | 資料集中繼資料與評估說明 |
| `infra/` | 本機驗證與選用的服務範例 |
| `docs/` | 影片時間契約與探索性追蹤研究 |
