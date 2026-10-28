# Re-ID 離線研究結果

**日期：** 2026-10-23
**決策：** 暫不接入正式 tracking。外觀特徵有配對訊號，但尚未證明能可靠減少斷軌；原始影格已確認部分歷史軌跡混入不同個體，以及播放時間與攝影機時間不一致。

## 本次完成範圍

完成既有實驗稽核、正式標註匯出、原始影像對齊、五種外觀方法比較、按影片留出門檻、統計核對與選定原始影格的視覺檢查。沒有訓練 Re-ID 模型、修改人工標籤、合併正式 detection rows、更新 detector 或增加正式依賴。

這是現有資料上的探索實驗，不是完整線上 tracker 驗證。不能把配對分數換算成 IDF1、HOTA、identity switches 或實際個體數；這些需要連續影格的個體標註。

## 資料與來源

| 項目 | 數量 |
|---|---:|
| 匯出候選配對 | 922 |
| 候選影片／軌跡 | 194／597 |
| 缺少逐幀框的軌跡 | 111 |
| 既有人工欄位標記 | 63 |
| same／different／uncertain | 7／48／8 |
| 有標記的影片／軌跡 | 60／123 |
| 正例影片 | 7 |
| 用於外觀特徵的 crop | 348 |

人工欄位標記保留原值，不等於本研究重新驗證過的個體真值。63 對涉及的 126 個端點，其完整框序列都與原重建 manifest 一致；60 支來源各對應一個原始 archive，影格數也全部一致。沒有使用畫上追蹤框的影片抽特徵。

每條軌跡按觀測順序分成最多三段，各取最高信心度的框；102 條取三張、21 條取兩張。這不是按真實經過時間等分。crop 寬高中位數約 56×57.5 pixels，30 張高度小於 32 pixels。未發現跨軌跡完全相同的 crop 檔案。

## 比較方法

1. **歷史幾何規則：** 原樣重跑 `tests/rebuild_clips.py::merge_track_fragments(appearance=None)`，涵蓋每支影片的全部已保存軌跡。這些軌跡已經過之前的處理，結果只能描述殘餘軌跡，不能當作目前線上 ByteTrack 的整體準確率。
2. **HSV histogram：** 沿用既有 64×64、16×8 HSV histogram 與最佳三組 correlation 平均；歷史重建原本就有此種外觀輔助，不能算成新增能力。
3. **Person Re-ID：** OpenVINO `person-reidentification-retail-0288`，256 維。原模型是人體全身特徵模型，獼猴表現必須實測。[模型說明](https://docs.openvino.ai/2023.3/omz_models_model_person_reidentification_retail_0288.html)
4. **YOLO features：** 現有 `best.pt` 倒數第二層 pooled feature，對 crop 額外前向運算；不是免費重用線上已計算的特徵。
5. **DINOv2 ViT-S/14：** 凍結的通用視覺特徵，CLS token；使用官方預處理。[模型卡](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
6. **MegaDescriptor-T-224：** 凍結的動物 Re-ID 模型，Swin-T backbone；使用模型預處理設定。[模型卡](https://huggingface.co/BVRA/MegaDescriptor-T-224)

神經模型使用外擴 10% 的 crop，各 crop 向量先 L2 正規化，平均後再正規化，以 cosine distance 配對。Histogram 使用緊框。DINOv2／MegaDescriptor 的中心裁切可能截掉長身體；person 模型會改變長寬比例，這些是本次預處理的限制。

候選規則預先固定為：同影片／攝影機、時間不重疊、播放時間間隔 ≤9 秒、端點中心距離 ≤平均對角線的兩倍、最大邊長比 ≤2.5。9 秒是實驗限制，並非完整線上 continuation 規則；後者設定為 20 秒。逐幀核對後發現播放時間不一定代表攝影機經過時間，因此不能用這道 gate 宣稱真實 9 秒內接軌有效。

門檻採 leave-one-clip-out：整支影片留出，用其他影片最大化 same recall，同時要求 different 與 uncertain 都不合併；同分時取最小門檻，沒有可接受正例就 abstain。這是保守校準策略，不是模型所能達到的最高 recall。同一批資料被用來比較模型，沒有未碰過的最終 holdout。

## 探索結果

以下計數以 **7 same、48 different、8 uncertain 的既有標記** 為準，尚未修訂混合個體的軌跡。uncertain 合併另外列出，不當成已知對錯。

| 方法 | 接回 same | 合併 different | 合併 uncertain | 未接回 same |
|---|---:|---:|---:|---:|
| 歷史幾何規則重跑 | 0 | 1 | 1 | 7 |
| 實驗候選 gate，無外觀限制 | 5 | 1 | 4 | 2 |
| Gate + HSV histogram | 1 | 0 | 1 | 6 |
| Gate + person Re-ID | 0 | 0 | 1 | 7 |
| Gate + YOLO features | 3 | 0 | 0 | 4 |
| Gate + DINOv2 | 3 | 0 | 0 | 4 |
| Gate + MegaDescriptor | 2 | 0 | 1 | 5 |

新 gate 在 922 對中留下 63 對；其中只有 10 對有標記（5 same、1 different、4 uncertain）。外觀分數涵蓋全部 63 對有標記的配對，沒有計算其他未標記配對的外觀，因此不能把未計算項目當作正確拒絕。研究只模擬 pair 決策，沒有驗證傳遞性合併群組。

DINOv2 與 YOLO 接受完全相同的三對：`16758:16759`、`16843:16847`、`16926:16928`。它們都集中在相似入口／牆面與低光場景；camera 名稱不同也不能證明物理場景獨立。

| 外觀排序（不含 gate） | 全部 55 對 AP | 僅時間分開的 17 對 AP |
|---|---:|---:|
| HSV histogram | 0.4333 | 0.9571 |
| Person Re-ID | 0.5644 | 0.8445 |
| YOLO features | 0.4302 | 0.8286 |
| DINOv2 | 0.6768 | 0.9571 |
| MegaDescriptor | 0.6586 | 0.8329 |

DINOv2 的排序訊號值得繼續研究，但在時間分開的資料上，簡單顏色特徵也有相同 AP。這不能排除背景相似造成的效果。

### 統計限制

- 3/3 被既有標記支持的合併，其雙側 exact 95% precision 區間仍是 **29.24%–100%**，遠不足以支持 99.5% precision。
- DINOv2／YOLO 相對殘餘軌跡幾何規則，4 對改善、0 對退步，exact McNemar **p=0.125**；相對新 gate 為 1 改善、2 退步，**p=1.0**。都不支持顯著改善；模型選擇、片段相關性與標記品質還會進一步限制解讀。
- 原設計要求 precision ≥99.5%、95% 下界 ≥99%。即使完全零錯，按雙側 exact 95% 區間、在獨立二項假設下也至少需要 **368 次正確接受的合併** 才可能達到下界；300 個正例再切 20% holdout 不可能滿足。實際相關片段需要更多證據。
- 不報 fragmentation 改善百分比：目前沒有完整個體數真值，不能用 track 數／最高同時偵測數替代。

## 原始影格揭露的問題

核對全部 11 張配對圖表、48 張場景圖，另取六支影片的 41 張原始影格查連續性。這些是獨立 AI 視覺稽核觀察，不是新增人工真值，也不覆寫原人工標記。

- **播放時間不等於攝影機時間。** `16758:16759` 的 A 結束 frame 40 為 18:37:37，B 開始 frame 79 為 18:38:31，相隔 **54 秒**，但 metadata 是 **3.9 秒**。`16843:16847` 對應 19:14:25→19:14:44，即 **19 秒**，metadata 卻是 **1.5 秒**。重建的 frame index／固定 FPS 不能作為真實接軌速度與間隔。
- **一條軌跡可能包含兩隻猴子。** `16843` 從 frame 0 的左側猴子換到 frame 5–6 的右側猴子；`17129` 從前方猴子換到後方猴子，`17131` 在 frame 25 又換到前方猴子、26 回到後方猴子。這類整條軌跡的 same/different 定義已經不成立，平均 embedding 不能修復已混合的身份。
- **低光與背景佔比。** `16926:16928` 等配對的 crop 缺乏足夠個體細節。外觀相近不能證明相同個體；可疑共同框或身體交疊也不能只靠「同時出現」自動造 different 真值。

上述問題來自歷史重建資料，尚未證明目前 live worker 也會出現相同行為。沒有將影格索引時間直接改成 OSD 時間，以免破壞既有影片播放座標。

## 舊實驗重新解讀

2026-08-31 的 524 對其實包含 8 個 `user_gt` same、2 個 `user_gt` different，以及 514 個自動時間重疊負例。7 個正例來自同一次移動的相鄰連結，並非七次獨立重返場景。

舊 person encoder 的四組 crop 設定都未達 pooled precision 99.5%，但門檻在同一批資料選擇，且部分自動負例的 cosine distance 記錄為 0；不能因此斷言所有 Re-ID 無效。只看十個 user-sourced pairs，有三種設定能在樣本內完全分開，仍不足以證明泛化。本次改用正式標記欄位，並保留來源區別。

## 成本與再現

本機 CPU 抽取 348 crops，未佔用監測主機的 GPU。Histogram 約 0.043 秒、person 約 0.807 秒、YOLO 約 5.448 秒、DINOv2 約 5.476 秒、MegaDescriptor 約 16.675 秒；不含各自載入／預熱，YOLO／DINOv2 計時含正規化與聚合，不含中間檔寫入；MegaDescriptor 則在推論後、聚合前停止計時。Histogram／OpenVINO 使用 2 threads，Torch 模型使用 8 threads，不能視為嚴格同資源速度排名，也不是 edge pipeline FPS。

前四種方法同程序 peak RSS 約 998 MB（含多模型與輸入），MegaDescriptor 獨立程序約 627 MB；未量測每模型隔離 RSS、MPS 或 330 支影片的完整成本。

研究使用的配對、裁切、分數與視覺核對紀錄未隨此專案提供；原始影像需要另行取得使用授權。下列結果屬於已記錄的探索實驗，不代表僅憑這份 checkout 就能完整重現。

本研究使用當時的單類別 YOLO checkpoint，並非目前內附的雙類別 checkpoint。DINOv2 revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`；MegaDescriptor revision `3ea58ff6c6195bc748bb86c111ff40c32bdddcba`。PyTorch 2.10.0；Ultralytics 8.4.14；Transformers 4.57.6；timm 1.0.27；OpenVINO 2026.0.0。

## 接入決策與下一個必要驗證

**目前 No-go。** 保留現有人工審查與 production tracking，不加入新的 encoder、同步狀態或自動合併路徑。

要讓 tracking 更可靠，先處理可觀測的根因：建立含真實時間的連續片段、把已混合不同猴子的軌跡拆成單一個體片段，再標記離開後返回、同位置先後不同猴子、低光與多人交會的案例。保留不確定選項，不用模型分數產生自己的真值。

資料通過後，才在完全相同 detections 上比較現行 ByteTrack＋continuation、BoT-SORT 關閉 Re-ID、同設定開啟 Re-ID；另測離線 tracklet 接回。BoT-SORT 的 appearance 與 geometry 結合不代表外觀必能否決錯誤幾何匹配，不能只切換設定就宣稱解決身份混淆。[BoT-SORT 原論文](https://arxiv.org/abs/2206.14651)、[Ultralytics tracking 文件](https://docs.ultralytics.com/modes/track/)

正式門檻維持：獨立 holdout 高 precision、錯誤合併受控、真實 fragmentation 至少改善 20%、不確定片段能 abstain、成本可接受且所有合併可追溯。這次結果沒有通過，因此沒有啟動正式合併或歷史回填。
