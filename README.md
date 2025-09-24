# XrayPnxSegment

## 簡介
- XrayPnxSegment 是一個用於胸腔 X 光 pneumothorax（氣胸）語意分割的實驗訓練與比對工具集合。
- 主要功能：
  - 多 stage 訓練管線 (train.py)，支援多種 segmentation backbone（DeepLabV3+, UNet, FPN, PSPNet）。
  - 資料預處理與樣本分層（preclassify.py、preparing_data.py）。
  - Ratio/strata-based sampler，維持訓練中正負樣本比率並支援 hard-sample boosting。
  - 實驗結果視覺化（comparison.py），合併各 stage 的驗證指標並繪圖比較。

## 專案目錄
- d:\XrayPnxSegment\
  - train.py              
  - preclassify.py         
  - preparing_data.py      
  - comparison.py           
  - .gitignore
  - XrayPnxSegment/
    - datasets/
      - pnxImgSegSet.py
    - models/
      - modeling_segModels.py
    - processors/
      - img_processor.py
    - trainer/
      - building_SegModelTrainer.py
    - common/
      - utils.py          

## 必要套件
- Python 3.8+
- torch, torchvision, numpy, matplotlib, scikit-learn
- 建議建立虛擬環境（Windows 範例）：
  - python -m venv .venv
  - .venv\Scripts\activate
  - pip install -r requirements.txt
  - 或：pip install torch torchvision numpy matplotlib scikit-learn

## 資料準備
1. 先執行 preclassify.py
   - 目的：使用預訓練分類器對影像做快速判斷或推論，輸出每張影像的分類機率（cls_prob）或其他輔助資訊，用於後續的 sampling 加權或難樣本提升。
   - 範例（專案根目錄）：
     - .venv\Scripts\activate
     - python preclassify.py
   - 輸出：通常會產生含 cls_prob 的中間資料或 JSON 檔（視程式實作）。

2. 再執行 preparing_data.py
   - 目的：整理/驗證原始影像與遮罩，建立 meta JSON（dataset list），計算 has_pnx、strata（分層）、並將 preclassify 的 cls_prob 整合進來。
   - 範例：
     - .venv\Scripts\activate
     - python preparing_data.py
   - 輸出：一個或多個 meta JSON（例如 `subset_data_YYYYMMDDHHMM.json`），以及可能的裁切後遮罩資料夾。train.py 以此 meta JSON 當作資料清單來源。

meta JSON 範例（單一 entry）
{
  "image_path": "images/00001.png",
  "mask_path": "masks/00001_mask.png",
  "cropped_mask_path": "masks/cropped/00001_mask.png",
  "has_pnx": true,
  "strata": "1_q3",
  "cls_prob": 0.42
}

## 訓練流程（run_pipeline / train.py）
1. 準備：確認已完成 preclassify.py 與 preparing_data.py，並且 meta JSON 放在 config['meta_path'] 指定的位置。
2. 在 train.py 中設定 config：
   - root_path, meta_path, mask_key (e.g. 'cropped_mask_path'), bsz, device, save_path, criterion 等。
3. 定義 stages（list of dict），每個 stage 可包含：
   - epochs, lr, image_size, sample_rate (目標正樣本比率), strata_weights, scheduler...
4. 執行 train.py：
   - .venv\Scripts\activate
   - python train.py
5. 執行細項：
   - validate_dataset()：檢查 meta JSON 與檔案完整性。
   - 建立 Dataset（pnxImgSegSet）與 Sampler（create_ratio_based_sampler）。
   - 若資料充足且可不重複抽樣，Sampler 回傳 SubsetRandomSampler；否則回傳 WeightedRandomSampler（with replacement）。
   - 建模型（MODEL_BUILDERS）-> 取 loss/optimizer -> train_model()。
   - 每個 stage 會儲存 best weights 以及該 stage 的 metrics JSON（供 comparison.py 使用）。