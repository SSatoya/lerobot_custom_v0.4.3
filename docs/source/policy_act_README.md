# ACT with DINOv3 Custom Policy (LeRobot)

## 📌 概要 (Overview)
本実装は、LeRobotフレームワークにおける標準の **ACT (Action Chunking Transformer)** ポリシーを拡張し、画像バックボーンをResNetから **DINOv3 (Vision Foundation Model)** に差し替えたカスタムポリシー `act_with_dinov3` の実装ドキュメントです。

AI WORKER（MAGIシステム）における段ボールやトレーの搬送・貯蔵タスクにおいて、従来の2D CNN（ResNet）では困難であった**「奥行き・3次元的な幾何学関係」**の理解をモデルに付与し、関節角度（Joint値）依存のポリシーから脱却することを目的としています。

---

## 🏗️ 既存ACTとのアーキテクチャ比較

| コンポーネント | 既存のACT (LeRobot標準) | 本実装 (`act_with_dinov3`) | 変更の目的・効果 |
| :--- | :--- | :--- | :--- |
| **Backbone** | ResNet18 (スクラッチまたは事前学習) | **DINOv3 vits16 (完全凍結)** | 物理世界の強力な3次元・幾何学特徴（深度・法線など）を抽出。破滅的忘却を防ぐため凍結して使用。 |
| **画像特徴量** | 2Dの空間マップが出力される | 1Dパッチ列が出力されるため、**手動で2Dマップへ再構成** | TransformerのPositional Embeddingで「画像のどこに何があるか」を正しく処理させるため。 |
| **次元圧縮** | CNNのストライドで自然に圧縮される | **2x2のAverage Poolingを適用** | トークン数爆発によるTransformerのOOM（メモリ不足）を回避し、意味を保ったまま計算量を1/16に削減。 |
| **画像正規化** | データセット固有の平均・標準偏差 | **ImageNet標準の平均・標準偏差** | DINOv3が本来持つポテンシャルを引き出すため。LeRobotの自動正規化をオフにし、Forward内で手動適用。 |

---

## 🚀 実装の全体像と工夫点（The "How" and "Why"）

### 1. バックボーンの読み込みと凍結 (`__init__`の改修)
DINOv3をローカルリポジトリから読み込み、後段のTransformer（ACT）のみを学習させる**階層的アプローチ**を採用しました。

* **工夫点:** DINOv3のパラメータに対して `requires_grad = False` を設定。少ないロボットのデモンストレーションデータで巨大なVFMをFine-tuningしようとすると、事前学習された「世界の法則」が壊れる（破滅的忘却）ため、DINOv3は「優秀な固定のセンサー」として扱います。

### 2. 特徴量の抽出と空間マップの再構成 (`forward`の改修)
DINOv3（ViT）の出力は `[Batch, N, Dim]` の1次元トークン列です。これをTransformerの空間エンコーディングに乗せるため、2Dのグリッド形状に再構成します。

* **工夫点:** パッチサイズ（vits16の場合は16）を用いて画像の縦横を割り、`(B, D, H/16, W/16)` の形に `reshape` してから後段の処理（`einops.rearrange`等）に渡すことで、ViTとCNNの構造的差異を吸収しました。

### 3. VRAM枯渇（OOM）の回避：2x2 Average Pooling
DINOv3 (vits16) はResNetと比べて空間の圧縮率が低く、出力されるトークン数が4倍になります。TransformerのAttention層のメモリ消費は**トークン数の2乗 ($O(N^2)$)** に比例するため、メモリ消費量が16倍に跳ね上がりOOMが発生しました。

* **工夫点:** 空間マップに再構成した直後に `torch.nn.functional.avg_pool2d(cam_features, kernel_size=2, stride=2)` を挿入。DINOv3のリッチな意味表現を保ったまま空間解像度を半分に落とし、トークン数をResNet同等に抑えることでバッチサイズを確保しました。

### 4. 正規化の「二重掛け」競合の解決
LeRobotはデフォルトで、対象データセット（今回の場合はAI Workerのシミュレーション画像）の色分布から平均と標準偏差を算出し、入力画像を自動で正規化（`MEAN_STD`）します。しかし、DINOv3は「ImageNetの正規化」を前提としています。

* **工夫点:**
    1. `configuration_act.py` の `normalization_mapping` から `"VISUAL"` を削除し、LeRobotの自動正規化を無効化。
    2. ポリシーの `forward` 内の一番最初で、ImageNetの `mean` と `std` を用いて手動で画像テンソルを正規化。これにより、DINOv3にとって未知の分布となることを防ぎました。

---

## 👁️ Attention Map (ヒートマップ) 可視化ツールへの対応

学習済みモデルを評価（`eval_v3.0.py`）し、Physical-AI-Interpretabilityツールで注意マップを可視化する際、ACT特有およびPyTorch 2.0以降の仕様による罠があり、これらを改修しました。

### Fix A: 空間形状計算のハードコード回避
* **問題:** ツール側が「バックボーン＝ResNet（`feature_map`辞書を返す）」前提でハードコードされていたためクラッシュ（`IndexError`）。
* **解決策:** `_get_image_spatial_shapes` 内に `is_dinov3` の分岐を作成。DINOv3の場合はバックボーンの推論を回さず、単に `(H // 16, W // 16)` という数学的計算のみで正確なグリッドサイズを返却するよう改修し、計算コストも削減しました。

### Fix B: PyTorch SDPA（Flash Attention）による重み消失の回避
* **問題:** `Expected 1200, got 0` エラー。PyTorch 2.0のSDPA（最適化アルゴリズム）がメモリ節約のためにAttentionの重み行列を出力せず破棄してしまう仕様が原因。
* **解決策:** 評価スクリプトで `torch.backends.cuda.enable_math_sdp(True)` などを設定し、推論の瞬間だけSDPAを無効化。強制的に数学ベースの計算を行わせることで、可視化ツールのHookが正しい重みテンソルを抽出できるようにしました。

---

## 🔧 ファイル別の主要な変更点まとめ

1. **`configuration_act.py`**
   * クラス名を `ACTWithDINOv3Config` に変更し、`"act_with_dinov3"` として登録。
   * `vision_backbone` の「`resnet`から始まらないとエラー」というバリデーションを削除。
   * 画像の `NormalizationMode.MEAN_STD` を削除。
2. **`modeling_act.py`**
   * クラス名を `ACTWithDINOv3Policy` に変更。
   * `__init__` で `torch.hub.load` を用いたDINOv3のローカルロードとパラメータ凍結（`requires_grad = False`）を実装。
   * `forward` にImageNet正規化、1Dトークンの2D再構成、2x2 Pooling処理を追加。
3. **`act_attention_mapper.py` (可視化ツール)**
   * `_get_image_spatial_shapes` をDINOv3のパッチ計算に対応。
   * `select_action` の推論ブロックを `torch.backends.cuda.sdp_kernel` コンテキストマネージャで囲み、Attention重みを強制出力。

---
**📝 開発者メモ:**
このアーキテクチャにより、AIはピクセルベースのパターン認識ではなく、「3D的なエッジや奥行き」を捉えた上でTransformerの行動計画（Action Chunking）に移行します。シミュレーション環境での行動成功率向上と、Sim-to-Realへのスムーズな移行が期待されます。