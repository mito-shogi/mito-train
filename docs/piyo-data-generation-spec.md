# ぴよ将棋 教師データ生成 — 依頼書 / 仕様書

**宛先**: ぴよ将棋 ipa 解析 + Frida hook + データ生成担当 (別リポジトリの LLM)
**発行元**: MITO (`/home/vscode/app` — 将棋盤 OCR パイプライン担当)
**発行日**: 2026-07-06
**関連文書**:
- `docs/ocr-phase2-plan.md` — Phase 2/3 全体計画 (背景)
- `docs/ocr-metrics.md` — 目的関数と評価指標
- `docs/piyo-piece-templates.md` — 駒テンプレ画像の命名規則 (既存の 420枚テンプレ)

---

## 0. 前提とゴール

### 依頼元 (MITO) が何をしている

MITO はブラウザ完結の将棋棋譜解析 Web アプリ。その一機能として **「Twitter 等の SNS に投稿された将棋盤スクショから SFEN を抽出する」** OCR パイプラインを Phase 2 で構築中。

Phase 0/1 でヒューリスティクス (pHash + NCC + 幾何検出) を実装したが、420 テンプレ相手の判別能力・Twitter 圧縮の domain gap で **Exact-Match Rate は実質 0%** に到達。学習ベースへ移行することが決まっている。

### 別レポ (依頼先) が何をしている

- ぴよ将棋 iOS アプリの **ipa 静的解析** (盤面 setter / renderer / UI トグルの RVA 特定)
- **Frida hook + JB 端末** を使った盤面画像自動キャプチャ
- NNUE 学習用の **80億局面プール** の SFEN リストを流用したデータ生成

### この依頼書のゴール

依頼先が生成するデータの **フォーマット・ディレクトリ構造・命名規則・品質基準** を明示し、MITO 側で受け取ってそのまま学習パイプラインに投入できる状態にする。

---

## 1. 認識対象のスコープ (絶対に守ってほしい範囲)

MITO 側の学習対象は **盤面 9×9 の駒配置 + 先手/後手の持ち駒** のみ。以下は認識対象外とする:

- 評価値グラフ・棋譜テキスト・広告バナー
- 手番 (先手番/後手番) 表示
- 手数・対局情報
- 駒動きアニメーション中の中間フレーム

**含めるべき**: 盤面 9×9、先手持ち駒、後手持ち駒
**含めない**: それ以外の UI 装飾

これに従ってキャプチャする UIView を選定してほしい。

---

## 2. Frida hook で実装してほしい機能

### 2.1 必須機能

#### F1. SFEN 入力 → 局面セット

任意の SFEN 文字列を受け取って、ぴよ将棋の内部盤面状態を強制的にその状態にする。

**期待する入力**: 標準 SFEN 文字列 (例: `"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"`)

**要求**:
- 王手放置局面 も受け付ける (validation を bypass する必要あり)
- 玉が敵陣内 等の異常配置も受け付ける
- 持ち駒に王がある など通常あり得ない SFEN も受け付ける (validation はスキップ)

理由: NNUE ランダム局面には非合法配置が含まれるため。CNN 学習の観点ではむしろ **多様性向上** に貢献する。

#### F2. UIView キャプチャ

局面セット後、以下の 3 種類の画像を PNG で保存:

1. **盤面画像**: 9×9 の盤面 UIView のみ
2. **先手持ち駒画像**: 先手側 (画面下) の持ち駒 UIView
3. **後手持ち駒画像**: 後手側 (画面上) の持ち駒 UIView

**代替案**: 上記 3 領域を含む縦長 1 枚画像でも可 (後手持ち駒 / 盤面 / 先手持ち駒 が縦に並ぶ形)。1 枚のほうが管理が楽なら **推奨**。

**画像サイズ**:
- 盤面: **512×512** 推奨 (最終的に 288×288 にリサイズして学習)
- 持ち駒: **384×96** 程度 (横長)
- 縦長 1 枚案: **512×768** 程度

**色空間**: RGB, 8bit/channel
**フォーマット**: PNG (無圧縮または低圧縮)
**背景**: 盤面 UIView そのままの見た目 (木目色 or ダーク背景)

#### F3. デザイン切り替え

ぴよ将棋の駒デザイン 14 種類 (k1〜k7 = ライト、k101〜k107 = ダーク) を Frida で切り替えて、同じ SFEN で複数バリエをキャプチャする。

**期待する切り替え粒度**: 1 SFEN あたり少なくとも 6 デザイン (ライト × 3 + ダーク × 3) をカバー

参考: 既存の駒テンプレ画像は `assets/piyo/k1/` 〜 `assets/piyo/k107/` として保存済み (`docs/piyo-piece-templates.md` に命名規則あり)。学習側でも同じデザイン ID を使う想定。

#### F4. 盤の向きトグル

先後逆表示 (盤回転) のトグルも切り替えられるようにしてほしい。同じ SFEN で 向き通常/逆 の 2 パターン取れると理想。

**代替**: 向き固定でも構わない (SFEN を反転させて 2 パターン生成する方法で代替可能)

### 2.2 オプション機能 (あると嬉しい)

#### O1. 盤面 UIView の frame (座標情報) を取得

盤面 UIView の親ビューにおける矩形 (`CGRect boardFrame`) を JSON に出力してくれると、盤面検出モデルの学習ラベルとして使える。

**用途**: 4隅座標の GT ラベル自動生成 (人手アノテ不要)

**フォーマット**: JSON で `{ "x": 0, "y": 100, "w": 512, "h": 512, "screen_w": 1080, "screen_h": 1920 }`

#### O2. スクリーンショット全体も同時に保存

盤面 UIView だけでなく、**画面全体のスクショ** も同時に保存できると、augmentation 用の背景として使える。

**フォーマット**: 上記 F2 と同じ PNG。ファイル名に `_full` サフィックス。

---

## 3. SFEN サンプリング仕様

### 3.1 プール
NNUE 学習用の **80億局面プール** から抽出する。

### 3.2 抽出条件

以下のフィルタで **層別抽出 (stratified sampling)** してほしい:

| 層 | 比率 | 条件 |
|---|---|---|
| 序中盤 | 30% | 盤上駒数 25〜38、成駒 0〜1 枚 |
| 中盤 | 40% | 盤上駒数 20〜30、成駒 1〜3 枚 |
| 中終盤 | 20% | 盤上駒数 15〜25、成駒 2〜5 枚 |
| 終盤・成駒多発 | 10% | 盤上駒数 10〜20、成駒 3 枚以上 |

**追加条件** (全層共通):
- 先手番/後手番 同数 (50% ずつ)
- 持ち駒総数の分布:
  - 0 枚: 20%
  - 1〜3 枚: 30%
  - 4〜8 枚: 30%
  - 9 枚+: 20%
- **特殊局面除外しない**: 王手放置、王が敵陣、玉と王が同陣営、極端持ち駒 も含める

### 3.3 駒種登場回数の事後チェック

サンプリング後、各駒種 (29 クラス) の登場回数を集計。以下の **最低ライン** を満たすまで追加抽出:

| 駒種 | 最低登場回数 (盤上マス数として) |
|---|---|
| P (歩、生駒) | 200,000 |
| L, N, S, G, B, R (香桂銀金角飛、生駒) | 各 20,000 |
| K (王/玉) | 10,000 |
| +P (と金) | 5,000 |
| +L (成香)、+N (成桂)、+S (成銀) | 各 3,000 |
| +B (馬)、+R (龍) | 各 5,000 |

**成駒系がレア** なので、成駒多発局面を選択的に多く抽出することが必要。

### 3.4 抽出数

**段階的に**:

| Tier | SFEN 数 | 用途 |
|---|---|---|
| Tier 1 | **10,000** | 試験生成、Frida hook 動作確認 |
| Tier 2 | **30,000** | 標準学習セット (推奨) |
| Tier 3 | **100,000** | 精度追い込み用 (Tier 2 で不足なら) |

**まず Tier 1 で 10,000 局面出力 → MITO 側で試験学習 → 精度確認 → Tier 2 に拡張** の流れ。

---

## 4. 出力ディレクトリ構造 (MITO 側で受け取る形)

**最終的にこの構造で受け取りたい**:

```
training/data/sfen-lists/
├── tier1-10k.txt              1 SFEN / 行、10,000 行
├── tier1-10k-meta.json        サンプリング統計 (下記スキーマ)
├── tier2-30k.txt              同上
└── tier2-30k-meta.json

training/data/piyo-raw/
├── {sfen_hash}/               SFEN の SHA-1 先頭 8 文字を hash として使う
│   ├── k1.png                 デザイン k1 でキャプチャ
│   ├── k3.png                 デザイン k3
│   ├── k5.png                 デザイン k5
│   ├── k101.png               ダーク k101
│   ├── k103.png               ダーク k103
│   ├── k105.png               ダーク k105
│   ├── k1_rotated.png         盤回転版 (向きトグル使用時のみ)
│   └── meta.json              下記スキーマ
├── {sfen_hash}/
│   └── ...
└── ...

training/data/piyo-full-screenshots/   (オプション、O2 で全画面も保存する場合)
└── {sfen_hash}/
    ├── k1_full.png
    └── ...
```

### 4.1 SFEN hash の作り方

```
sfen_hash = sha1(sfen_normalized).hex()[:8]
```

**正規化ルール** (MITO 側の `docs/ocr-metrics.md` に準拠):
- 手数フィールド (最後の数字) は除去
- 持ち駒は SFEN 慣習の R,B,G,S,N,L,P 順に統一
- 空白は 1 個で正規化
- 例: `"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -"` (手数無し)

同じ盤面 + 持ち駒 + 手番なら同じ hash になる。**同一 hash の SFEN は 1 度だけキャプチャで OK**。

### 4.2 meta.json スキーマ

`{sfen_hash}/meta.json` の内容:

```json
{
  "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
  "sfen_normalized": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -",
  "sfen_hash": "a1b2c3d4",
  "captured_at": "2026-07-06T10:00:00Z",
  "app_version": "6.5.0",
  "device": "iPhone 15 Pro",
  "captures": [
    {
      "file": "k1.png",
      "design_id": "k1",
      "rotated": false,
      "board_frame": { "x": 0, "y": 100, "w": 512, "h": 512 },
      "hand_black_frame": { "x": 0, "y": 620, "w": 512, "h": 60 },
      "hand_white_frame": { "x": 0, "y": 40, "w": 512, "h": 60 },
      "screen_size": { "w": 512, "h": 768 },
      "screenshot_full_file": "k1_full.png"
    },
    {
      "file": "k101.png",
      "design_id": "k101",
      "rotated": false,
      "board_frame": { "x": 0, "y": 100, "w": 512, "h": 512 },
      ...
    }
  ]
}
```

**必須フィールド**:
- `sfen`, `sfen_hash`, `captures[].file`, `captures[].design_id`

**推奨フィールド** (可能なら):
- `board_frame`, `hand_black_frame`, `hand_white_frame` — 学習ラベル自動化に大貢献
- `app_version`, `device` — 後で分布分析するときに使う

### 4.3 sampling meta.json スキーマ

`training/data/sfen-lists/tier1-10k-meta.json`:

```json
{
  "tier": "tier1",
  "sfen_count": 10000,
  "generated_at": "2026-07-06T10:00:00Z",
  "sampling_config": {
    "strata": [
      { "name": "opening_middle", "ratio": 0.3, "board_pieces": [25, 38], "promoted": [0, 1] },
      { "name": "middle", "ratio": 0.4, "board_pieces": [20, 30], "promoted": [1, 3] },
      { "name": "middle_end", "ratio": 0.2, "board_pieces": [15, 25], "promoted": [2, 5] },
      { "name": "endgame_promoted", "ratio": 0.1, "board_pieces": [10, 20], "promoted": [3, 999] }
    ],
    "turn_balance": { "b": 5000, "w": 5000 },
    "hand_distribution": { "0": 2000, "1-3": 3000, "4-8": 3000, "9+": 2000 }
  },
  "actual_stats": {
    "turn": { "b": 5001, "w": 4999 },
    "board_piece_dist": { "10-14": 200, "15-19": 1800, "20-24": 3500, "25-29": 3200, "30-38": 1300 },
    "promoted_count_dist": { "0": 3200, "1-2": 4300, "3-5": 2000, "6+": 500 },
    "piece_appearance_on_board": {
      "P": 210523, "L": 22100, "N": 22050, "S": 22030, "G": 22120,
      "B": 20050, "R": 20040, "K": 10001,
      "+P": 5200, "+L": 3100, "+N": 3100, "+S": 3050, "+B": 5050, "+R": 5100
    }
  }
}
```

**MITO 側でこの meta.json を見て「レア駒足りない → 追加抽出リクエスト」を判断する**。

---

## 5. 品質チェックリスト (依頼先で最低限確認してほしい)

Tier 1 (10,000 SFEN) キャプチャ完了時に、**目視でランダム 20 枚を確認**:

- [ ] 盤面 9×9 が完全に画像内に収まっている (端切れなし)
- [ ] 持ち駒エリアが画像内に収まっている
- [ ] 駒が意図した通りに配置されている (SFEN と実表示の一致)
- [ ] デザイン ID が正しい (k1 の画像に k101 の駒が出ていない、など)
- [ ] 盤回転指定時、実際に回転している
- [ ] ダーク指定時、実際にダーク配色になっている
- [ ] JPEG 圧縮アーティファクトが目立たない (PNG 保存が正しく効いている)
- [ ] 前局面の描画残滓 (キャッシュ) が写っていない

**問題があれば MITO 側に報告** (どのビューが古いキャッシュを保持しているか、次のバッチで修正)。

---

## 6. データ受け渡し方法

### 6.1 転送手段

以下のいずれかで OK (依頼先の都合に合わせる):

- **git-lfs** (小規模、Tier 1 のみ)
- **rsync / SSH tarball** (中規模、Tier 2)
- **クラウドストレージ** (R2, S3, Google Drive) (Tier 3)

### 6.2 ディレクトリ配置先

MITO レポの `training/data/` 配下に、上記 (Section 4) の構造で配置。

`training/` は現時点で存在しないが、Phase 2 開始時に MITO 側で作成する。**依頼先はこの構造で dump する tarball を用意してくれれば OK**。

### 6.3 検証用サブセット

Tier 1 の中から **100 SFEN** を「検証用」として先行配布してほしい。MITO 側でこれを使って:

- 学習パイプラインの動作確認
- augmentation の効き測定
- ラベル JSON スキーマの実装確認

を実施する。

**フォーマットは本番と同じ**、ディレクトリは `training/data/piyo-raw-verify/` に配置。

---

## 7. 具体的な作業タスクリスト (依頼先で確認)

### Task 1: ipa 静的解析 (半日)

- [ ] ぴよ将棋 iOS の ipa を取得 (JB 端末から `frida-ios-dump` などで抽出、または App Store 版を使用)
- [ ] Mach-O バイナリを解析、ObjC/il2cpp/Swift 構造を確認
- [ ] 以下の関数の RVA を特定:
  - 盤面 setter (SFEN or 内部形式を受ける関数)
  - 盤面 renderer (UIView 描画)
  - デザイン ID setter
  - 盤面回転トグル
- [ ] 結果を **`docs/piyo-ipa-analysis.md`** に転記 (MITO 側にも共有)

### Task 2: JB 端末環境確認 (半日)

- [ ] frida-server が JB 端末で常駐している
- [ ] 母艦から `frida-ls-devices` で JB 端末が見える
- [ ] `frida-trace` で盤面 setter 関数の呼び出しをトレースできる (Task 1 の RVA 検証)

### Task 3: Frida hook script 実装 (1〜2日)

- [ ] SFEN 入力 → 盤面セット の hook
- [ ] キャプチャ (盤面 UIView + 持ち駒 UIView → PNG)
- [ ] デザイン ID 切り替え
- [ ] 盤回転トグル
- [ ] SSH で母艦へ転送する自動化
- [ ] meta.json の生成

### Task 4: SFEN サンプリング (数時間)

- [ ] 80億プールから Tier 1 (10,000 SFEN) を Section 3 の条件で抽出
- [ ] sampling meta.json 生成
- [ ] `training/data/sfen-lists/tier1-10k.txt` に出力

### Task 5: Tier 1 キャプチャ (4〜6時間)

- [ ] 10,000 SFEN × 6 デザイン = 60,000 画像 生成
- [ ] meta.json 60,000 個生成 (もしくは統合形式で 10,000 個)
- [ ] 品質チェック (Section 5 のチェックリスト)
- [ ] MITO 側に受け渡し (Section 6)

### Task 6: Tier 2 キャプチャ (12〜18時間、Tier 1 で問題なければ)

- [ ] 30,000 SFEN × 8 デザイン = 240,000 画像 生成
- [ ] 同上

---

## 8. 質問・相談窓口

依頼先で判断に迷ったときは、以下の順で意思決定:

1. **SFEN の異常局面をどう扱うか**: すべて含める (validation スキップ)、UI が壊れた画像は捨てる
2. **UI トグルがうまく効かない**: MITO 側に相談、代替案 (SFEN 反転で盤回転を代替 等) を検討
3. **キャプチャ画像サイズ**: 迷ったら大きめ (512×512) にする、後で MITO 側でリサイズできる
4. **時間がかかりすぎる**: Tier 1 を 5,000 SFEN に減らして先行検証、その後拡大

MITO 側の依頼書として本文書を更新すること。**両レポで共通の版**として扱う。

---

## 9. 完了基準 (Definition of Done)

以下すべて達成で Task 完了:

- [ ] `docs/piyo-ipa-analysis.md` が作成されている (RVA + クラス名の地図)
- [ ] Tier 1 (10,000 SFEN) の PNG × 60,000 枚 と meta.json が MITO の `training/data/piyo-raw/` に配置されている
- [ ] `training/data/sfen-lists/tier1-10k-meta.json` に統計情報が記載されている
- [ ] 品質チェック (Section 5) を通過している
- [ ] MITO 側の学習パイプラインが Tier 1 データで少なくとも 1 エポック回せる (実際の精度は問わない)

---

## 10. 変更履歴

| 日付 | 変更内容 |
|---|---|
| 2026-07-06 | 初版作成 (MITO Phase 2 開始に合わせて) |
