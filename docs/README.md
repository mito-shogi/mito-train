# docs/

MITO ↔ mito-train 契約と mito-train 固有ドキュメント。

## MITO ↔ mito-train 契約

- `ocr-model-interface.md` — モデル契約 (input/output 名・shape・dtype)
- `ocr-metrics.md` — 評価指標の定義 (piece accuracy / IoU / exact-match)

## mito-train 固有

- `piyo-piece-templates.md` — 駒テンプレの命名規則 (14 デザイン × 30 駒)
- `backbones.md` — BoardOCR がサポートする 9 バックボーンの params / 用途 / 配信先まとめ
- `ocr-scaling-outlook.md` — sweep 結果からのエポック数・画像サイズ増の効き見込み (w224 版, 2026-07-12)
- `w384-sweep-analysis.md` — w384 backbone sweep の考察 (収束速度・過剰 epoch・train↔val ギャップ, 2026-07-13)
- `resolution-tradeoff.md` — 学習/推論解像度 (w224 vs w384) の運用トレードオフと配信別の推奨

## データセットの正規

学習/評価データは Hugging Face Hub に集約：

- [ultemica/piyoshogi](https://huggingface.co/datasets/ultemica/piyoshogi) — 学習/検証 (paired 形式、`ocr_paired` / `detector_paired` config)
- [ultemica/piyoshogi-eval](https://huggingface.co/datasets/ultemica/piyoshogi-eval) — leak-free eval (paired 形式、1,000 SFEN × 4機種)

各リポの README に schema・行数・使い方が記載されている。
