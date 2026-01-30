# EEZO 複利成長シミュレーター

## プロジェクト概要

北海道食材EC「EEZO」における**開封体験品質への投資が生む複利効果**をシミュレーションするツール。

### ビジネス仮説

```
開封体験品質↑ → 感動度↑ → 再購入意向↑ + UGC投稿確率↑
                                ↓
UGC蓄積 → 信頼シグナル↑ → 新規CVR↑ → 顧客ベース拡大
                                ↓
                    拡大顧客 × 高再購入率 = さらなるUGC
                                ↓
                           複利的成長ループ
```

## 使い方

### 基本実行

```bash
python src/simulation.py
```

### パラメータ変更

`src/simulation.py` 内の `SimulationParams` を編集してシナリオを変更。

## ディレクトリ構成

```
├── src/
│   └── simulation.py    # メインシミュレーションコード
├── data/                # 入力データ（あれば）
├── outputs/             # グラフ・レポート出力先
└── requirements.txt
```

## 主要パラメータ

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| experience_quality | 開封体験品質スコア (1-10) | 5 (通常) / 8 (投資後) |
| base_repurchase_rate | 基準再購入率 | 0.15 |
| quality_to_repurchase | 品質→再購入率への変換係数 | 0.03 |
| quality_to_ugc | 品質→UGC投稿確率への変換係数 | 0.01 |
| ugc_to_cvr_boost | UGC件数→CVR改善への寄与 | 0.001 |
| initial_customers | 初期顧客数 | 500 |
| monthly_visitors | 月間新規流入数 | 1000 |
| base_cvr | 基準CVR | 0.02 |

## 出力

- `outputs/compound_growth.png`: 投資あり/なしの比較グラフ
- コンソール: 月次サマリーと複利効果の数値

## 開発メモ

- Python 3.9+
- 依存: matplotlib, numpy, dataclasses (標準)
