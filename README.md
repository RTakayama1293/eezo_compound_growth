# EEZO 複利成長シミュレーター

北海道食材EC「EEZO」における開封体験品質への投資が生む**複利効果**をシミュレーションするツール。

## 背景

食品ECにおいて、開封体験は単なる「おまけ」ではなく、複利的な成長エンジンになりうる。

```
高品質体験 → 感動 → 再購入↑ + UGC投稿↑
                          ↓
                    UGC蓄積 → 信頼↑ → CVR↑
                          ↓
                    顧客増 × 高リピート = さらなるUGC
                          ↓
                       複利成長ループ
```

## クイックスタート

```bash
# 依存インストール
pip install -r requirements.txt

# シミュレーション実行
python src/simulation.py
```

## 出力例

- 24ヶ月後の累計顧客数比較（投資なし vs 投資あり）
- UGC蓄積推移
- CVR変化
- 月次購入数の推移

## パラメータ調整

`src/simulation.py` 内の `SimulationParams` クラスを編集してシナリオを変更可能。

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| experience_quality | 開封体験品質 (1-10) | 5 / 8 |
| base_repurchase_rate | 基準再購入率 | 15% |
| quality_to_repurchase | 品質→再購入率係数 | 3%/pt |
| quality_to_ugc | 品質→UGC率係数 | 1.5%/pt |

## Claude Code on the Webで使う

このリポジトリはCCW（Claude Code on the Web）での実行を想定。

1. GitHubにプッシュ
2. https://claude.ai/code でリポジトリを選択
3. 「シミュレーションを実行して」と指示

## ライセンス

Internal use only - 新日本海商事
