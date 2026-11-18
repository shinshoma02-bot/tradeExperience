# DAT 部(データ)

## 担当
- データ取得・検証・モデル: `keiba/scrape*.py`・`keiba/model.py`、`boat/scrape.py`・`boat/model.py`・`boat/odds_hist.py`、`pachinko/ingest.py`、`research/`
- データ本体: `keiba/data/`、`boat/data/`、各 `site/*.json`、`docs/research-*.md`、`docs/pachinko-sources-*.md`

## やること
- 定期更新(`scripts/keiba_weekly.sh`・`scripts/boat_daily.sh`)、検証、前向き記録(`forward.py`)
- 結果は「効かなかった」も成果として記録。多重比較・ルックアヘッドに注意(CLAUDE.md「検証で分かったこと」)

## ルール
- 公式データの利用規約を守る(robots・サイトポリシー。CLAUDE.md「約束事」)。回避しない
- 極端な数字はデータ異常を疑う。ベンチマークと比べる。勝てる保証はないと正直に書く
- 作業の区切りで `memos/MEMO_DAT.md` に残す
