#!/bin/sh
# 丸亀ボートの定期更新: 取得 → 展示・オッズ → 予想+前向き記録 → 変更をコミットしてpush
# (締切前に1日数回回す。同じレースは最後の記録が評価に使われる)
set -e
cd "$(dirname "$0")/.."
python -m pip install -q -r requirements-boat.txt
python -m boat.cache unpack
TODAY=$(TZ=Asia/Tokyo date +%F)
FROM=$(TZ=Asia/Tokyo date -d "7 days ago" +%F)
python -m boat.scrape "$FROM" "$TODAY"   # 直近1週間の穴埋め(取得済みは取りに行かない)
python -m boat.odds_hist unpack
python -m boat.odds_hist fetch "$FROM"   # 直近の締切時オッズ(検証用)
python -m boat.upcoming
python -m boat.model | tail -3
python -m boat.cache pack
python -m boat.odds_hist pack
git add boat/site/data.json boat/site/live.json boat/data
if git diff --cached --quiet; then echo "変更なし"; else
  git commit -q -m "boat: 定期更新 $(TZ=Asia/Tokyo date '+%F %H:%M')"
  git push -q origin HEAD
fi
