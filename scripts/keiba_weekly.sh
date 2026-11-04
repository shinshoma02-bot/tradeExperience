#!/bin/sh
# 競馬の定期更新: 直近の結果を追加取得 → 今週の出馬表で予想+前向き記録 → 答え合わせ → 変更をコミットしてpush
# (木〜日の発走前に何度か回す。同じレースは「オッズ付きの最新の予想」が記録に残る)
set -e
cd "$(dirname "$0")/.."
pip install -q -r requirements-keiba.txt
git pull -q --rebase origin "$(git branch --show-current)" || true
YEAR=$(TZ=Asia/Tokyo date +%Y)
python -m keiba.scrape_all "$YEAR"       # 今年の未取得の開催日だけ追加(取得済みは飛ばす)
python -m keiba.model upcoming            # 今週の予想 + 前向き記録
python -m keiba.forward | tail -8         # 結果が出たレースの答え合わせ
git add keiba/data keiba/site/upcoming.json keiba/site/forward.json
if git diff --cached --quiet; then echo "変更なし"; else
  git commit -q -m "keiba: 定期更新 $(TZ=Asia/Tokyo date '+%F %H:%M')"
  git push -q origin HEAD
fi
