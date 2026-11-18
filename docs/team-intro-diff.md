# 導入時の差分報告(R1、2026-11-18)

既存の CLAUDE.md・ルールは上書きしていない(末尾に節を追加しただけ)。

## ぶつかる/注意が要る点
1. **仕様の正を読めていない**: R1 が指す wadome-tools の `busei-system/SPEC.md` と career-car-game の運用は、この環境のリポジトリ範囲外で、接続(add_repo)が自動判定で拒否された。R1 本文の記述だけを仕様として組んだ。→ オーナーが接続を許可するか SPEC.md を貼れば、差分を調整する
2. **旧名**: `README.md` 冒頭・`fly.toml`・`deploy-fly.yml`・CLAUDE.md に tradeExperience/tradeexperience が残る。リポジトリ名は wadome-predictions に改名済みだが、Fly のアプリ名は変えると本番が別物になるので触っていない。README の書き直しは CON 部の課題
3. **CLAUDE.md の2つの役割**: 既存は「プロジェクトメモ」、R1 は「部の編成表・共通ルール・番号体系・連絡箱の場所」も入れる指定。1ファイルに同居させた(約束事と編成表が別の節)。分けたい場合はオーナーに確認
4. **連絡箱の形**: SPEC.md が読めないので、1ファイルの表(`inbox/INBOX.md`)にした。SPEC.md が1件1ファイル方式なら変える
5. **ブランチ**: このセッションの指定は `claude/determined-ritchie-c0iewy`。CLAUDE.md の自動デプロイ用 `claude/trading-skeleton` には push していない(デプロイは走らない)
