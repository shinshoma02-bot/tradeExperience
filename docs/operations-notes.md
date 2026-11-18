# 運用メモ(エージェントチーム)

- ブランチ: 各セッションの指定ブランチで作業。`claude/trading-skeleton` への push は Fly 自動デプロイが走る(コード・画面が変わったときだけ)
- コミットメッセージ末尾に Co-Authored-By と Claude-Session を付ける。PR は頼まれたときだけ
- 完了済みの依頼・古い引き継ぎは `archive/` に移す(消さない)。`archive/README.md` 参照
- 削除は `docs/cleanup-candidates.md` の一覧をオーナーが承認してから
- 仕様の正(wadome-tools の busei-system/SPEC.md、career-car-game の運用)は、この環境から読めていない。読めたら差分を `docs/team-intro-diff.md` に追記する
