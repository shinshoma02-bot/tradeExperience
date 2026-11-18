# DEV 部(設計・実装)

## 担当
- `app/`(FastAPI・ランナー・リスク・ブローカー・maintenance)、`boat/`・`keiba/`・`pachinko/` のコード、`tests/`、`.github/workflows/`、`Dockerfile`、`fly.toml`、`scripts/`

## やること
- 機能の設計と実装、バグ修正、テスト(`pytest`)、デプロイ設定
- 画面(`*/site/index.html`)のコード変更。データJSONの中身は触らない(DAT部)

## ルール
- push前に `pytest` を通す。コードや画面が変わると Fly に自動デプロイされる点に注意(CLAUDE.md「デプロイ」)
- 売買ルールは `plan_orders` に集約(本番とバックテストで共通)。壊さない
- 他部の担当ファイルを直したいときは連絡箱(`inbox/INBOX.md`)で依頼する
- 作業の区切りで `memos/MEMO_DEV.md` に残す
