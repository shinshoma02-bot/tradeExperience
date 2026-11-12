# CIテスト失敗の調査結果（2026-11-12・うみ調べ）

## 状況
- リポジトリ: shinshoma02-bot/tradeExperience
- ブランチ: claude/trading-skeleton
- ワークフロー: test（pytest、Python 3.13）
- 失敗したラン（2026-11-12）: #88 15:00（8400bec）、#89 16:22（7d7ac48）、#90 17:26（ca7f127）、#91 18:21（bb60827）
- 同じブランチで11月4日・5日にも同じワークフローの失敗メールが多数ある。今日だけの不具合ではない。
- 本番への影響なし。2026-11-12 18:50頃に https://tradeexperience.fly.dev/boat/ と /keiba/ がともに 200 で応答することを確認済み。

## 失敗しているテスト（ローカル再現済み）
- `tests/test_boat.py::test_live_probs_match_model_formula`
- ローカル再現（同ブランチの最新コミット bb60827 をクローンし、requirements.txt / requirements-boat.txt / requirements-keiba.txt を入れて `python -m pytest -q` を実行）: 61 passed、1 failed。
- エラー: `pre["races"][0]` で `IndexError: list index out of range`（tests/test_boat.py:272）。

## 原因
- `boat/site/pre_14.json`（built 2026-11-12）の `races` が空配列（0件）。
- 会場14にその日の直前予想データがない日に、テストが `races[0]` を前提として読むため落ちる。
- 参考: 同日の `boat/site/pre_16.json` は races が12件ある。
- 定期更新コミット（boat/keiba のデータ更新）自体の変更が原因ではなく、テストがデータの有無に依存している設計が原因。更新のたびに pre_14.json が空で作り直され、毎回同じテストが落ちる。

## 直してほしいこと（Claude Code担当）
- `test_live_probs_match_model_formula` がレースがない日でも落ちないようにする。例: races が空なら skip する、レースがある会場（例: 16）のデータを使う、または固定のフィクスチャデータで計算する。
- コード修正はこちら（Claude Code）の担当。うみは修正しない。

## 確認してほしいこと
- 修正後、次回の定期更新コミットで test ワークフローが通ること。
- 直前予想のサーバー側計算とモデル計算の一致自体は、このテスト以外では通っている（61 passed）。

