# tradeExperience

TradingViewのアラート(Webhook)を受けて、複数マーケットで**模擬売買(ペーパートレード)**する骨組み。
実際のお金は一切動かない。まず「どのマーケット×戦略が勝てそうか」を比べるのが目的。

```
TradingView アラート ──Webhook──▶ /webhook ──▶ リスク関門 ──▶ マーケット別ブローカー(今は模擬) ──▶ journal.jsonl
                                                                                          └▶ /report で成績比較
```

## 動かし方

```bash
pip install -r requirements.txt
cp .env.example .env           # WEBHOOK_SECRET を長いランダム文字列に変える
set -a; . ./.env; set +a
uvicorn app.server:app --port 8080
pytest                          # テスト
```

| URL | 役割 |
|---|---|
| `POST /webhook` | シグナル受信 |
| `GET /positions` | 現在の模擬ポジション |
| `GET /report` | `マーケット/戦略` ごとの確定損益 |
| `GET /health` | 死活確認 |

## TradingView側の設定

アラート作成 → 通知タブで Webhook URL に `https://<デプロイ先>/webhook` を指定(**有料プラン必須**)。
メッセージ欄に次のJSONを入れる(`{{close}}` は発火時の価格に自動置換される):

```json
{"secret":"<WEBHOOK_SECRET>","market":"crypto","strategy":"sma_cross","symbol":"BTCUSDT","action":"buy","price":{{close}},"qty":0.01}
```

- `market`: `crypto` / `fx` / `us_stock` / `jp_stock` / `futures`
- `action`: `buy` / `sell` / `close`(全決済。qty不要)
- `strategy`: 好きな名前。戦略ごとに成績が分かれる

## 安全装置(`.env`)

1注文の上限金額、1日の注文回数、1日の損失上限、`KILL_SWITCH=true` で全停止。
弾かれた注文も `journal.jsonl` に理由つきで残る。

## クラウドにデプロイ

`Dockerfile` と `fly.toml` 付き。手順は [docs/deploy-flyio.md](docs/deploy-flyio.md)。
注意: 模擬ポジションと記録はメモリとローカルファイルなので、コンテナ再起動で消える
(本格運用前に永続ストレージへ。ペーパー検証中は 1 インスタンス固定にすること)。

## 次にやること

1. TradingViewで各マーケットのアラートを作り、数週間ペーパーで回す
2. `/report` で勝てているマーケット×戦略を絞る
3. 絞ったものだけ、`app/brokers/base.py` を継承して実ブローカー(取引所API)を実装
4. 実ブローカーは最初ごく少額、APIキーは出金権限なし

## 戦略ランナー(TradingView不要・無料)

`RUNNER_ENABLED=true` にすると、サーバー自身が Kraken の公開API(キー不要)から1時間足を取り、
`pine/` と同じ3戦略(SMAクロス/RSI逆張り/ブレイクアウト)を計算して、模擬売買する。
信号は webhook と同じ経路(リスク確認→模擬ブローカー→記録)を通るので、`/report` で成績が見られる。
- 起動直後は最新の足を処理済みにするだけ(再起動での二重発注防止)。新しい確定足から売買する
- 1時間足なので、最初の売買まで数時間〜数日かかることもある(クロスやブレイクが出るまで待つ)
- 動作確認: `GET /health` の `runner` が `true`、ログに `runner:` が出る

## Pine Script(`pine/`)

| ファイル | タイプ | 中身 |
|---|---|---|
| `sma_cross.pine` | トレンドフォロー | SMAのゴールデン/デッドクロス |
| `rsi_reversion.pine` | 逆張り | RSIの売られすぎ/買われすぎからの戻り |
| `donchian_breakout.pine` | ブレイクアウト | 直近高値/安値の更新 |

使い方: TradingViewのPineエディタに貼る → チャートに追加 → アラート作成で条件にその戦略を選び、
メッセージ欄に `{{strategy.order.alert_message}}` と入れる → Webhook URLを指定。
`secret` 入力欄に `.env` の値を入れる(**スクリプトは公開しない**こと。公開するとsecretが漏れる)。
タイプの違う戦略を同じ銘柄で回すと、相場環境ごとの向き不向きが `/report` で比べられる。

TradingViewなしの動作確認は `python scripts/send_signal.py --symbol BTCUSDT --action buy --price 65000 --qty 0.01`。
