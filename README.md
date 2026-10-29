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
| `GET /positions` | 現在の模擬ポジション(含み損益つき) |
| `GET /report` | `マーケット/戦略` ごとの確定損益・含み損益・合計(USD換算) |
| `GET /backtest` | 過去の足で全銘柄×3戦略を試した結果(`?days=180&fee_bps=5`)。重いので1時間キャッシュ |
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

含み損益の評価価格は、直近の確定足(1時間足)の終値。最大で約1時間遅れる。Webhook経由のポジションは評価価格が無いので含み損益は出ない。

`RUNNER_ENABLED=true` にすると、サーバー自身が無料の価格データを取り、`pine/` と同じ3戦略
(SMAクロス/RSI逆張り/ブレイクアウト)を計算して模擬売買する。信号は webhook と同じ経路
(リスク確認→模擬ブローカー→記録→通知)を通り、`/report` で成績が見られる。

| マーケット | データ源 | 初期の銘柄 |
|---|---|---|
| crypto | Kraken(公開API) | XBTUSD, ETHUSD |
| fx | Yahoo Finance(非公式) | USDJPY, EURUSD |
| us_stock | Yahoo Finance | AAPL, NVDA |
| jp_stock | Yahoo Finance | トヨタ(7203.T) |
| futures | Yahoo Finance | ES=F(S&P500先物) |

銘柄は `RUNNER_INSTRUMENTS` で変えられる(書式は `.env.example`)。
- 1時間足。**確定した足だけ**を見る。株・FXは市場が閉まっている間は新しい足が来ないので何も起きない
- 起動直後は最新の足を処理済みにするだけ(再起動での二重発注防止)
- **株は空売りしない**(売りシグナルは、持っていれば決済するだけ)
- 円建て銘柄は `JPY_PER_USD` の固定レートでUSDに換算(上限判定・`/report`)。ごく概算
- Yahoo Finance は非公式APIなので、突然仕様が変わる/サーバーのIPからだと弾かれることがある。
  その場合ログに `fetch failed` が出て、5回連続で全部失敗すると通知が来る
- 先物はレバレッジ・限月・証拠金を模擬していない。値動きの方向を見るだけの参考値

## バックテスト

`/backtest` は、直近の足(Yahoo系は最大 `days` 日、Kraken は約30日分)に3戦略を当てはめた結果を返す。
売買ルールはランナーと共通(`app/runner/runner.py` の `plan_orders`)。簡略化している点:
約定は足の終値、スリッページなし、手数料は片道 `fee_bps`、1注文は一定額。
**過去で勝てても未来は保証されない**。24通りの中のベストを選ぶと、たまたま勝っただけの組み合わせを拾う(過剰適合)ので、
期間を変えても同じ傾向が出るか、売買回数が十分かを見ること。

## 再起動での復元

ポジションとその日のリスク集計は、記録(`journal.jsonl` の約定履歴)を起動時に再生して復元する。
ディスク(Fly の volume)が無いと復元できない。再デプロイ・再起動してもポジションは消えない。

## スマホ通知(ntfy)

無料アプリ **ntfy**(iPhone/Android)に通知を送る。アカウント不要。
1. アプリをインストール → 「+」→ トピック名に、推測されにくい長い文字列(例 `tradeexp-` + ランダム英数字)を入れて購読
2. Fly.io の Secrets に `NTFY_TOPIC` = 同じ文字列を登録 → 再起動
3. `/health` の `notify` が `true` になればOK

通知されるもの: 約定(損益つき)、リスク制限による拒否(同じ理由は1時間に1回)、価格データ取得の5回連続失敗、サーバー起動。
**トピック名を知っている人は誰でも読み書きできる**ので、公開しないこと。

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
