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
| `GET /summary` | 元手(円)に対する今の損益・合計資産(円)。想定手数料込み、SPY持ちっぱなしとの比較、実資金に進む合格ライン(`real_money_gates`)つき |
| `GET /backtest` | 過去の日足で全銘柄×3戦略を試した結果(`?days=3650`)。重いので1時間キャッシュ |
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

`RUNNER_ENABLED=true` にすると、サーバー自身が無料の価格データを取り、3戦略を計算して模擬売買する。
信号は webhook と同じ経路(リスク確認→模擬ブローカー→記録→通知)を通る。

**前提は「元手1万円」**(`CAPITAL_JPY`)。銘柄×戦略に均等に配分する(既定は4銘柄×3戦略=1ボット約833円)。

| マーケット | データ源 | 既定の銘柄 |
|---|---|---|
| fx | Yahoo Finance(非公式) | USDJPY, EURJPY, AUDJPY |
| us_stock | Yahoo Finance | SPY(S&P500連動ETF) |
| crypto | Kraken(公開API) | 既定では外した(変動が大きすぎるため)。足せる |

先物は最小単位が大きく1万円では成り立たないので既定から外した。日本株は1株が配分額を超えがち
(トヨタ約3,000円 vs 1ボット約833円)で外したが、`RUNNER_INSTRUMENTS` で足せる。書式は `.env.example`。

- **日足**(`RUNNER_INTERVAL_MIN=1440`)。1時間足は売買が多すぎて手数料負けしやすかった
- 3戦略(`app/runner/strategies.py`): SMAクロス(50/100)、ブレイクアウト(55日高値/20日安値)、RSI逆張り(14, 30→50)。
  パラメータは過去10年の日足で検証した定番値。検証で一番良く見えた値を選ぶと過去に合わせただけになるので、あえて定番にした
- **現物(暗号資産・株)は空売りしない**。売りシグナルは「持っていれば決済」だけ。売りから入れるのはFX
- 日本株は1株単位。配分額で1株も買えなければ `skipped`(ログのみ)
- 起動直後は最新の足を処理済みにするだけ(再起動での二重発注防止)。日足なので新しい足は1日1回
- 日足の判定はその日の確定後(取引所の日付変わり以降)。「終値で判断して翌日に約定する」程度の遅れがある
- Yahoo Finance は非公式APIなので、突然仕様が変わる/サーバーのIPからだと弾かれることがある。
  `fetch failed` が5回連続で全部失敗すると通知が来る
- 含み損益の評価価格は、直近の確定足(日足)の終値

円換算は `JPY_PER_USD` の固定レート(概算)。`GET /summary` で元手に対して今いくらか(円)が見られる。

## バックテスト

`/backtest` は、過去の日足(Yahoo系は最大10年、Kraken は約2年分)に3戦略を当てはめた結果を返す。
売買ルールはランナーと共通(`app/runner/runner.py` の `plan_orders`)。簡略化している点:
約定は足の終値、スリッページなし、手数料は片道(既定: 暗号資産0.2%/FX0.02%/株0.1%。`?fee_bps=` で固定可)、1ボットの配分額で全力買い。
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

**注意: `pine/` は旧1時間足版のパラメータのまま。ランナーの日足版とは別物。** 使い方: TradingViewのPineエディタに貼る → チャートに追加 → アラート作成で条件にその戦略を選び、
メッセージ欄に `{{strategy.order.alert_message}}` と入れる → Webhook URLを指定。
`secret` 入力欄に `.env` の値を入れる(**スクリプトは公開しない**こと。公開するとsecretが漏れる)。
タイプの違う戦略を同じ銘柄で回すと、相場環境ごとの向き不向きが `/report` で比べられる。

TradingViewなしの動作確認は `python scripts/send_signal.py --symbol BTCUSDT --action buy --price 65000 --qty 0.01`。
