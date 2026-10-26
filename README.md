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

`Dockerfile` 付き。Google Cloud Run / Fly.io などでそのまま動く。
注意: 模擬ポジションと記録はメモリとローカルファイルなので、コンテナ再起動で消える
(本格運用前に永続ストレージへ。ペーパー検証中は 1 インスタンス固定にすること)。

## 次にやること

1. TradingViewで各マーケットのアラートを作り、数週間ペーパーで回す
2. `/report` で勝てているマーケット×戦略を絞る
3. 絞ったものだけ、`app/brokers/base.py` を継承して実ブローカー(取引所API)を実装
4. 実ブローカーは最初ごく少額、APIキーは出金権限なし
