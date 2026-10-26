# Fly.io へのデプロイ手順

## 0. 「クラウドのアカウント」って?
サーバーを24時間貸してくれる会社(Fly.io、Google、Amazonなど)の会員登録のこと。
自分のPCやこのチャット環境は止まったり消えたりするので、TradingViewの通知を24時間受けるには、
こういう会社のサーバーに置く必要がある。

## 1. Fly.io に登録(5分)
1. https://fly.io/app/sign-up で登録(メール or GitHub)
2. クレジットカードを登録(必須。無料枠のみでも登録が要る)
3. 料金は小さい1台で月に数ドル程度。最新の料金は公式の Pricing で確認すること

## 2. コマンドをインストール(`flyctl` = Fly.ioを操作するコマンド)
- Mac: `brew install flyctl`
- Windows(PowerShell): `iwr https://fly.io/install.ps1 -useb | iex`
- Linux: `curl -L https://fly.io/install.sh | sh`

```bash
fly auth login      # ブラウザが開くのでログイン
```

## 3. コードを取ってくる
```bash
git clone https://github.com/shinshoma02-bot/tradeExperience.git
cd tradeExperience
git checkout claude/trading-skeleton   # mainにマージ済みなら不要
```

## 4. アプリを作る(世界で1つの名前が必要)
```bash
fly launch --no-deploy --copy-config --name tradeexp-あなたの名前
```
途中で「設定を上書きするか?」と聞かれたら、既存の `fly.toml` を使う(No / コピーを選ぶ)。
Postgres や Redis は **作らない**(No)。

## 5. 記録用ディスクを作る(1GB)
```bash
fly volumes create data --region nrt --size 1
```

## 6. 合言葉(secret)を作って登録
```bash
# 長いランダム文字列を作る。表示されるので **メモ**(TradingView側にも入れる)
openssl rand -hex 32

fly secrets set WEBHOOK_SECRET=ここに上の文字列
```
(Windowsで `openssl` がない場合は、20文字以上の適当な英数字でOK)

## 7. デプロイ
```bash
fly deploy
fly scale count 1      # 1台に固定(模擬ポジションはメモリ上なので複数台だとズレる)
```

## 8. 動作確認
```bash
fly status
curl https://tradeexp-あなたの名前.fly.dev/health
# → {"ok":true,"mode":"paper"} が返ればOK

fly logs               # ログを見る
```

手動で信号を送ってテスト:
```bash
export WEBHOOK_SECRET=手順6の文字列
export BASE_URL=https://tradeexp-あなたの名前.fly.dev
python scripts/send_signal.py --symbol BTCUSDT --action buy --price 65000 --qty 0.01
curl $BASE_URL/report
```

## 9. TradingView に設定
- Webhook URL: `https://tradeexp-あなたの名前.fly.dev/webhook`
- Pine Script の `secret` 入力欄に手順6の文字列
- アラートのメッセージ欄: `{{strategy.order.alert_message}}`

## 困ったとき
| 症状 | 見る場所 |
|---|---|
| 起動しない | `fly logs`。`WEBHOOK_SECRET が未設定` なら手順6をやり直す |
| 401 が返る | secret がTradingView側とずれている |
| 注文が `rejected` | `/report` ではなく `fly ssh console` で `cat /srv/data/journal.jsonl` に理由が残る |
| 全部止めたい | `fly secrets set KILL_SWITCH=true` |
| 完全に消したい | `fly apps destroy アプリ名`(課金も止まる) |

## 注意
- 今は模擬売買のみ(`TRADING_MODE=paper`)。お金は動かない。
- 模擬ポジションは再起動で消える。記録(journal)はディスクに残る。
