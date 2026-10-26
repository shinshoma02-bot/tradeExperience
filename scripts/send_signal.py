"""TradingViewなしでWebhookを試す用。例:
  python scripts/send_signal.py --market crypto --symbol BTCUSDT --action buy --price 65000 --qty 0.01
環境変数 WEBHOOK_SECRET と、必要なら BASE_URL(既定 http://localhost:8080)を使う。
"""
import argparse
import os

import json
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--market", default="crypto")
p.add_argument("--strategy", default="manual")
p.add_argument("--symbol", required=True)
p.add_argument("--action", choices=["buy", "sell", "close"], required=True)
p.add_argument("--price", type=float, required=True)
p.add_argument("--qty", type=float, default=0)
a = p.parse_args()

body = json.dumps({"secret": os.environ["WEBHOOK_SECRET"], **vars(a)}).encode()
req = urllib.request.Request(
    os.environ.get("BASE_URL", "http://localhost:8080") + "/webhook",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    print(r.status, r.read().decode())
