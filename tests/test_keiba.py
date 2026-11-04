import numpy as np
import pandas as pd

from keiba.model import race_normalize
from keiba.upcoming import parse_entries

HTML = """
<table class="basic narrow-xy striped"><caption>
<div class="date_line"><div class="cell date">2026年11月1日（日曜） 4回東京11日</div><div class="cell time">発走時刻：<strong>15時40分</strong></div></div>
<div class="race_number"><img alt="11レース"></div>
<div class="race_title"><h2><span class="race_name">天皇賞(秋)<img alt="GⅠ"></span></h2>
<div class="type"><div class="cell category">3歳以上</div><div class="cell class">オープン</div>
<div class="cell course">コース：2,000<span class="unit">メートル</span><span class="detail">（芝・左）</span></div></div></div>
</caption><tbody>
<tr><td class="waku">1</td><td class="num">1</td><td class="horse"><a>アー</a></td><td class="h_weight">438 kg (+8)</td>
<td class="age">牡4</td><td class="weight">58.0 kg</td><td class="jockey"><a>C.ルメール</a></td><td class="trainer"><a>和田 勇介</a></td><td class="odds">2.1</td></tr>
<tr><td class="waku">2</td><td class="num">2</td><td class="horse"><a>ベー</a></td><td class="h_weight">470 kg (-2)</td>
<td class="age">牝5</td><td class="weight">56.0 kg</td><td class="jockey">▲ 木幡 巧也</td><td class="trainer"><a>牧 光二</a></td><td class="odds">5.5</td></tr>
</tbody></table>
"""


def test_parse_entries_reads_race_and_odds_rank():
    rows = parse_entries(HTML)
    assert len(rows) == 2
    a, b = rows
    assert (a["venue"], a["race_no"], a["grade"], a["ground"], a["dist"]) == ("東京", 11, "G1", "芝", 2000)
    assert (a["horse"], a["sex"], a["age"], a["bw"], a["bw_diff"], a["weight"]) == ("アー", "牡", 4, 438.0, 8.0, 58.0)
    assert a["jockey"] == "C.ルメール" and b["jockey"] == "木幡巧也"  # 見習い印と空白は除く(過去データの表記に合わせる)
    assert (a["pop"], b["pop"]) == (1, 2)


def test_parse_entries_without_odds_has_no_popularity():
    rows = parse_entries(HTML.replace(">2.1<", ">---<").replace(">5.5<", ">---<"))
    assert [r["pop"] for r in rows] == [0, 0]


def test_race_normalize_sums_to_total_per_race():
    p = np.array([0.2, 0.3, 0.5, 0.1, 0.1])
    ids = pd.Series(["a", "a", "a", "b", "b"])
    out = race_normalize(p, ids, 1.0)
    assert abs(out[:3].sum() - 1) < 1e-9 and abs(out[3:].sum() - 1) < 1e-9
