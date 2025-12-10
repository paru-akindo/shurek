# main.py
"""
水準評価モード専用アプリケーション
- 部品レベル入力を 2行×7列 のプルダウンで行う
- 各サイクルで得られる reward_table の報酬名別確率分布を表で表示
- 次に上げたい部品と現在の玉龍幣残高を評価前に入力できるようにした
  （評価実行ボタンを押すと、評価結果と到達予定時刻・残り時間・アップ後期待値を表示）
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import math

# === 定数・テーブル定義 ===

upgrade_info = {
    "受付_A": [
        {"level": 1, "time": 45, "cost": 0},
        {"level": 2, "time": 40, "cost": 1000},
        {"level": 3, "time": 35, "cost": 8250},
        {"level": 4, "time": 30, "cost": 14700},
        {"level": 5, "time": 25, "cost": 19900},
        {"level": 6, "time": 20, "cost": 27600},
        {"level": 7, "time": 15, "cost": 40000},
    ],
    "受付_B": [
        {"level": 1, "time": 30, "cost": 0},
        {"level": 2, "time": 15, "cost": 17250},
        {"level": 3, "time": 7.5, "cost": 44650},
    ],
    "計測_A": [
        {"level": 1, "cost": 0},
        {"level": 2, "cost": 1000},
        {"level": 3, "cost": 8250},
        {"level": 4, "cost": 14700},
        {"level": 5, "cost": 19900},
        {"level": 6, "cost": 27600},
        {"level": 7, "cost": 40000},
    ],
    "計測_B": [
        {"level": 1, "time": 30, "cost": 0},
        {"level": 2, "time": 15, "cost": 17250},
        {"level": 3, "time": 7.5, "cost": 44650},
    ],
    "教室_A": [
        {"level": 1, "cost": 0},
        {"level": 2, "cost": 1000},
        {"level": 3, "cost": 8250},
        {"level": 4, "cost": 14700},
        {"level": 5, "cost": 19900},
        {"level": 6, "cost": 27600},
        {"level": 7, "cost": 40000},
    ],
    "教室_B": [
        {"level": 1, "time": 60, "cost": 0},
        {"level": 2, "time": 30, "cost": 6900},
        {"level": 3, "time": 20, "cost": 11500},
        {"level": 4, "time": 15, "cost": 23000},
        {"level": 5, "time": 12, "cost": 31850},
        {"level": 6, "time": 10, "cost": 58800},
    ],
}

gym_A_levels = {
    1: ((2, 8), 0.15),
    2: ((7, 13), 0.20),
    3: ((12, 18), 0.25),
    4: ((15, 25), 0.30),
    5: ((22, 38), 0.35),
    6: ((37, 53), 0.40),
    7: ((52, 68), 0.45),
    8: ((65, 85), 0.50),
}

class_A_levels = {
    1: (7, 13),
    2: (10, 20),
    3: (17, 33),
    4: (30, 50),
    5: (47, 74),
    6: (70, 100),
    7: (97, 133),
}

# (合計ポイント low–high, 金貨)
reward_table = [
    (0, 79, 2),
    (80, 139, 5),
    (140, 219, 10),
    (220, 339, 15),
    (340, 459, 20),
    (460, 479, 25),
    (480, 619, 35),
    (620, 819, 40),
    (820, 1039, 50),
]

# 金貨量 → 報酬名
reward_names = {
    2:  "料理初心者",
    5:  "見習い料理人",
    10: "入門料理人",
    15: "初級料理人",
    20: "中級料理人",
    25: "上級料理人",
    35: "超級料理人",
    40: "特級厨師",
    50: "主席厨師",
    60: "厨神",
}

# コード → 部品名
code_to_part = {
    "1a": "受付_A", "1b": "受付_B",
    "2a": "計測_A", "2b": "計測_B",
    "3a": "教室_A1", "4a": "教室_A2", "5a": "教室_A3",
    "6a": "教室_A4", "7a": "教室_A5",
    "3b": "教室_B1", "4b": "教室_B2", "5b": "教室_B3",
    "6b": "教室_B4", "7b": "教室_B5",
}

# === データモデル ===

@dataclass
class EvaluationParams:
    levels: Dict[str, int]
    risk_factor: float = 1.0

@dataclass
class EvaluationResult:
    total_level: int
    cycle_reward: float
    cycle_time: int
    coin_rate: float
    hourly_rate: float

# === PMF ヘルパー ===

def convolve_pmfs(p1: Dict[int, float], p2: Dict[int, float]) -> Dict[int, float]:
    res = defaultdict(float)
    for v1, pr1 in p1.items():
        for v2, pr2 in p2.items():
            res[v1 + v2] += pr1 * pr2
    return dict(res)

def pmf_uniform(a: int, b: int, n: int) -> Dict[int, float]:
    if n <= 0:
        return {0: 1.0}
    base = 1.0 / (b - a + 1)
    pmf = {x: base for x in range(a, b + 1)}
    for _ in range(1, n):
        pmf = convolve_pmfs(pmf, {x: base for x in range(a, b + 1)})
    return pmf

def merge_pmfs(pmfs: List[Dict[int, float]]) -> Dict[int, float]:
    res = {0: 1.0}
    for pmf in pmfs:
        res = convolve_pmfs(res, pmf)
    return res

# === 期待値・レート計算 ===

_expected_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}

def get_classroom_hist(levels: Dict[str, int]) -> Tuple[int, ...]:
    counts = {i: 0 for i in range(1, 8)}
    for i in range(1, 6):
        lvl = levels.get(f"教室_A{i}", 1)
        counts[lvl] += 1
    return tuple(counts[i] for i in range(1, 8))

def combine_classroom_distribution(hist: Tuple[int, ...]) -> Dict[int, float]:
    pmfs = []
    for lvl, count in enumerate(hist, start=1):
        if count > 0:
            a, b = class_A_levels[lvl]
            pmfs.append(pmf_uniform(a, b, count))
    return merge_pmfs(pmfs) if pmfs else {0: 1.0}

def expected_cycle_reward_compressed(
    gym_level: int,
    class_hist: Tuple[int, ...]
) -> float:
    key = (gym_level, class_hist)
    if key in _expected_cache:
        return _expected_cache[key]
    (gmin, gmax), p_double = gym_A_levels[gym_level]
    gym_pmf = {x: 1.0/(gmax-gmin+1) for x in range(gmin, gmax+1)}
    class_pmf = combine_classroom_distribution(class_hist)
    total_pmf = convolve_pmfs(gym_pmf, class_pmf)
    exp_val = 0.0
    for pts, prob in total_pmf.items():
        coin = next((c for low, high, c in reward_table if low <= pts <= high), 0)
        exp_val += coin * (1 + p_double) * prob
    _expected_cache[key] = exp_val
    return exp_val

def compute_cycle_time(levels: Dict[str, int]) -> int:
    times: List[float] = []
    for part in ["受付_A", "受付_B", "計測_B"]:
        t = next((it.get("time") for it in upgrade_info[part]
                  if it["level"] == levels.get(part, 1)), None)
        if t is not None:
            times.append(t)
    for i in range(1, 6):
        t = next((it.get("time") for it in upgrade_info["教室_B"]
                  if it["level"] == levels.get(f"教室_B{i}", 1)), None)
        if t is not None:
            times.append(t)
    return int(max(times)) if times else 60

def get_coin_rate(levels: Dict[str, int], risk: float) -> float:
    gym_lv = levels.get("計測_A", 1)
    hist = get_classroom_hist(levels)
    reward = expected_cycle_reward_compressed(gym_lv, hist)
    cycle_time = compute_cycle_time(levels)
    return reward * risk / cycle_time if cycle_time > 0 else 0.0

def total_level(levels: Dict[str, int]) -> int:
    keys = (
        ["受付_A", "受付_B", "計測_A", "計測_B"] +
        [f"教室_A{i}" for i in range(1, 6)] +
        [f"教室_B{i}" for i in range(1, 6)]
    )
    return sum(levels.get(k, 1) for k in keys)

# === 報酬分布計算 ===

def reward_distribution(
    gym_level: int,
    class_hist: Tuple[int, ...]
) -> Dict[int, float]:
    (gmin, gmax), p_double = gym_A_levels[gym_level]
    gym_pmf = {x: 1.0/(gmax-gmin+1) for x in range(gmin, gmax+1)}
    class_pmf = combine_classroom_distribution(class_hist)
    total_pmf = convolve_pmfs(gym_pmf, class_pmf)
    dist: Dict[int, float] = defaultdict(float)
    for pts, prob in total_pmf.items():
        coin = next((c for low, high, c in reward_table if low <= pts <= high), 0)
        dist[coin] += prob
    return dict(sorted(dist.items()))

# === 評価関数 ===

def evaluate(params: EvaluationParams) -> EvaluationResult:
    lv = params.levels
    rf = params.risk_factor
    tot_lv = total_level(lv)
    cy_rew = expected_cycle_reward_compressed(lv.get("計測_A", 1), get_classroom_hist(lv))
    cy_time = compute_cycle_time(lv)
    coin_rt = cy_rew * rf / cy_time if cy_time > 0 else 0.0
    hourly_rt = coin_rt * 3600
    return EvaluationResult(
        total_level=tot_lv,
        cycle_reward=cy_rew,
        cycle_time=cy_time,
        coin_rate=coin_rt,
        hourly_rate=hourly_rt,
    )

# === レベルアップ試算用関数 ===

def get_upgrade_cost(part: str, current_level: int) -> Optional[int]:
    if part.startswith("教室_A"):
        key = "教室_A"
    elif part.startswith("教室_B"):
        key = "教室_B"
    else:
        key = part
    info = upgrade_info.get(key, [])
    next_level = current_level + 1
    return next((it.get("cost") for it in info if it.get("level") == next_level), None)

def accumulate_minutes_ceiled(current_coins: int, needed_coins: int, coin_rate_per_sec: float) -> Optional[int]:
    deficit = needed_coins - current_coins
    if deficit <= 0:
        return 0
    if coin_rate_per_sec is None or coin_rate_per_sec <= 0:
        return None
    wait_seconds = deficit / coin_rate_per_sec
    return math.ceil(wait_seconds / 60)

def arrival_time_from_minutes(minutes_ceiled: int) -> datetime:
    return datetime.now() + timedelta(minutes=minutes_ceiled)

# === Streamlit UI ===

def main():
    st.title("玉龍幣シミュ")

    # リスク調整係数
    risk = st.sidebar.number_input(
        "リスク調整係数 (-r)", min_value=0.0, max_value=2.0, value=1.0, step=0.01
    )

    # 表示用行・列ラベル
    row_labels  = ["教師", "席"]
    row_codes   = ["a",   "b"]
    col_labels  = ["受付","腕前審査","料理","包丁","製菓","調理","盛付"]
    col_nums    = list(range(1, 8))

    st.markdown("### レベル入力（ここで次に上げる部品と所持玉龍幣も指定）")
    with st.form("level_form"):
        # 列ヘッダー
        header_cols = st.columns(7)
        for col, lbl in zip(header_cols, col_labels):
            col.markdown(f"**{lbl}**")

        level_inputs: Dict[str, int] = {}
        # 各行を別々に columns で作る（重なりを避けるため）
        for rcode, rlabel in zip(row_codes, row_labels):
            st.write(f"**{rlabel}**")
            row_cols = st.columns(7)
            for col, num in zip(row_cols, col_nums):
                code = f"{num}{rcode}"
                part = code_to_part.get(code)
                if part is None:
                    part = "教室_A1"
                if part.startswith("教室_A"):
                    key = "教室_A"
                elif part.startswith("教室_B"):
                    key = "教室_B"
                else:
                    key = part
                max_lv = max(it.get("level", 1) for it in upgrade_info[key])
                lvl = col.selectbox(
                    "",
                    options=list(range(1, max_lv+1)),
                    index=0,
                    key=code,
                    label_visibility="collapsed"
                )
                level_inputs[code] = lvl

        # ここで「次に上げたい部品」と「現在の玉龍幣残高」をフォーム内で入力できるようにする
        # 表示順を安定させるためコード順で部品名リストを作成
        part_options = [code_to_part[c] for c in sorted(level_inputs.keys())]
        part_choice = st.selectbox("次に上げたい部品", options=part_options, index=0, key="part_to_upgrade")
        current_coins = st.number_input("現在の玉龍幣残高", min_value=0, value=0, step=100, key="current_coins")

        submitted = st.form_submit_button("評価実行")

    if not submitted:
        return

    # 入力確認：2×7 テーブル表示
    data = [
        [level_inputs.get(f"{num}{rcode}", 1) for num in col_nums]
        for rcode in row_codes
    ]
    df_input = pd.DataFrame(data, index=row_labels, columns=col_labels)
    st.markdown("#### 入力内容")
    st.table(df_input)

    # 評価実行
    # lvl_dict のキーは部品名（例: "教室_A1"）になる
    lvl_dict = {code_to_part[c]: level_inputs[c] for c in level_inputs}
    result = evaluate(EvaluationParams(levels=lvl_dict, risk_factor=risk))

    st.markdown("## 評価結果")
    st.write(f"- 合計レベル: {result.total_level}")
    st.write(f"- 料理人あたり玉龍幣期待値: {result.cycle_reward:.2f}")
    st.write(f"- サイクルタイム: {result.cycle_time} 秒")
    st.write(f"- 1時間あたり玉龍幣期待値: {result.hourly_rate:.2f}")

    # 報酬分布
    hist = get_classroom_hist(lvl_dict)
    dist = reward_distribution(lvl_dict.get("計測_A", 1), hist)
    names = [reward_names.get(c, str(c)) for c in dist.keys()]
    probs = [round(p*100, 2) for p in dist.values()]
    df_dist = pd.DataFrame({"確率(%)": probs}, index=names)
    st.markdown("## 料理人分布")
    st.table(df_dist)

    # 次のレベルアップ試算（フォーム内で選択した値を使う）
    st.markdown("## 次のレベルアップ試算（フォームで指定済み）")
    part_to_upgrade = part_choice
    current_coins = int(current_coins)

    cur_level = lvl_dict.get(part_to_upgrade, 1)
    cost = get_upgrade_cost(part_to_upgrade, cur_level)

    if cost is None:
        st.info("この部品はこれ以上レベルアップできません。")
    else:
        st.write(f"- 次のレベルアップ必要玉龍幣: {cost}")
        minutes_ceiled = accumulate_minutes_ceiled(current_coins, cost, result.coin_rate)
        if minutes_ceiled is None:
            st.warning("現在の獲得速度では到達予定時刻を計算できません（獲得速度が 0 の可能性）。")
        else:
            arrival_dt = arrival_time_from_minutes(minutes_ceiled)
            remaining_hours = minutes_ceiled // 60
            remaining_minutes = minutes_ceiled % 60
            st.write(f"- 到達予定時刻: {arrival_dt.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"- 残り時間: {remaining_hours}時間{remaining_minutes}分")

            # レベルアップ後の期待値（仮想的に +1 して再評価）
            new_levels = lvl_dict.copy()
            new_levels[part_to_upgrade] = cur_level + 1
            new_result = evaluate(EvaluationParams(levels=new_levels, risk_factor=risk))
            st.write(f"- レベルアップ後の1時間あたり玉龍幣期待値: {new_result.hourly_rate:.2f}")

            diff = new_result.hourly_rate - result.hourly_rate
            st.write(f"- 期待値の増加量（1時間あたり）: {diff:+.2f}")

if __name__ == "__main__":
    main()
