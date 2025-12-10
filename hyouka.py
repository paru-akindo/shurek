# main.py
"""
水準評価モード専用アプリケーション
- 部品レベル入力を 2行×7列 のプルダウンで行う
- 各サイクルで得られる reward_table の報酬名別確率分布を表で表示
- 次に上げたい部品と現在の玉龍幣残高を入力すると、
  到達予定時刻と残り時間、レベルアップ後の期待値を表示
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import math

# === 定数・テーブル定義 ===
# （ここは元のまま省略せずに全部入れてください）
# upgrade_info, gym_A_levels, class_A_levels, reward_table, reward_names, code_to_part
# …省略…

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
# （元のまま）

# === 期待値・レート計算 ===
# （元のまま）

# === 報酬分布計算 ===
# （元のまま）

# === 評価関数 ===
# （元のまま）

# === レベルアップ試算用関数 ===
def get_upgrade_cost(part: str, current_level: int) -> int | None:
    key = "教室_A" if part.startswith("教室_A") else "教室_B" if part.startswith("教室_B") else part
    info = upgrade_info.get(key, [])
    next_level = current_level + 1
    return next((it["cost"] for it in info if it.get("level") == next_level), None)

def accumulate_minutes_ceiled(current_coins: int, needed_coins: int, coin_rate_per_sec: float) -> int | None:
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

    st.markdown("### レベル入力")
    with st.form("level_form"):
        cols = st.columns(7)
        for col, lbl in zip(cols, col_labels):
            col.markdown(f"**{lbl}**")

        level_inputs: Dict[str, int] = {}
        for rcode in row_codes:
            for col, num in zip(cols, col_nums):
                code = f"{num}{rcode}"
                part = code_to_part[code]
                if part.startswith("教室_A"):
                    key = "教室_A"
                elif part.startswith("教室_B"):
                    key = "教室_B"
                else:
                    key = part
                max_lv = max(it["level"] for it in upgrade_info[key])
                lvl = col.selectbox(
                    "",
                    options=list(range(1, max_lv+1)),
                    index=0,
                    key=code,
                    label_visibility="collapsed"
                )
                level_inputs[code] = lvl

        submitted = st.form_submit_button("評価実行")

    if not submitted:
        return

    # 入力確認
    data = [[level_inputs[f"{num}{rcode}"] for num in col_nums] for rcode in row_codes]
    df_input = pd.DataFrame(data, index=row_labels, columns=col_labels)
    st.markdown("#### 入力内容")
    st.table(df_input)

    # 評価実行
    lvl_dict = {code_to_part[c]: level_inputs[c] for c in level_inputs}
    result = evaluate(EvaluationParams(levels=lvl_dict, risk_factor=risk))

    st.markdown("## 評価結果")
    st.write(f"- 合計レベル: {result.total_level}")
    st.write(f"- 料理人あたり玉龍幣期待値: {result.cycle_reward:.2f}")
    st.write(f"- サイクルタイム: {result.cycle_time} 秒")
    st.write(f"- 1時間あたり玉龍幣期待値: {result.hourly_rate:.2f}")

    # 報酬分布
    hist = get_classroom_hist(lvl_dict)
    dist = reward_distribution(lvl_dict["計測_A"], hist)
    names = [reward_names.get(c, str(c)) for c in dist.keys()]
    probs = [round(p*100, 2) for p in dist.values()]
    df_dist = pd.DataFrame({"確率(%)": probs}, index=names)
    st.markdown("## 料理人分布")
    st.table(df_dist)

    # 次のレベルアップ試算
    st.markdown("## 次のレベルアップ試算")
    part_to_upgrade = st.selectbox("次に上げたい部品", list(lvl_dict.keys()))
    current_coins = st.number_input("現在の玉龍幣残高", min_value=0, value=0)

    cur_level = lvl_dict[part_to_upgrade]
    cost = get_upgrade_cost(part_to_upgrade, cur_level)

    if cost is None:
        st.info("この部品はこれ以上レベルアップできません。")
    else:
        st.write(f"- 次のレベルアップ必要玉龍幣: {cost}")
        minutes_ceiled = accumulate_minutes_ceiled(current_coins, cost, result.coin_rate)
        if minutes_ceiled is None:
            st.warning("現在の獲得速度では到達予定時刻を計算できません。")
        else:
            arrival_dt = arrival_time_from_minutes(minutes_ceiled)
            remaining_hours = minutes_ceiled // 60
            remaining_minutes = minutes_ceiled % 60
            st.write(f"- 到達予定時刻: {arrival_dt.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"- 残り時間: {remaining_hours}時間{remaining_minutes}分")

            new_levels = lvl_dict.copy()
            new_levels[part_to_upgrade] = cur_level + 1
            new_result = evaluate(EvaluationParams(levels=new_levels, risk_factor=risk))
            st.write(f"- レベルアップ後の1時間あたり玉龍幣期待値: {new_result.hourly_rate:.2f}")
            diff = new_result.hourly_rate - result.hourly_rate
            st.write(f"- 期待値の増加量（1時間あたり）: {diff:+.2f}")

if __name__ == "__main__":
    main()
