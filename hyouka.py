# main.py
"""
金貨生成シミュレーション／水準評価モード
このファイルひとつで、Streamlit UI と評価ロジックを内包しています。
"""

import streamlit as st
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, List

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
        {"level": 1, "time": 60, "cost": 0},
        {"level": 2, "time": 30, "cost": 6900},
        {"level": 3, "time": 20, "cost": 11500},
        {"level": 4, "time": 15, "cost": 23000},
        {"level": 5, "time": 12, "cost": 31850},
        {"level": 6, "time": 10, "cost": 58800},
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

# === 評価ロジック（純粋関数群） ===

def get_reward(total_points: int) -> int:
    for low, high, coin in reward_table:
        if low <= total_points <= high:
            return coin
    return 0

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

_expected_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}

def get_classroom_hist(levels: Dict[str, int]) -> Tuple[int, ...]:
    counts = {i: 0 for i in range(1, 7)}
    for i in range(1, 6):
        lvl = levels.get(f"教室_A{i}", 1)
        counts[lvl] += 1
    return tuple(counts[i] for i in range(1, 7))

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
    gym_pmf = {x: 1.0 / (gmax - gmin + 1) for x in range(gmin, gmax + 1)}
    class_pmf = combine_classroom_distribution(class_hist)
    total = convolve_pmfs(gym_pmf, class_pmf)

    exp_val = 0.0
    for pts, prob in total.items():
        exp_val += get_reward(pts) * (1 + p_double) * prob

    _expected_cache[key] = exp_val
    return exp_val

def get_part_time(part: str, level: int):
    lst = upgrade_info.get(part, [])
    for item in lst:
        if item["level"] == level:
            return item.get("time")
    return None

def compute_cycle_time(levels: Dict[str, int]) -> int:
    times = []
    for part in ["受付_A", "受付_B", "計測_B"]:
        t = get_part_time(part, levels.get(part, 1))
        if t is not None:
            times.append(t)
    for i in range(1, 6):
        t = get_part_time("教室_B", levels.get(f"教室_B{i}", 1))
        if t is not None:
            times.append(t)
    return max(times) if times else 60

def get_coin_rate(levels: Dict[str, int], risk: float) -> float:
    gym_lv = levels["計測_A"]
    hist = get_classroom_hist(levels)
    reward = expected_cycle_reward_compressed(gym_lv, hist)
    cycle_time = compute_cycle_time(levels)
    return reward * risk / cycle_time

def total_level(levels: Dict[str, int]) -> int:
    keys = (
        ["受付_A", "受付_B", "計測_A", "計測_B"] +
        [f"教室_A{i}" for i in range(1, 6)] +
        [f"教室_B{i}" for i in range(1, 6)]
    )
    return sum(levels.get(k, 1) for k in keys)

def evaluate(params: EvaluationParams) -> EvaluationResult:
    lv = params.levels
    rf = params.risk_factor

    tot_lv = total_level(lv)
    cy_rew = expected_cycle_reward_compressed(lv["計測_A"], get_classroom_hist(lv))
    cy_time = compute_cycle_time(lv)
    coin_rt = get_coin_rate(lv, rf)
    hourly_rt = coin_rt * 3600

    return EvaluationResult(
        total_level=tot_lv,
        cycle_reward=cy_rew,
        cycle_time=cy_time,
        coin_rate=coin_rt,
        hourly_rate=hourly_rt
    )

# === 入力パースヘルパー ===

def parse_levels_input(text: str) -> Dict[str, int]:
    parts_order = [
        "受付_A", "受付_B", "計測_A", "計測_B",
        "教室_A1", "教室_A2", "教室_A3", "教室_A4", "教室_A5",
        "教室_B1", "教室_B2", "教室_B3", "教室_B4", "教室_B5",
    ]
    items = [x.strip() for x in text.split(",")]
    if len(items) != len(parts_order):
        raise ValueError(f"14個の数値が必要です (現在: {len(items)})")
    levels = {}
    for key, val in zip(parts_order, items):
        if not val.isdigit():
            raise ValueError(f"数値変換エラー: {val}")
        levels[key] = int(val)
    return levels

# === Streamlit UI ===

def main():
    st.title("金貨生成シミュレーション／水準評価モード")

    risk = st.sidebar.number_input(
        "リスク調整係数 (-r) (例: 0.95)", min_value=0.0, value=1.0, step=0.01
    )

    st.markdown(
        "各部品レベルをカンマ区切りで入力してください\n"
        "順序：1a,1b,2a,2b,3a,4a,5a,6a,7a,3b,4b,5b,6b,7b"
    )
    user_input = st.text_input(
        "部品レベル (例: 1,1,2,2,3,3,3,3,3,1,1,1,1,1)"
    )

    if st.button("評価実行"):
        try:
            lv_dict = parse_levels_input(user_input)
        except ValueError as err:
            st.error(f"入力エラー: {err}")
            st.stop()

        st.write("指定されたレベル構成:", lv_dict)

        result = evaluate(EvaluationParams(levels=lv_dict, risk_factor=risk))

        st.markdown("**評価結果**")
        st.write(f"合計レベル: {result.total_level}")
        st.write(f"1サイクルあたり金貨期待値: {result.cycle_reward:.2f}")
        st.write(f"サイクルタイム: {result.cycle_time}秒")
        st.write(f"1時間あたり生成期待値: {result.hourly_rate:.2f}")

if __name__ == "__main__":
    main()
