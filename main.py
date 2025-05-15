import streamlit as st
import pandas as pd
import math
from collections import defaultdict
from io import StringIO

# ------------- ヘルパー関数の定義（シミュレーション処理で使用する前に定義） ----------------

def get_local_time(t):
    local_minutes = t % 1440
    hour = local_minutes // 60
    minute = local_minutes % 60
    return hour, minute

def format_time(t):
    day = t // 1440 + 1
    hour, minute = get_local_time(t)
    return f"Day {day} {hour:02d}:{minute:02d}"


# ------------- 以下、元の関数定義や定数など ----------------

# ===================== アップグレードテーブル =====================
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
    "計測_B": [
        {"level": 1, "time": 30, "cost": 0},
        {"level": 2, "time": 15, "cost": 17250},
        {"level": 3, "time": 7.5, "cost": 44650},
    ],
    "教室_B": [
        {"level": 1, "time": 60, "cost": 0},
        {"level": 2, "time": 30, "cost": 6900},
        {"level": 3, "time": 20, "cost": 11500},
        {"level": 4, "time": 15, "cost": 23000},
        {"level": 5, "time": 12, "cost": 31850},
        {"level": 6, "time": 10, "cost": 58800},
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
    "教室_A": [
        {"level": 1, "cost": 0},
        {"level": 2, "cost": 1000},
        {"level": 3, "cost": 8250},
        {"level": 4, "cost": 14700},
        {"level": 5, "cost": 19900},
        {"level": 6, "cost": 27600},
    ]
}

# =================== 弟子取得ポイント・両手攻撃設定 ===================
gym_A_levels = {
    1: ((2, 8), 0.15),
    2: ((7, 13), 0.20),
    3: ((12, 18), 0.25),
    4: ((15, 25), 0.30),
    5: ((22, 38), 0.35),
    6: ((37, 53), 0.40),
    7: ((52, 68), 0.45),
    8: ((65, 85), 0.50)
}
# 教室_A 各レベルにおける取得レンジ
class_A_levels = {
    1: (7, 13),
    2: (10, 20),
    3: (17, 33),
    4: (30, 50),
    5: (47, 74),
    6: (70, 100),
    7: (97, 133)
}

# ==================== 弟子レベルと報酬テーブル ====================
reward_table = [
    (0, 79, 2),
    (80, 139, 5),
    (140, 219, 10),
    (220, 339, 15),
    (340, 459, 20),
    (460, 479, 25),
    (480, 619, 35),
    (620, 819, 40),
    (820, 1039, 50)
]

def get_reward(total_points):
    for low, high, coin in reward_table:
        if low <= total_points <= high:
            return coin
    return 0

def pmf_uniform(a, b, n):
    if n == 0:
        return {0: 1.0}
    base = 1.0 / (b - a + 1)
    pmf = {x: base for x in range(a, b+1)}
    for _ in range(1, n):
        pmf = convolve_pmfs(pmf, {x: base for x in range(a, b+1)})
    return pmf

def convolve_pmfs(pmf1, pmf2):
    result = defaultdict(float)
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            result[v1+v2] += p1 * p2
    return dict(result)

def merge_pmfs(pmf_list):
    result = {0: 1.0}
    for pmf in pmf_list:
        result = convolve_pmfs(result, pmf)
    return result

expected_reward_cache = {}

def next_upgrade_cost(part, current_level):
    if part.startswith("教室_A"):
        key = "教室_A"
    elif part.startswith("教室_B"):
        key = "教室_B"
    else:
        key = part
    info_list = upgrade_info.get(key)
    if info_list is None:
        return None
    for i, item in enumerate(info_list):
        if item["level"] == current_level:
            if i+1 < len(info_list):
                return info_list[i+1]["cost"]
            else:
                return None
    return None

def get_classroom_hist(levels):
    counts = {i: 0 for i in range(1, 7)}
    for i in range(1, 6):
        key = f"教室_A{i}"
        lvl = levels.get(key, 1)
        counts[lvl] += 1
    return tuple(counts[i] for i in range(1, 7))

def combine_classroom_distribution(hist):
    pmf_list = []
    for level in range(1, 7):
        count = hist[level-1]
        if count > 0:
            pmf_list.append(pmf_uniform(*class_A_levels[level], count))
    if not pmf_list:
        return {0: 1.0}
    return merge_pmfs(pmf_list)

def expected_cycle_reward_compressed(gym_level, class_hist):
    key = (gym_level, class_hist)
    if key in expected_reward_cache:
        return expected_reward_cache[key]
    (gym_min, gym_max), p_double = gym_A_levels[gym_level]
    n_gym = gym_max - gym_min + 1
    gym_pmf = {x: 1.0/n_gym for x in range(gym_min, gym_max+1)}
    classroom_pmf = combine_classroom_distribution(class_hist)
    overall_pmf = convolve_pmfs(gym_pmf, classroom_pmf)
    expected = 0.0
    for total_points, prob in overall_pmf.items():
        R = get_reward(total_points)
        effective_R = R * (1 + p_double)
        expected += effective_R * prob
    expected_reward_cache[key] = expected
    return expected

def get_part_time(part, level):
    for item in upgrade_info[part]:
        if item["level"] == level:
            return item.get("time")
    return None

def compute_cycle_time(levels):
    parts = ["受付_A", "受付_B", "計測_B"]
    times = []
    for part in parts:
        t = get_part_time(part, levels.get(part,1))
        if t is not None:
            times.append(t)
    for i in range(1,6):
        key = f"教室_B{i}"
        t = get_part_time("教室_B", levels.get(key,1))
        if t is not None:
            times.append(t)
    return max(times) if times else 60

def get_coin_rate(levels, risk_factor=1.0):
    gym_level = levels["計測_A"]
    class_hist = get_classroom_hist(levels)
    reward_cycle = expected_cycle_reward_compressed(gym_level, class_hist)
    cycle_time = compute_cycle_time(levels)
    adjusted_reward = reward_cycle * risk_factor
    return adjusted_reward / cycle_time

def total_level(levels):
    tot = levels["受付_A"] + levels["受付_B"] + levels["計測_A"] + levels["計測_B"]
    for i in range(1,6):
        tot += levels.get(f"教室_A{i}", 1)
        tot += levels.get(f"教室_B{i}", 1)
    return tot

# ===================== 固定パラメータ =====================
START_TIME = 600    
END_TIME = 4200     
bonus_events = {
    1440: 154200,
    2040: 5300,
    2880: 154200,
    3480: 5300
}
INITIAL_COINS = 159500

code_to_part = {
    "1a": "受付_A",
    "1b": "受付_B",
    "2a": "計測_A",
    "2b": "計測_B",
    "3a": "教室_A1",
    "4a": "教室_A2",
    "5a": "教室_A3",
    "6a": "教室_A4",
    "7a": "教室_A5",
    "3b": "教室_B1",
    "4b": "教室_B2",
    "5b": "教室_B3",
    "6b": "教室_B4",
    "7b": "教室_B5",
}

# ===================== Streamlit UI ----------------====
st.title("金貨生成シミュレーション／水準評価アプリ")

st.sidebar.header("モード選択")
mode = st.sidebar.radio("モードを選択", ("シミュレーションモード", "水準評価モード"))
risk_factor_input = st.sidebar.number_input("リスク調整係数 (-r) (例: 0.95 で期待値5%%下げる)", 
                                            min_value=0.0, value=1.0, step=0.01)

if mode == "シミュレーションモード":
    st.header("シミュレーションモード")
    csv_text = st.text_area("CSV 指示を入力（ヘッダー: part,target）", height=200,
                            value="part,target\n1a,4\n3b,2\n4b,2\n5b,2")
    start_forbid = st.sidebar.number_input("作業禁止開始時刻 (-s) (時)", min_value=0, max_value=23, value=1, step=1)
    end_forbid = st.sidebar.number_input("作業禁止終了時刻 (-e) (時)", min_value=0, max_value=23, value=7, step=1)
    late = st.sidebar.number_input("開始遅延 (-l) (分)", min_value=0, value=0, step=1)
    
    if csv_text.strip() == "":
        st.error("CSV 指示を入力してください。")
        st.stop()
    
    try:
        csv_io = StringIO(csv_text)
        df = pd.read_csv(csv_io)
    except Exception as e:
        st.error(f"CSV 読み込みエラー: {e}")
        st.stop()
    
    upgrade_schedule = []
    for i, row in df.iterrows():
        code = str(row["part"]).strip()
        if code not in code_to_part:
            st.error(f"不明な部品コード: {code}")
            st.stop()
        part = code_to_part[code]
        try:
            target = int(row["target"])
        except Exception as e:
            st.error(f"ターゲットレベル変換エラー: {e}")
            st.stop()
        upgrade_schedule.append((part, target))
    
    st.write("アップグレード指示:")
    st.write(upgrade_schedule)
    
    if st.button("シミュレーション実行"):
        current_time = START_TIME + int(late)
        coin_balance = INITIAL_COINS
        levels = {
            "受付_A": 1,
            "受付_B": 1,
            "計測_A": 1,
            "計測_B": 1,
            "教室_A1": 1, "教室_A2": 1, "教室_A3": 1, "教室_A4": 1, "教室_A5": 1,
            "教室_B1": 1, "教室_B2": 1, "教室_B3": 1, "教室_B4": 1, "教室_B5": 1,
        }
        upgrade_log = []
        schedule_index = 0
        sim_output = []
        while current_time <= END_TIME and total_level(levels) < 64:
            if current_time in bonus_events:
                coin_balance += bonus_events[current_time]
            rate = get_coin_rate(levels, risk_factor_input)
            coin_balance += rate * 60  
            local_hour = (current_time % 1440) // 60
            if not (start_forbid <= local_hour < end_forbid):
                upgrades_this_minute = []
                while schedule_index < len(upgrade_schedule):
                    part, target_level = upgrade_schedule[schedule_index]
                    if levels.get(part, 0) >= target_level:
                        schedule_index += 1
                        continue
                    cost = next_upgrade_cost(part, levels[part])
                    if cost is None:
                        schedule_index += 1
                        continue
                    if coin_balance >= cost:
                        coin_balance -= cost
                        levels[part] += 1
                        upgrades_this_minute.append((part, cost, levels[part]))
                        if levels[part] >= target_level:
                            schedule_index += 1
                    else:
                        break
                if upgrades_this_minute:
                    upgrade_log.append((current_time, upgrades_this_minute))
                    current_reward = expected_cycle_reward_compressed(levels["計測_A"], get_classroom_hist(levels))
                    current_cycle_time = compute_cycle_time(levels)
                    current_coin_rate = get_coin_rate(levels, risk_factor_input)
                    hourly_rate = current_coin_rate * 3600
                    sim_output.append(f"{format_time(current_time)} （アップグレード後） 金貨期待値: {current_reward:.2f} (1サイクル), サイクルタイム: {current_cycle_time}秒, 1時間あたり生成期待値: {hourly_rate:.2f}")
            if total_level(levels) >= 64:
                break
            current_time += 1
        if total_level(levels) >= 64:
            st.success(f"目標達成！到達時刻: {format_time(current_time)}")
        else:
            st.warning("終了時刻までに全場所の合計レベル64に到達できませんでした。")
        st.write("アップグレード実績:")
        for t, events in upgrade_log:
            ev_str = ", ".join([f"{p}→Lv{lvl} (cost {cost})" for p, cost, lvl in events])
            st.write(f"{format_time(t)}: {ev_str}")
        st.write("最終状態:", levels)
        st.write("合計レベル:", total_level(levels))
        st.write("最終残高:", math.floor(coin_balance))
        final_reward = expected_cycle_reward_compressed(levels["計測_A"], get_classroom_hist(levels))
        final_cycle_time = compute_cycle_time(levels)
        final_coin_rate = get_coin_rate(levels, risk_factor_input)
        hourly_final = final_coin_rate * 3600
        st.markdown(f"**【最終設定の詳細】** 1サイクルあたり金貨期待値: {final_reward:.2f}, サイクルタイム: {final_cycle_time}秒, 1時間あたり生成期待値: {hourly_final:.2f}")
        
elif mode == "水準評価モード":
    st.header("水準評価モード")
    st.markdown("各部品のレベルをカンマ区切りで入力してください（順序：1a,1b,2a,2b,3a,4a,5a,6a,7a,3b,4b,5b,6b,7b）。")
    levels_input = st.text_input("部品レベル (例: 1,1,2,2,3,3,3,3,3,1,1,1,1,1)")
    if st.button("評価実行"):
        try:
            level_strs = levels_input.split(',')
            if len(level_strs) != 14:
                st.error("部品レベルは14個の値をカンマ区切りで指定してください。")
                st.stop()
            level_values = [int(x.strip()) for x in level_strs]
        except Exception as e:
            st.error(f"レベル値変換エラー: {e}")
            st.stop()
        level_keys_order = ["受付_A", "受付_B", "計測_A", "計測_B",
                            "教室_A1", "教室_A2", "教室_A3", "教室_A4", "教室_A5",
                            "教室_B1", "教室_B2", "教室_B3", "教室_B4", "教室_B5"]
        levels_dict = dict(zip(level_keys_order, level_values))
        st.write("指定されたレベル構成:")
        for key in level_keys_order:
            st.write(f"{key}: Lv {levels_dict[key]}")
        st.write("合計レベル:", total_level(levels_dict))
        gym_level_val = levels_dict["計測_A"]
        classroom_hist_val = get_classroom_hist(levels_dict)
        cycle_reward_val = expected_cycle_reward_compressed(gym_level_val, classroom_hist_val)
        cycle_time_val = compute_cycle_time(levels_dict)
        coin_rate_val = get_coin_rate(levels_dict, risk_factor_input)
        hourly_rate_val = coin_rate_val * 3600
        st.markdown("**【指定された構成の詳細】**")
        st.write(f"1サイクルあたりの金貨期待値: {cycle_reward_val:.2f}")
        st.write(f"サイクルタイム: {cycle_time_val}秒")
        st.write(f"1時間あたり生成期待値: {hourly_rate_val:.2f}")
