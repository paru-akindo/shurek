# main.py （UI 部分のみ抜粋・差分イメージ）

def main():
    st.title("水準評価モード（表形式入力＋報酬分布）")

    risk = st.sidebar.number_input(
        "リスク調整係数 (-r)", min_value=0.0, max_value=2.0, value=1.0, step=0.01
    )

    # 「教師」「席」を行ラベルとし、
    # 受付→腕前審査→料理→包丁→製菓→調理→盛付 を列ラベルに使う
    row_labels  = ["教師", "席"]
    row_codes   = ["a",   "b"]      # 内部処理用の接尾コード
    col_labels  = ["受付","腕前審査","料理","包丁","製菓","調理","盛付"]
    col_numbers = list(range(1, 8))

    st.markdown("### 部品レベル入力 (2行×7列)")
    with st.form("level_form"):
        cols = st.columns(7)
        # 列ヘッダーを新しいラベルで描画
        for col, lbl in zip(cols, col_labels):
            col.markdown(f"**{lbl}**")

        level_inputs: Dict[str,int] = {}
        # 各セルでプルダウン
        for rcode, rlabel in zip(row_codes, row_labels):
            for col, num in zip(cols, col_numbers):
                code = f"{num}{rcode}"                # 内部コード例: "1a","2b"...
                part = code_to_part[code]             # 元のマッピングをそのまま使う
                # 教室系だけテーブルキーを使い分け
                if part.startswith("教室_A"):  key = "教室_A"
                elif part.startswith("教室_B"):key = "教室_B"
                else:                          key = part
                max_lv = max(item["level"] for item in upgrade_info[key])
                lvl = col.selectbox(
                    label=code, 
                    options=list(range(1, max_lv+1)),
                    index=0,
                    key=code
                )
                level_inputs[code] = lvl

        submitted = st.form_submit_button("評価実行")

    if not submitted:
        return

    # ─── 入力確認テーブル ───────────────────────────
    data = [
        [level_inputs[f"{num}{rcode}"] for num in col_numbers]
        for rcode in row_codes
    ]
    df_input = pd.DataFrame(data, index=row_labels, columns=col_labels)
    st.markdown("#### 入力レベル")
    st.table(df_input)

    # ─── 以降は既存の evaluate 呼び出し以降の処理 ─────
    lvl_dict = { code_to_part[c]: level_inputs[c] for c in level_inputs }
    result   = evaluate(EvaluationParams(levels=lvl_dict, risk_factor=risk))

    st.markdown("## 評価結果")
    st.write(f"- 合計レベル: {result.total_level}")
    st.write(f"- 1サイクルあたり金貨期待値: {result.cycle_reward:.2f}")
    st.write(f"- サイクルタイム: {result.cycle_time} 秒")
    st.write(f"- 1時間あたり生成期待値: {result.hourly_rate:.2f}")

    # 報酬分布テーブル（報酬名をインデックスに）
    hist = get_classroom_hist(lvl_dict)
    dist = reward_distribution(lvl_dict["計測_A"], hist)
    names = [reward_names.get(c,c) for c in dist.keys()]
    probs = [round(p*100,2) for p in dist.values()]
    df_dist = pd.DataFrame({"確率(%)": probs}, index=names)
    st.markdown("## 報酬テーブル分布")
    st.table(df_dist)
