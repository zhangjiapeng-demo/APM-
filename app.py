import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from difflib import get_close_matches
import numpy as np

# 1. 页面配置
st.set_page_config(page_title="APM数据可视化 Pro", layout="wide", page_icon="📊")

# --- 🎨 样式优化 ---
st.markdown("""
    <style>
        .block-container { text-align: center; padding-top: 2rem; }
        h1, h2, h3, h4, h5, h6 { text-align: center !important; width: 100%; }
        div[data-testid="stDataFrame"] { display: inline-block; text-align: left; margin: 0 auto; }
        div.stDownloadButton { text-align: center; }
        .metric-card {
            background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin: 10px 0; text-align: left;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 APM 性能分析 (完整修复版)")

# --- 翻译字典 ---
TRANS_MAP = {
    "cpu_app": "应用CPU", "cpu_sys": "系统CPU", "mem_total": "总内存", "mem_swap": "交换内存",
    "battery_level": "电量", "battery_tem": "电池温度", "fps": "帧率FPS", "gpu": "GPU",
    "timestamp": "采集时间", "time": "时间", "value": "数值",
    "upflow": "上行流量", "downflow": "下行流量", "net_usage": "网络使用率",
    "mem_rss": "物理内存(RSS)", "mem_vss": "虚拟内存(VSS)", "heap_size": "堆内存", "heap_alloc": "已用堆内存"
}


def get_smart_name(name):
    name_str = str(name).strip()
    if name_str in TRANS_MAP: return TRANS_MAP[name_str]
    if name_str.lower() in TRANS_MAP: return TRANS_MAP[name_str.lower()]
    matches = get_close_matches(name_str.lower(), TRANS_MAP.keys(), n=1, cutoff=0.8)
    if matches: return TRANS_MAP[matches[0]]
    return name_str


# --- 缓存读取 ---
def get_file_info(file):
    if file.name.endswith('.csv'): return None, ["CSV数据"]
    xls = pd.ExcelFile(file)
    return xls, xls.sheet_names


@st.cache_data(ttl=3600)
def load_data_from_sheet(file, sheet_name, is_csv):
    try:
        if is_csv:
            file.seek(0);
            return pd.read_csv(file)
        return pd.read_excel(file, sheet_name=sheet_name)
    except:
        return pd.DataFrame()


# --- 🧠 内存诊断逻辑 ---
def diagnose_memory(df, mem_col, total_mem_limit=None):
    result = {"status": "normal", "messages": [], "slope": 0.0, "is_oom_risk": False}
    series = df[mem_col].dropna()
    if len(series) < 10: return result, (0, 0)

    y = series.values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    result["slope"] = slope

    start_val = np.mean(y[:10])
    end_val = np.mean(y[-10:])
    growth_rate = (end_val - start_val) / (start_val + 1e-5)

    if slope > 0.05 and growth_rate > 0.1:
        result["status"] = "warning"
        result["messages"].append(f"📉 **泄漏风险**: 内存呈上升趋势 (增长率 {growth_rate:.1%})")
        if slope > 0.5:
            result["status"] = "critical"
            result["messages"].append("🚫 **严重泄漏**: 增长极快，请检查代码！")
    else:
        result["messages"].append("✅ **趋势正常**: 未检测到持续泄漏")

    max_val = np.max(y)
    if total_mem_limit and total_mem_limit > 0:
        usage_ratio = max_val / total_mem_limit
        if usage_ratio > 0.95:
            result["is_oom_risk"] = True
            result["messages"].append(f"🔥 **OOM 警告**: 峰值已达上限的 {usage_ratio:.1%}")

    return result, (slope, intercept)


# --- 主程序 ---
uploaded_file = st.file_uploader("📂 上传测试数据 (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    try:
        xls_obj, sheet_names = get_file_info(uploaded_file)

        with st.sidebar:
            st.header("⚙️ 控制面板")

            # 1. 数据源
            sheet_alias = [get_smart_name(s) for s in sheet_names]
            selected_alias = st.selectbox("数据项:", sheet_alias)
            selected_sheet_raw = sheet_names[sheet_alias.index(selected_alias)]

            is_csv = uploaded_file.name.endswith('.csv')
            df_raw = load_data_from_sheet(uploaded_file, selected_sheet_raw, is_csv)

            if df_raw.empty: st.error("数据为空"); st.stop()

            # 2. 轴设置
            columns = df_raw.columns.tolist()
            default_x = columns[0]
            for col in columns:
                if any(k in str(col).lower() for k in ['time', 'date', '时间']):
                    default_x = col;
                    break

            col_map = {c: get_smart_name(c) for c in columns}
            x_col = st.selectbox("X 轴 (时间):", columns, index=columns.index(default_x),
                                 format_func=lambda x: col_map[x])

            st.divider()

            # 3. 轴配置
            st.subheader("📈 轴配置")
            numeric_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
            valid_y = [c for c in columns if c in numeric_cols and c != x_col]

            y_left = st.multiselect("⬅️ 左 Y 轴:", valid_y, default=[valid_y[0]] if valid_y else [],
                                    format_func=lambda x: col_map[x])
            remaining_y = [c for c in valid_y if c not in y_left]
            y_right = st.multiselect("➡️ 右 Y 轴:", remaining_y, format_func=lambda x: col_map[x])

            st.divider()

            # 4. 功能开关
            st.subheader("🛠️ 高级功能")
            st.markdown("**辅助线显示:**")
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            show_avg = col_opt1.checkbox("平均值", False)
            show_max = col_opt2.checkbox("最大值", False)
            show_min = col_opt3.checkbox("最小值", False)

            st.markdown("**智能诊断:**")
            enable_diag = st.checkbox("开启内存泄漏/OOM分析", False)
            mem_limit_input = 0.0
            target_mem_col = None
            if enable_diag:
                mem_candidates = [c for c in (y_left + y_right) if "mem" in c.lower() or "内存" in str(col_map[c])]
                default_mem = mem_candidates[0] if mem_candidates else (y_left[0] if y_left else None)
                target_mem_col = st.selectbox("诊断目标列:", y_left + y_right,
                                              index=(y_left + y_right).index(default_mem) if default_mem else 0,
                                              format_func=lambda x: col_map[x])
                mem_limit_input = st.number_input("OOM 阈值 (0不检测):", value=0.0)

        if x_col and (y_left or y_right):
            df = df_raw.copy()
            df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            df = df.dropna(subset=[x_col]).sort_values(by=x_col)

            # --- 时间筛选 ---
            min_date, max_date = df[x_col].min(), df[x_col].max()
            if min_date and max_date and min_date != max_date:
                range_start, range_end = st.slider(
                    "⏳ 时间范围筛选:",
                    min_value=min_date.to_pydatetime(),
                    max_value=max_date.to_pydatetime(),
                    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                    format="MM-DD HH:mm"
                )
                df = df[(df[x_col] >= range_start) & (df[x_col] <= range_end)]

            # --- 智能诊断 ---
            diag_result = None;
            trend_line = None
            if enable_diag and target_mem_col and not df.empty:
                diag_result, (slope, intercept) = diagnose_memory(df, target_mem_col, mem_limit_input)
                trend_line = slope * np.arange(len(df)) + intercept

            # --- 绘图 ---
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            time_fmt = "%Y.%m.%d %H:%M:%S"
            colors = px.colors.qualitative.Plotly


            def add_series(col_name, is_secondary, color_idx):
                series_color = colors[color_idx % len(colors)]
                # 画主曲线
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[col_name], name=f"{col_map[col_name]}",
                    mode='lines', line=dict(width=2, color=series_color, dash='dot' if is_secondary else 'solid')
                ), secondary_y=is_secondary)

                # 画诊断趋势线
                if enable_diag and col_name == target_mem_col and trend_line is not None and not is_secondary:
                    fig.add_trace(go.Scatter(
                        x=df[x_col], y=trend_line, name="📈 趋势线",
                        mode='lines', line=dict(width=2, color='red', dash='dash'), opacity=0.7
                    ), secondary_y=False)

                # 画 Min/Max/Avg 辅助线
                stats_val = []
                if show_avg: stats_val.append((df[col_name].mean(), "Avg", "dash"))
                if show_max: stats_val.append((df[col_name].max(), "Max", "dot"))
                if show_min: stats_val.append((df[col_name].min(), "Min", "dot"))

                for val, label, dash_style in stats_val:
                    fig.add_hline(
                        y=val, line_dash=dash_style, line_width=1, line_color=series_color,
                        annotation_text=f"{label}:{val:.2f}",
                        annotation_position="top right" if not is_secondary else "top left",
                        secondary_y=is_secondary
                    )


            for i, col in enumerate(y_left): add_series(col, False, i)
            for j, col in enumerate(y_right): add_series(col, True, len(y_left) + j)

            if enable_diag and mem_limit_input > 0:
                fig.add_hline(y=mem_limit_input, line_dash="solid", line_color="red", line_width=2,
                              annotation_text=f"OOM阈值:{mem_limit_input}")

            fig.update_layout(
                title_text=f"{get_smart_name(selected_sheet_raw)} 趋势图",
                title_x=0.5, title_font=dict(size=20),
                hovermode="x unified", height=550,
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
                margin=dict(l=20, r=20, t=80, b=20)
            )
            fig.update_xaxes(tickformat=time_fmt, tickangle=-30)
            fig.update_yaxes(title_text="主数值", secondary_y=False)
            fig.update_yaxes(title_text="对比数值", secondary_y=True, showgrid=False)

            st.plotly_chart(fig, use_container_width=True)

            # --- 诊断报告 ---
            if enable_diag and diag_result:
                st.subheader("🧠 智能诊断报告")
                color = "green" if diag_result["status"] == "normal" else (
                    "orange" if diag_result["status"] == "warning" else "red")
                msg = "\n".join(diag_result["messages"])

                # 根据状态显示不同颜色的提示框
                if color == "green":
                    st.success(f"**诊断结果**: \n\n{msg}")
                elif color == "orange":
                    st.warning(f"**诊断结果**: \n\n{msg}")
                else:
                    st.error(f"**诊断结果**: \n\n{msg}")

            # --- 统计表格 (已修复括号问题) ---
            st.subheader("📊 详细统计数据")
            if not df.empty:
                stats_data = []
                for col in y_left + y_right:
                    s = df[col]
                    stats_data.append({
                        "数据指标": col_map[col],
                        "最小值": s.min(),
                        "最大值": s.max(),
                        "平均值": s.mean(),
                        "当前值": s.iloc[-1],
                        "波动范围": s.max() - s.min()
                    })

                stats_df = pd.DataFrame(stats_data)

                # 🚀 修复点：提取计算最大值的逻辑，避免在 column_config 里写太复杂的单行代码
                max_fluctuation = stats_df["波动范围"].max() if not stats_df.empty else 100

                st.dataframe(
                    stats_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "数据指标": st.column_config.TextColumn("指标名称", help="数据列名"),
                        "最小值": st.column_config.NumberColumn("最小值 (Min)", format="%.2f"),
                        "最大值": st.column_config.NumberColumn("最大值 (Max)", format="%.2f"),
                        "平均值": st.column_config.NumberColumn("平均值 (Avg)", format="%.2f"),
                        "当前值": st.column_config.NumberColumn("当前值 (Current)", format="%.2f"),
                        # 使用提取出来的 max_fluctuation 变量
                        "波动范围": st.column_config.ProgressColumn(
                            "波动幅度",
                            format="%.2f",
                            min_value=0,
                            max_value=max_fluctuation
                        ),
                    }
                )

                # 下载按钮
                csv = stats_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下载统计报告", csv, "report.csv", "text/csv")

        else:
            st.info("👈 请选择 Y 轴数据")

    except Exception as e:
        st.error(f"Error: {e}")