import io
import re
from collections import Counter
from datetime import timedelta
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

REQUIRED_COLUMNS = [
    "date",
    "platform",
    "brand",
    "text",
    "likes",
    "comments",
    "saves",
    "category",
]

STOP_PHRASES = {
    "爆款",
    "好物",
    "推荐",
    "链接",
    "官方",
    "福利",
    "活动",
    "下单",
    "购买",
    "必买",
    "入手",
    "同款",
    "专属",
    "新品",
    "测评",
    "分享",
    "真的",
    "这个",
    "那个",
    "今天",
    "感觉",
    "适合",
    "评论",
    "评论区",
    "论区",
    "内容",
    "收藏",
    "场景",
    "继续",
    "开始",
    "真实",
    "体验",
    "反馈",
    "需要",
    "话术",
    "标题",
    "模板",
    "一点",
    "一个",
    "这次",
    "当前",
    "热度",
    "感香",
    "围感香",
}

GENERIC_MARKETING_WORDS = ["官方", "链接", "爆款", "好物", "推荐", "购买", "下单", "福利"]

LEVEL_DESCRIPTIONS = {
    "夯": "强烈推荐，适合直接跟进",
    "人上人": "表现不错，建议小范围试水",
    "NPC": "普通，继续观察",
    "拉完了": "热度或可用性较差，不建议使用",
}
LEVEL_ORDER = ["夯", "人上人", "NPC", "拉完了"]
LEVEL_COLORS = {
    "夯": "#5B5BD6",
    "人上人": "#2F9A83",
    "NPC": "#9AA4B2",
    "拉完了": "#D7795A",
}
LEVEL_STYLE_CLASSES = {
    "夯": "level-callout-top",
    "人上人": "level-callout-good",
    "NPC": "level-callout-neutral",
    "拉完了": "level-callout-low",
}

DISPLAY_LABELS = {
    "phrase": "候选短语",
    "freq_recent": "近7天频次",
    "freq_prev": "前7天频次",
    "growth_rate": "增长率",
    "brand_coverage": "品牌覆盖数",
    "platform_coverage": "平台覆盖数",
    "pmi_like_score": "语义凝固度",
    "engagement_score": "互动强度",
    "usability_score": "可用性评分",
    "lifecycle": "生命周期",
    "decision": "热点等级",
    "risk_tip": "风险提示",
}


def load_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(BASE_DIR / "sample_hotspot_data.csv")
    return validate_and_prepare(df)


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少字段: {', '.join(missing)}")

    prepared = df[REQUIRED_COLUMNS].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    for col in ["likes", "comments", "saves"]:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0).astype(int)
    prepared = prepared.dropna(subset=["date", "text"])
    prepared["text"] = prepared["text"].astype(str)
    return prepared.sort_values("date").reset_index(drop=True)


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@[\w\u4e00-\u9fff-]+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", "", text)
    return text


def extract_char_ngrams(text: str, n_values=(2, 3)) -> list[str]:
    cleaned = clean_text(text)
    phrases = []
    for n in n_values:
        phrases.extend(cleaned[i : i + n] for i in range(max(len(cleaned) - n + 1, 0)))
    return [phrase for phrase in phrases if is_valid_phrase(phrase)]


def is_valid_phrase(phrase: str) -> bool:
    if not phrase or phrase in STOP_PHRASES:
        return False
    if phrase.isdigit():
        return False
    if any(word == phrase for word in GENERIC_MARKETING_WORDS):
        return False
    if len(set(phrase)) == 1:
        return False
    if not re.search(r"[\u4e00-\u9fff]", phrase):
        return False
    return True


def normalized_growth(recent: int, previous: int) -> float:
    if previous == 0:
        return float(recent) if recent > 0 else 0.0
    return (recent - previous) / previous


def calculate_pmi_like_score(phrase: str, phrase_freq: int, char_counter: Counter, total_chars: int) -> float:
    if total_chars == 0 or phrase_freq == 0:
        return 0.0
    phrase_prob = phrase_freq / total_chars
    char_probs = [max(char_counter.get(char, 0) / total_chars, 1e-9) for char in phrase]
    independent_prob = float(np.prod(char_probs))
    return round(float(np.log2(phrase_prob / independent_prob + 1e-9)), 3)


def build_candidates(
    df: pd.DataFrame,
    min_freq: int,
    min_brand_coverage: int,
    growth_threshold: float,
    n_values=(2, 3),
) -> pd.DataFrame:
    working = df.copy()
    working["clean_text"] = working["text"].map(clean_text)
    max_date = working["date"].max()
    recent_start = max_date - timedelta(days=6)
    prev_start = max_date - timedelta(days=13)

    recent_df = working[working["date"] >= recent_start]
    prev_df = working[(working["date"] >= prev_start) & (working["date"] < recent_start)]

    phrase_rows = []
    char_counter = Counter("".join(working["clean_text"].tolist()))
    total_chars = sum(char_counter.values())

    for row in working.itertuples(index=False):
        phrases = extract_char_ngrams(row.text, n_values=n_values)
        for phrase in phrases:
            phrase_rows.append(
                {
                    "phrase": phrase,
                    "date": row.date,
                    "platform": row.platform,
                    "brand": row.brand,
                    "likes": row.likes,
                    "comments": row.comments,
                    "saves": row.saves,
                    "category": row.category,
                }
            )

    if not phrase_rows:
        return pd.DataFrame()

    phrase_df = pd.DataFrame(phrase_rows)
    all_counter = Counter(phrase_df["phrase"])
    recent_counter = Counter(phrase_df[phrase_df["date"] >= recent_start]["phrase"])
    prev_counter = Counter(
        phrase_df[(phrase_df["date"] >= prev_start) & (phrase_df["date"] < recent_start)]["phrase"]
    )

    rows = []
    for phrase, total_freq in all_counter.items():
        freq_recent = recent_counter.get(phrase, 0)
        freq_prev = prev_counter.get(phrase, 0)
        if freq_recent < min_freq:
            continue

        subset = phrase_df[phrase_df["phrase"] == phrase]
        brand_coverage = subset["brand"].nunique()
        platform_coverage = subset["platform"].nunique()
        if brand_coverage < min_brand_coverage:
            continue

        growth_rate = normalized_growth(freq_recent, freq_prev)
        engagement_raw = (
            subset["likes"].sum() * 1.0 + subset["comments"].sum() * 2.0 + subset["saves"].sum() * 1.5
        ) / max(total_freq, 1)
        pmi_like_score = calculate_pmi_like_score(phrase, total_freq, char_counter, total_chars)
        coverage_score = min(brand_coverage / 4, 1) * 15 + min(platform_coverage / 3, 1) * 10
        growth_cap = growth_threshold * 4 if growth_threshold > 0 else 4
        growth_score = min(max(growth_rate, 0), growth_cap) * 10
        engagement_score = min(np.log1p(engagement_raw) * 6, 40)
        pmi_score = min(max(pmi_like_score, 0) * 3, 25)
        usability_score = round(float(min(100, coverage_score + growth_score + engagement_score + pmi_score)), 2)

        lifecycle = classify_lifecycle(freq_recent, freq_prev, growth_rate, brand_coverage)
        decision = make_decision(usability_score, lifecycle)
        risk = make_risk_tip(phrase, lifecycle, brand_coverage, platform_coverage)

        rows.append(
            {
                "phrase": phrase,
                "freq_recent": freq_recent,
                "freq_prev": freq_prev,
                "growth_rate": round(growth_rate, 3),
                "brand_coverage": brand_coverage,
                "platform_coverage": platform_coverage,
                "pmi_like_score": pmi_like_score,
                "engagement_score": round(float(engagement_score), 2),
                "usability_score": usability_score,
                "lifecycle": lifecycle,
                "decision": decision,
                "risk_tip": risk,
                "llm_prompt": generate_llm_prompt(
                    {
                        "phrase": phrase,
                        "freq_recent": freq_recent,
                        "freq_prev": freq_prev,
                        "growth_rate": round(growth_rate, 3),
                        "brand_coverage": brand_coverage,
                        "platform_coverage": platform_coverage,
                        "lifecycle": lifecycle,
                        "decision": decision,
                        "risk_tip": risk,
                    }
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(["usability_score", "freq_recent"], ascending=False).reset_index(drop=True)


def classify_lifecycle(freq_recent: int, freq_prev: int, growth_rate: float, brand_coverage: int) -> str:
    if freq_recent < freq_prev:
        return "衰减期"
    if freq_recent >= 10 and growth_rate >= 0.2:
        return "爆发中"
    if freq_recent < 8 and growth_rate >= 1.0 and brand_coverage >= 2:
        return "潜力期"
    if growth_rate >= 0.5 and brand_coverage >= 2:
        return "潜力期"
    if freq_recent < 6 and growth_rate < 0.5:
        return "低热观察"
    return "低热观察"


def make_decision(usability_score: float, lifecycle: str) -> str:
    if usability_score >= 75 and lifecycle in {"爆发中", "潜力期"}:
        return "夯"
    if lifecycle == "低热观察" and usability_score >= 70:
        return "人上人"
    if usability_score >= 60 and lifecycle != "衰减期":
        return "人上人"
    if lifecycle == "衰减期" or usability_score < 45:
        return "拉完了"
    return "NPC"


def make_risk_tip(phrase: str, lifecycle: str, brand_coverage: int, platform_coverage: int) -> str:
    risks = []
    if lifecycle == "衰减期":
        risks.append("热度可能已经衰减")
    if len(phrase) <= 2:
        risks.append("语义不完整，需人工复核")
    if brand_coverage <= 1:
        risks.append("可能与品牌调性不匹配")
    if platform_coverage <= 1:
        risks.append("跨平台扩散不足")
    if any(word in phrase for word in ["笑", "疯", "躺", "摆"]):
        risks.append("可能过度娱乐化")
    return "；".join(risks) if risks else "风险较低，但仍需人工复核语境"


def generate_llm_prompt(row: dict) -> str:
    return f"""你是消费品牌内容策略顾问。请基于以下模拟热点候选短语，输出一段适合运营团队阅读的解释。

候选短语：{row['phrase']}
近 7 天频次：{row['freq_recent']}
前 7 天频次：{row['freq_prev']}
增长率：{row['growth_rate']}
品牌覆盖数：{row['brand_coverage']}
平台覆盖数：{row['platform_coverage']}
生命周期：{row['lifecycle']}
热点等级：{row['decision']}
风险提示：{row['risk_tip']}

请按以下结构输出：
1. 语义解释
2. 情绪倾向
3. 商业价值
4. 内容建议
5. 风险提示
6. CTA 引导

注意：数据为虚构模拟数据，不要声称来自真实平台或真实品牌。"""


def build_markdown_report(candidates: pd.DataFrame) -> str:
    top = candidates.head(10)
    lines = [
        "# AI 辅助热点洞察与趋势决策报告",
        "",
        "说明：本报告基于本地模拟数据生成，不连接真实平台，不包含公司内部数据。",
        "",
        "## Top 10 候选热梗",
        "",
    ]
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        lines.extend(
            [
                f"### {idx}. {row.phrase}",
                "",
                f"- 近 7 天频次：{row.freq_recent}",
                f"- 前 7 天频次：{row.freq_prev}",
                f"- 增长率：{row.growth_rate}",
                f"- 品牌覆盖：{row.brand_coverage}",
                f"- 平台覆盖：{row.platform_coverage}",
                f"- 生命周期：{row.lifecycle}",
                f"- 热点等级：{row.decision}",
                f"- 可用性评分：{row.usability_score}",
                f"- 风险提示：{row.risk_tip}",
                "",
                "如需生成语义解释、内容建议和 CTA，请在页面选择该词条并复制 AI Prompt。",
                "",
            ]
        )
    return "\n".join(lines)


def apply_ui_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            background: #F7F8FC;
            border: 1px solid #E7E9F2;
            border-radius: 14px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] {
            color: #697386;
            font-size: 0.82rem;
        }
        [data-testid="stMetricValue"] {
            color: #171A2B;
            font-weight: 700;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #E7E9F2;
            border-radius: 12px;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            min-height: 58px;
            border-radius: 14px;
            background: #F7F8FC;
            border-color: #DDE1EA;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] span {
            font-size: 1.08rem;
        }
        [data-testid="stSelectbox"] label {
            font-size: 1.05rem;
            font-weight: 600;
        }
        .stAlert {
            border-radius: 12px;
        }
        .phrase-section {
            background: #FAFBFF;
            border: 1px solid #E7E9F2;
            border-radius: 14px;
            padding: 18px 20px 20px;
            min-height: 150px;
        }
        .phrase-section-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .phrase-section-title {
            color: #171A2B;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .phrase-section-count {
            color: #697386;
            font-size: 0.85rem;
        }
        .phrase-section-caption {
            color: #697386;
            font-size: 0.86rem;
            margin-bottom: 14px;
        }
        .phrase-cloud {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            max-height: 190px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .phrase-chip {
            background: #EEEDFF;
            border: 1px solid #D9D7FF;
            border-radius: 999px;
            color: #4744A8;
            font-size: 0.9rem;
            line-height: 1.4;
            padding: 6px 11px;
        }
        .phrase-chip.good {
            background: #E8F6F1;
            border-color: #C8E9DE;
            color: #237963;
        }
        .level-callout {
            border-radius: 16px;
            padding: 18px 20px 20px;
            margin: 12px 0 18px;
        }
        .level-callout-top {
            background: #EEEDFF;
            border: 1px solid #D9D7FF;
        }
        .level-callout-good {
            background: #E8F6F1;
            border: 1px solid #C8E9DE;
        }
        .level-callout-neutral {
            background: #F2F4F7;
            border: 1px solid #E0E4EA;
        }
        .level-callout-low {
            background: #FFF0EA;
            border: 1px solid #F3D0C2;
        }
        .level-callout-label {
            color: #697386;
            font-size: 0.86rem;
            margin-bottom: 6px;
        }
        .level-callout-main {
            color: #171A2B;
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 6px;
        }
        .level-callout-description {
            color: #4B5565;
            font-size: 1.02rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_bar_chart(top_candidates: pd.DataFrame) -> None:
    if top_candidates.empty:
        st.info("暂无可展示的候选热梗。")
        return

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    chart_data = top_candidates.sort_values("usability_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    colors = [LEVEL_COLORS.get(level, "#9AA4B2") for level in chart_data["decision"]]
    ax.barh(chart_data["phrase"], chart_data["usability_score"], color=colors)
    ax.set_xlabel("可用性评分", color="#697386")
    ax.set_ylabel("")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, color="#DDE1EA", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDE1EA")
    ax.tick_params(axis="y", length=0, colors="#252A3A", labelsize=10)
    ax.tick_params(axis="x", colors="#697386", labelsize=9)
    for index, value in enumerate(chart_data["usability_score"]):
        ax.text(value + 1.2, index, f"{value:.1f}", va="center", color="#252A3A", fontsize=9)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_level_distribution(candidates: pd.DataFrame) -> None:
    counts = candidates["decision"].value_counts().reindex(LEVEL_ORDER, fill_value=0)
    chart_data = counts.iloc[::-1]

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    colors = [LEVEL_COLORS[level] for level in chart_data.index]
    ax.barh(chart_data.index, chart_data.values, color=colors, height=0.58)
    ax.set_xlim(0, max(int(chart_data.max()) * 1.22, 5))
    ax.set_xlabel("候选数量", color="#697386")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", linewidth=0.8, color="#DDE1EA", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDE1EA")
    ax.tick_params(axis="y", length=0, colors="#252A3A", labelsize=10)
    ax.tick_params(axis="x", colors="#697386", labelsize=9)
    for index, value in enumerate(chart_data.values):
        ax.text(value + 0.35, index, str(int(value)), va="center", color="#252A3A", fontsize=10)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_priority_phrase_lists(candidates: pd.DataFrame) -> None:
    sections = [
        ("夯", "level-top", "建议直接跟进的重点词条", "优先做内容测试，并结合语境人工复核。"),
        ("人上人", "level-good", "适合小范围试水的词条", "可以先做轻量内容，观察收藏、评论和转发反馈。"),
    ]
    left, right = st.columns(2)

    for column, (level, css_class, subtitle, description) in zip((left, right), sections):
        level_candidates = candidates[candidates["decision"] == level].sort_values(
            ["usability_score", "freq_recent"], ascending=False
        )
        chips = "".join(
            f'<span class="phrase-chip {"good" if level == "人上人" else ""}">{escape(str(row.phrase))}</span>'
            for row in level_candidates.itertuples(index=False)
        )
        if not chips:
            chips = '<span class="phrase-section-caption">当前筛选条件下暂无词条。</span>'

        with column:
            st.markdown(
                f"""
                <div class="phrase-section {css_class}">
                    <div class="phrase-section-head">
                        <span class="phrase-section-title">{level} · {subtitle}</span>
                        <span class="phrase-section-count">{len(level_candidates)} 条</span>
                    </div>
                    <div class="phrase-section-caption">{description}</div>
                    <div class="phrase-cloud">{chips}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_level_callout(level: str) -> None:
    style_class = LEVEL_STYLE_CLASSES.get(level, "level-callout-neutral")
    description = LEVEL_DESCRIPTIONS.get(level, "请结合语境人工复核")
    st.markdown(
        f"""
        <div class="level-callout {style_class}">
            <div class="level-callout-label">当前热点等级</div>
            <div class="level-callout-main">{escape(level)}</div>
            <div class="level-callout-description">{escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="AI 辅助热点洞察与趋势决策工作流", layout="wide")
    apply_ui_styles()
    st.title("AI 辅助热点洞察与趋势决策工作流")
    st.caption("本 demo 仅使用模拟数据，本地运行，不爬取网站，不调用外部 API。")
    st.caption("热点等级：夯＝直接跟进 · 人上人＝小范围试水 · NPC＝继续观察 · 拉完了＝不建议使用")

    with st.expander("项目说明", expanded=True):
        st.write(
            "这是一个作品集用的简化工作流：从模拟热点内容进入，经过清洗、候选短语提取、"
            "增长与覆盖判断、生命周期标注、AI Prompt 生成，最后输出可下载报告。"
        )

    uploaded_file = st.sidebar.file_uploader("上传自己的 CSV", type=["csv"])
    min_freq = st.sidebar.slider("min_freq：近 7 天最低频次", min_value=1, max_value=20, value=3)
    min_brand_coverage = st.sidebar.slider("min_brand_coverage：最低品牌覆盖", 1, 5, 2)
    growth_threshold = st.sidebar.slider("growth_threshold：增长判断阈值", 0.0, 3.0, 0.5, 0.1)
    ngram_mode = st.sidebar.radio("候选短语长度", ["2-gram + 3-gram", "仅 2-gram", "仅 3-gram"], index=0)
    n_values = {"2-gram + 3-gram": (2, 3), "仅 2-gram": (2,), "仅 3-gram": (3,)}[ngram_mode]

    try:
        df = load_data(uploaded_file)
    except Exception as exc:
        st.error(f"数据读取失败：{exc}")
        return

    st.subheader("数据概览")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("内容条数", len(df))
    col_b.metric("覆盖平台", df["platform"].nunique())
    col_c.metric("模拟品牌", df["brand"].nunique())
    col_d.metric("日期跨度", f"{df['date'].min().date()} 至 {df['date'].max().date()}")
    with st.expander(f"查看原始数据（前 20 条 / 共 {len(df)} 条）", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("候选热梗结果表")
    candidates = build_candidates(df, min_freq, min_brand_coverage, growth_threshold, n_values=n_values)
    if candidates.empty:
        st.warning("当前筛选条件下没有识别到候选热梗。建议先把 min_freq 调到 3、品牌覆盖调到 2，再逐步收紧条件。")
        return

    top_candidate = candidates.iloc[0]
    decision_counts = candidates["decision"].value_counts()
    st.info(
        f"当前识别出 {len(candidates)} 个候选短语。优先关注「{top_candidate['phrase']}」："
        f"可用性评分 {top_candidate['usability_score']}，当前热点等级为「{top_candidate['decision']}」。"
    )
    summary_a, summary_b, summary_c, summary_d, summary_e = st.columns(5)
    summary_a.metric("候选短语数", len(candidates))
    summary_b.metric("夯", int(decision_counts.get("夯", 0)))
    summary_c.metric("人上人", int(decision_counts.get("人上人", 0)))
    summary_d.metric("NPC", int(decision_counts.get("NPC", 0)))
    summary_e.metric("拉完了", int(decision_counts.get("拉完了", 0)))

    compact_cols = [
        "phrase",
        "usability_score",
        "growth_rate",
        "lifecycle",
        "decision",
        "risk_tip",
    ]
    display_cols = [
        "phrase",
        "freq_recent",
        "freq_prev",
        "growth_rate",
        "brand_coverage",
        "platform_coverage",
        "pmi_like_score",
        "engagement_score",
        "usability_score",
        "lifecycle",
        "decision",
        "risk_tip",
    ]
    st.caption("默认展示最适合快速判断的关键指标；完整评分明细已收起。")
    st.dataframe(
        candidates[compact_cols].rename(columns=DISPLAY_LABELS),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("查看完整评分明细", expanded=False):
        st.dataframe(
            candidates[display_cols].rename(columns=DISPLAY_LABELS),
            use_container_width=True,
            hide_index=True,
        )

    chart_left, chart_right = st.columns([1.55, 1])
    top10 = candidates.head(10)
    with chart_left:
        st.subheader("Top 10 热梗排名")
        st.caption("按可用性评分排序，分数越高越值得优先关注。")
        render_bar_chart(top10)
    with chart_right:
        st.subheader("热点等级分布")
        st.caption("看整体候选池中，各等级所占数量。")
        render_level_distribution(candidates)

    st.subheader("重点词条清单")
    st.caption("把等级分布进一步拆开，直接查看“夯”和“人上人”分别包含哪些词条。")
    render_priority_phrase_lists(candidates)

    st.subheader("选择热梗，查看详细结果")
    st.info("请选择一个候选热梗，下面的评分、热点等级、风险提示和 AI Prompt 会同步更新。")
    selected_phrase = st.selectbox(
        "选择候选热梗",
        candidates["phrase"].tolist(),
        key="selected_hotspot_phrase",
    )
    selected = candidates[candidates["phrase"] == selected_phrase].iloc[0]

    st.subheader("所选热梗分析")
    selected_state_key = f"{selected_phrase}_{selected['usability_score']}_{selected['decision']}"
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 规则判断结果")
        st.caption("这些结果由本地规则根据频次、增长、覆盖和互动数据计算。")
        st.metric("当前热梗可用性评分", selected["usability_score"])
        st.write(f"生命周期：**{selected['lifecycle']}**")
        render_level_callout(selected["decision"])
        st.write(f"风险提示：{selected['risk_tip']}")
    with right:
        st.markdown("#### 交给大模型的 AI Prompt")
        st.caption("复制这段 Prompt 到 Kimi、通义或 ChatGPT，让大模型根据当前数据制定内容建议。")
        st.text_area(
            "提示词内容",
            selected["llm_prompt"],
            height=360,
            key=f"llm_prompt_{selected_state_key}",
        )

    st.subheader("导出报告")
    csv_bytes = candidates.to_csv(index=False).encode("utf-8-sig")
    markdown_report = build_markdown_report(candidates)
    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            "导出候选结果 CSV",
            data=csv_bytes,
            file_name="hotspot_candidates.csv",
            mime="text/csv",
        )
    with export_right:
        st.download_button(
            "导出 Markdown 报告",
            data=markdown_report.encode("utf-8"),
            file_name="hotspot_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
