import io
import json
import re
from collections import Counter, defaultdict
from datetime import timedelta
from html import escape
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
LEXICON_DIR = BASE_DIR / "lexicons"
LEXICON_ROLE_META = {
    "audience": {
        "role": "人群/场景",
        "usage_note": "用于判断可能的人群或使用场景，不代表真实用户调研结论。",
    },
    "ingredients": {
        "role": "成分/产品方向",
        "usage_note": "用于补充成分和产品方向背景，不代表产品真实配方或功效。",
    },
    "vertical_tags": {
        "role": "行业/品类标签",
        "usage_note": "用于判断美妆行业、品类或场景背景；标签命中不能直接解释候选词义。",
    },
    "pain_points": {
        "role": "用户痛点/需求",
        "usage_note": "用于提出可能的用户需求假设，不代表真实用户调研结论。",
    },
    "efficacy": {
        "role": "功效内容方向",
        "usage_note": "用于辅助内容关联；不能直接作为产品功效证明或宣传承诺。",
    },
}
ACTIVE_LEXICON_IDS = set(LEXICON_ROLE_META)
LOW_CONFIDENCE_NOTICE = "低信度，代人工核验"
EMBEDDING_MODEL = "text-embedding-3-small"
SEMANTIC_RAG_THRESHOLD = 0.58

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
CORE_INPUT_COLUMNS = ["date", "text"]
FIELD_LABELS = {
    "date": "日期/时间",
    "platform": "平台/来源",
    "brand": "品牌/主体",
    "text": "热词/标题/文本",
    "likes": "点赞数",
    "comments": "评论数",
    "saves": "收藏/转发数",
    "category": "品类/行业",
}
COLUMN_ALIASES = {
    "date": ["date", "日期", "发布时间", "上榜时间", "采集时间", "榜单日期", "时间", "time"],
    "platform": ["platform", "平台", "来源", "榜单", "渠道", "source"],
    "brand": ["brand", "品牌", "品牌名", "账号主体", "发布者", "主体", "作者"],
    "text": [
        "text",
        "文本",
        "内容",
        "标题",
        "热搜词",
        "关键词",
        "话题",
        "词条",
        "热词",
        "主题",
        "title",
        "keyword",
    ],
    "likes": ["likes", "点赞", "点赞数", "喜欢", "喜欢数"],
    "comments": ["comments", "评论", "评论数"],
    "saves": ["saves", "收藏", "收藏数", "转发", "转发数", "分享", "分享数"],
    "category": ["category", "品类", "行业", "分类", "赛道", "内容类别", "内容品类"],
}
MAPPING_SKIP_LABEL = "不使用该列"

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
GENERIC_STANDALONE_TERMS = {
    "妆容",
    "妆面",
    "美妆",
    "彩妆",
    "底妆",
    "眼妆",
    "唇妆",
    "护肤",
    "护肤品",
    "面膜",
    "精华",
    "面霜",
    "乳液",
    "洁面",
    "防晒",
    "香水",
    "香氛",
    "口红",
    "眼影",
    "腮红",
    "粉底",
    "产品",
    "品牌",
    "成分",
    "肌肤",
    "皮肤",
    "化妆品",
}

HOTNESS_COLORS = {
    "高": "#5B5BD6",
    "中": "#2F9A83",
    "低": "#9AA4B2",
}
LIFECYCLE_ORDER = ["爆发中", "潜力期", "低热观察", "衰减期"]
LIFECYCLE_COLORS = {
    "爆发中": "#5B5BD6",
    "潜力期": "#2F9A83",
    "低热观察": "#9AA4B2",
    "衰减期": "#D7795A",
}

BEAUTY_BRAND_TIERS = ["高端", "轻奢", "中端", "大众", "其他"]
BEAUTY_BRAND_ORIGINS = ["国际", "国货", "跨境/海外小众", "其他"]
BEAUTY_PRODUCT_STAGES = ["新品上新", "爆品主推", "常青款", "清库存/临期", "其他"]
BEAUTY_CATEGORIES = ["清洁", "护肤", "彩妆", "香氛", "个护身体/洗护", "其他"]
BEAUTY_TONES = ["专业科学", "年轻活泼", "高端克制", "温和可信", "自然有机", "性价比实用", "潮流先锋", "其他"]
CAMPAIGN_GOALS = ["提升曝光", "提升互动/种草", "引导咨询", "促进成交", "拉新/引流", "其他"]
NEED_ARCHETYPES = [
    "皮肤问题解决型",
    "成分机理研究型",
    "妆效表现追求型",
    "效率省事型",
    "性价比精算型",
    "高端体验享受型",
    "潮流尝鲜种草型",
    "安全温和谨慎型",
    "其他",
]
EFFICACY_GOALS = ["抗衰", "紧致淡纹", "美白提亮", "淡斑", "修护", "舒缓褪红", "控油祛痘", "清洁去黑头", "保湿", "其他"]
TARGET_GROUPS = ["学生", "职场", "宝妈", "熟龄", "其他"]
SKIN_TYPES = ["油皮", "干皮", "混合皮", "敏感肌", "痘肌", "未知", "其他"]

DEFAULT_BRAND_PROFILE = {
    "industry": "美妆",
    "brand_tier": "中端",
    "brand_origin": "国货",
    "product_stage": "常青款",
    "category": "护肤",
    "tone": "年轻活泼",
    "goal": "提升互动/种草",
    "audience_need": "皮肤问题解决型",
    "efficacy_goal": "修护",
    "target_group": "职场",
    "skin_type": "未知",
    "notes": "未补充其他品牌信息",
    "product_name": "未填写具体产品",
    "product_positioning": "未补充产品定位和核心卖点",
    "product_evidence": "未补充已确认的成分、功效或使用证据",
    "prohibited_claims": "未补充禁用或规避表达",
}

LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "semantic_explanation": {"type": "string"},
        "emotion": {"type": "string"},
        "business_value": {"type": "string"},
        "content_actions": {"type": "array", "items": {"type": "string"}},
        "risk_tips": {"type": "array", "items": {"type": "string"}},
        "cta": {"type": "string"},
    },
    "required": [
        "semantic_explanation",
        "emotion",
        "business_value",
        "content_actions",
        "risk_tips",
        "cta",
    ],
}

LLM_SYSTEM_INSTRUCTIONS = """
你是一个美妆品牌内容策略顾问，负责把跨行业榜单中的热点表达，转译成美妆内容运营可以使用的判断和行动建议。

请遵守以下规则：
1. 先理解热点表达，再做商业判断，不要看到一个词就直接推荐跟进。
2. 把输入中的统计数据视为数据事实；把美妆词库视为辅助参考；把你对语义、情绪和内容方向的判断视为推断。三者不能混写成同一种证据。
3. 不要编造平台、品牌、用户反馈、搜索量、产品功效或市场事实。输入没有提供的信息，明确说“无法从当前数据判断”。
4. RAG 置信度只表示词库检索证据是否充分，不代表热点显著性，也不代表产品功效真实有效；“原始语境”或“标签”命中的结果只能作为行业或场景背景，不能当作候选短语或具体词库条目的直接语义证据。
5. 痛点词库命中只能形成“可能的用户需求假设”；功效词库命中只能辅助内容方向，不能被写成产品功效事实或承诺。
6. 如果热点表达本身不自然、语义不完整、难以被普通中文用户理解，必须在风险提示中指出，并降低或否定跟进建议。
7. 功效、修护、美白、抗衰等表达不得自动写成产品功效承诺；涉及功效时，只能建议核验产品证据、合规边界和具体语境。
8. 内容行动必须能执行，至少说明内容形式、切入角度、如何连接当前美妆品牌，以及建议观察的指标。
9. 如果数据证据不足或品牌适配性低，直接给出“不建议跟进”或“先小范围验证”，不要为了输出完整而强行推荐。
10. 产品信息是用户提供的业务上下文，不是自动验证过的事实；没有填写或没有证据的产品成分、功效和合规结论，不得自行补全。
11. 严格按照用户消息中的 JSON Schema 返回结果，不输出 Schema 之外的字段。
""".strip()

INDUSTRY_GUIDANCE = {
    "通用": "优先判断这个热点是否容易转化为品牌内容场景，并关注用户是否愿意参与和分享。",
    "美妆": "重点关注妆容、护肤、使用场景、效果表达和情绪共鸣，避免无法验证的功效承诺。",
    "食品饮料": "重点关注口味、早餐、解馋、囤货、聚会和日常生活场景，避免夸大健康功效。",
    "家清": "重点关注清洁痛点、家庭场景、前后对比、效率和真实体验，避免过度承诺效果。",
    "数码": "重点关注通勤、效率、性能、续航、便携和真实体验，避免只追逐娱乐化表达。",
}

GOAL_GUIDANCE = {
    "提升曝光": "优先建议低门槛、容易理解和容易扩散的内容形式，并给出可观察的传播信号。",
    "提升互动/种草": "优先建议提问、评论区参与、投票、共创或用户分享等互动机制。",
    "引导咨询": "优先建议围绕肤质、功效和使用场景设计问题，引导用户评论区咨询。",
    "促进成交": "优先建议把热点与产品使用场景、购买理由和明确 CTA 连接起来。",
    "拉新/引流": "优先建议适合新用户理解和参与的内容，并设计关注、私信或落地页引导。",
}

DISPLAY_LABELS = {
    "phrase": "候选短语",
    "freq_recent": "近7天频次",
    "freq_prev": "前7天频次",
    "growth_rate": "增长率",
    "trend_label": "趋势判断",
    "brand_coverage": "品牌覆盖数",
    "platform_coverage": "平台覆盖数",
    "content_support": "独立内容支持数",
    "pmi_score": "PMI非随机共现分数",
    "left_entropy": "左边界熵",
    "right_entropy": "右边界熵",
    "phrase_quality_score": "短语质量分数",
    "hotness_score": "热度趋势分",
    "hotness_level": "热度趋势",
    "expression_quality": "表达质量",
    "review_reason": "复查原因",
    "evidence_context": "原始语境证据",
    "engagement_score": "互动强度",
    "beauty_usability_score": "美妆可用性",
    "usability_score": "美妆可用性",
    "lifecycle": "生命周期",
    "risk_tip": "风险提示",
    "rag_confidence": "RAG置信度",
    "rag_match_count": "词库匹配数",
    "high_freq_cutoff": "动态爆发频次线",
}


@st.cache_data(show_spinner=False)
def load_raw_data_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV 编码，请另存为 UTF-8 或 GBK 编码后重试：{last_error}")


def load_raw_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        return load_raw_data_from_bytes(uploaded_file.getvalue())
    return load_raw_data_from_bytes((BASE_DIR / "sample_hotspot_data.csv").read_bytes())


@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes, mapping_items: Optional[tuple] = None) -> pd.DataFrame:
    df = load_raw_data_from_bytes(file_bytes)
    mapping = dict(mapping_items) if mapping_items else None
    return validate_and_prepare(df, mapping=mapping)


def load_data(uploaded_file=None, mapping: Optional[dict] = None) -> pd.DataFrame:
    mapping_items = tuple(sorted(mapping.items())) if mapping else None
    if uploaded_file is not None:
        return load_data_from_bytes(uploaded_file.getvalue(), mapping_items=mapping_items)
    return load_data_from_bytes((BASE_DIR / "sample_hotspot_data.csv").read_bytes(), mapping_items=mapping_items)


def normalize_column_name(value: object) -> str:
    normalized = str(value).strip().lower()
    return re.sub(r"[\s_\-—·/:：()（）\[\]【】]+", "", normalized)


def infer_column_mapping(df: pd.DataFrame) -> tuple[dict, dict]:
    """按别名和简单数据类型推断榜单 CSV 到内部字段的映射。"""
    columns = list(df.columns)
    normalized_columns = {column: normalize_column_name(column) for column in columns}
    normalized_aliases = {
        field: {normalize_column_name(alias) for alias in aliases}
        for field, aliases in COLUMN_ALIASES.items()
    }
    mapping = {field: None for field in REQUIRED_COLUMNS}
    methods = {field: "未识别" for field in REQUIRED_COLUMNS}
    used_columns = set()

    for field in REQUIRED_COLUMNS:
        exact_candidates = [
            column
            for column in columns
            if column not in used_columns and normalized_columns[column] in normalized_aliases[field]
        ]
        if exact_candidates:
            mapping[field] = exact_candidates[0]
            methods[field] = "列名匹配"
            used_columns.add(exact_candidates[0])

    if mapping["date"] is None:
        date_candidates = []
        for column in columns:
            if column in used_columns:
                continue
            values = df[column].dropna().astype(str).str.replace(",", "", regex=False).str.replace("，", "", regex=False)
            numeric_ratio = float(pd.to_numeric(values, errors="coerce").notna().mean()) if len(values) else 0.0
            if numeric_ratio > 0.8:
                continue
            parsed = pd.to_datetime(df[column], errors="coerce")
            parse_ratio = float(parsed.notna().mean()) if len(df) else 0.0
            if parse_ratio >= 0.7:
                date_candidates.append((parse_ratio, column))
        if date_candidates:
            _, column = max(date_candidates)
            mapping["date"] = column
            methods["date"] = "日期格式推断"
            used_columns.add(column)

    if mapping["text"] is None:
        text_candidates = []
        for column in columns:
            if column in used_columns:
                continue
            values = df[column].dropna().astype(str).str.strip()
            if values.empty:
                continue
            numeric_ratio = float(pd.to_numeric(values, errors="coerce").notna().mean())
            if numeric_ratio > 0.8:
                continue
            median_length = float(values.str.len().median())
            unique_ratio = float(values.nunique() / max(len(values), 1))
            text_candidates.append((median_length * (0.5 + unique_ratio), column))
        if text_candidates:
            _, column = max(text_candidates)
            mapping["text"] = column
            methods["text"] = "文本特征推断"
            used_columns.add(column)

    return mapping, methods


def build_mapping_report(mapping: dict, methods: dict) -> pd.DataFrame:
    rows = []
    for field in REQUIRED_COLUMNS:
        source = mapping.get(field)
        if source:
            status = "已识别"
            note = methods.get(field, "自动识别")
        elif field in CORE_INPUT_COLUMNS:
            status = "缺少核心字段"
            note = "必须手动指定，否则无法完成趋势分析"
        else:
            status = "使用默认值"
            note = "该字段缺失时不会阻止分析"
        rows.append(
            {
                "系统字段": FIELD_LABELS[field],
                "输入列名": source or MAPPING_SKIP_LABEL,
                "状态": status,
                "说明": note,
            }
        )
    return pd.DataFrame(rows)


def mapping_warnings(mapping: dict) -> list[str]:
    warnings = []
    for field in CORE_INPUT_COLUMNS:
        if not mapping.get(field):
            warnings.append(f"缺少核心字段“{FIELD_LABELS[field]}”，请在字段映射中手动指定。")
    for field in REQUIRED_COLUMNS:
        if field not in CORE_INPUT_COLUMNS and not mapping.get(field):
            warnings.append(f"未识别“{FIELD_LABELS[field]}”，系统将使用默认值，相关指标可能不准确。")
    mapped_columns = [value for value in mapping.values() if value]
    duplicates = sorted({value for value in mapped_columns if mapped_columns.count(value) > 1})
    if duplicates:
        warnings.append(f"同一输入列被重复映射：{', '.join(duplicates)}，请检查字段映射。")
    return warnings


def render_column_mapping(raw_df: pd.DataFrame, default_mapping: dict, default_methods: dict) -> tuple[dict, dict]:
    st.caption("系统会先按常见别名自动识别；如果识别不准确，可以在这里手动调整。")
    mapping = {}
    methods = {}
    options = [MAPPING_SKIP_LABEL, *[str(column) for column in raw_df.columns]]
    left, right = st.columns(2)
    for index, field in enumerate(REQUIRED_COLUMNS):
        container = left if index % 2 == 0 else right
        default_source = default_mapping.get(field) or MAPPING_SKIP_LABEL
        default_index = options.index(str(default_source)) if str(default_source) in options else 0
        selected_source = container.selectbox(
            FIELD_LABELS[field],
            options,
            index=default_index,
            key=f"input_column_mapping_{field}",
        )
        mapping[field] = None if selected_source == MAPPING_SKIP_LABEL else selected_source
        methods[field] = (
            "用户手动指定"
            if mapping[field] != default_mapping.get(field)
            else default_methods.get(field, "自动识别")
        )
    st.dataframe(build_mapping_report(mapping, methods), use_container_width=True, hide_index=True)
    return mapping, methods


@st.cache_data(show_spinner=False)
def load_lexicon_entries() -> list[dict]:
    """读取美妆词库；词库只在首次分析时加载，后续从 Streamlit 缓存读取。"""
    entries = []
    if not LEXICON_DIR.exists():
        return entries

    for path in sorted(LEXICON_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("lexicon_id", path.stem) not in ACTIVE_LEXICON_IDS:
            continue
        for item in data.get("terms", []):
            term = str(item.get("term", "")).strip()
            if not term:
                continue
            entries.append(
                {
                    "lexicon_id": data.get("lexicon_id", path.stem),
                    "title": data.get("title", path.stem),
                    "role": LEXICON_ROLE_META.get(data.get("lexicon_id", path.stem), {}).get("role", "知识参考"),
                    "usage_note": LEXICON_ROLE_META.get(data.get("lexicon_id", path.stem), {}).get(
                        "usage_note", "仅作辅助参考，不代表事实。"
                    ),
                    "term": term,
                    "synonyms": [str(value).strip() for value in item.get("synonyms", []) if str(value).strip()],
                    "labels": [str(value).strip() for value in item.get("labels", []) if str(value).strip()],
                }
            )
    return entries


def _lexicon_embedding_text(entry: dict) -> str:
    """把一条词库记录压成适合语义检索的短文本。"""
    parts = [
        entry.get("role", ""),
        entry.get("title", ""),
        entry.get("term", ""),
        *entry.get("synonyms", []),
        *entry.get("labels", []),
    ]
    return "；".join(str(part).strip() for part in parts if str(part).strip())


@st.cache_data(ttl=86400, show_spinner=False)
def get_lexicon_embeddings(texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    """缓存词库向量，避免每次切换页面都重复调用 Embeddings API。"""
    api_key, _ = get_llm_settings()
    if not api_key:
        raise ValueError("尚未配置大模型 API Key")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=list(texts),
        encoding_format="float",
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    return tuple(tuple(float(value) for value in item.embedding) for item in ordered)


@st.cache_data(ttl=86400, show_spinner=False)
def get_query_embedding(text: str) -> tuple[float, ...]:
    """缓存当前候选查询向量；同一词条在页面 rerun 时不重复计费。"""
    api_key, _ = get_llm_settings()
    if not api_key:
        raise ValueError("尚未配置大模型 API Key")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        encoding_format="float",
    )
    return tuple(float(value) for value in response.data[0].embedding)


def retrieve_semantic_lexicon_matches(
    query_text: str,
    entries: list[dict],
    excluded_terms: set[tuple[str, str]],
    evidence_examples: list[str],
    product_context: str = "",
    top_k: int = 4,
) -> list[dict]:
    """用 Embeddings 找同义或近义词库条目；语义命中只作为辅助证据。"""
    if not entries or not query_text.strip():
        return []

    embedding_texts = tuple(_lexicon_embedding_text(entry) for entry in entries)
    lexicon_vectors = np.asarray(get_lexicon_embeddings(embedding_texts), dtype=float)
    semantic_query = (
        f"候选短语：{query_text}\n"
        f"原始语境：{'；'.join(evidence_examples[:3]) or '未提供'}\n"
        f"产品信息：{product_context or '未提供'}"
    )
    query_vector = np.asarray(get_query_embedding(semantic_query), dtype=float)
    query_norm = float(np.linalg.norm(query_vector))
    vector_norms = np.linalg.norm(lexicon_vectors, axis=1)
    if query_norm == 0:
        return []
    similarities = (lexicon_vectors @ query_vector) / np.maximum(vector_norms * query_norm, 1e-12)
    ranked_indices = np.argsort(similarities)[::-1]

    matches = []
    for index in ranked_indices:
        similarity = float(similarities[index])
        if similarity < SEMANTIC_RAG_THRESHOLD:
            break
        entry = entries[int(index)]
        entry_key = (entry["lexicon_id"], entry["term"])
        if entry_key in excluded_terms:
            continue
        matches.append(
            {
                "lexicon_id": entry["lexicon_id"],
                "title": entry["title"],
                "role": entry["role"],
                "usage_note": entry["usage_note"],
                "term": entry["term"],
                "labels": "、".join(entry["labels"]) or "未分类",
                "score": 1.0 + similarity,
                "matched_surfaces": "语义相似",
                "matched_details": f"语义相似度 {similarity:.3f}（候选短语、原始语境和产品信息）",
                "matched_types": ["语义相似"],
                "direct_match_count": 0,
                "candidate_match_count": 0,
                "match_scopes": ["语义相关"],
                "semantic_similarity": round(similarity, 3),
                "evidence_examples": evidence_examples or [f"候选短语：{query_text}"],
            }
        )
        if len(matches) >= top_k:
            break
    return matches


def retrieve_lexicon_matches(
    text: str,
    top_k: int = 8,
    source_texts: Optional[list[str]] = None,
    product_context: Optional[str] = None,
    semantic: bool = False,
) -> list[dict]:
    """先做可解释词面检索；需要时再追加 Embeddings 语义召回。"""
    evidence_texts = [str(item) for item in (source_texts or [text]) if str(item).strip()]
    product_context = str(product_context or "").strip()
    normalized_query = clean_text(text)
    normalized_product = clean_text(product_context)
    normalized_text = clean_text(" ".join([text, *evidence_texts, product_context]))
    if not normalized_text:
        return []

    matches = []
    label_groups = {}
    direct_entry_keys = set()
    entries = load_lexicon_entries()
    for entry in entries:
        direct_candidates = [
            (entry["term"], 1.0),
            *((synonym, 0.9) for synonym in entry["synonyms"]),
        ]
        matched = []
        for candidate, weight in direct_candidates:
            normalized_candidate = clean_text(candidate)
            if len(normalized_candidate) < 2 or normalized_candidate not in normalized_text:
                continue
            match_type = (
                "词条"
                if candidate == entry["term"]
                else "同义词"
                if candidate in entry["synonyms"]
                else "标签"
            )
            if normalized_candidate in normalized_query:
                match_scope = "候选短语"
            elif normalized_product and normalized_candidate in normalized_product:
                match_scope = "产品信息"
            else:
                match_scope = "原始语境"
            matched.append((candidate, weight, match_type, normalized_candidate, match_scope))
            direct_entry_keys.add((entry["lexicon_id"], entry["term"]))
        if matched:
            evidence_examples = []
            seen_evidence = set()
            for candidate, _, _, normalized_candidate, _ in matched:
                for example in extract_evidence_examples(normalized_candidate, evidence_texts, limit=2):
                    if example not in seen_evidence:
                        evidence_examples.append(example)
                        seen_evidence.add(example)
            matched_surfaces = [candidate for candidate, _, _, _, _ in matched]
            score = max(
                len(normalized_candidate) * weight
                for _, weight, _, normalized_candidate, _ in matched
            ) + len(matched) * 0.5
            matched_details = "、".join(
                dict.fromkeys(
                    f"{candidate}（{match_type}·{match_scope}）"
                    for candidate, _, match_type, _, match_scope in matched
                )
            )
            matches.append(
                {
                    "lexicon_id": entry["lexicon_id"],
                    "title": entry["title"],
                    "role": entry["role"],
                    "usage_note": entry["usage_note"],
                    "term": entry["term"],
                    "labels": "、".join(entry["labels"]) or "未分类",
                    "score": score,
                    "matched_surfaces": "、".join(dict.fromkeys(matched_surfaces)),
                    "matched_details": matched_details,
                    "matched_types": list(dict.fromkeys(match_type for _, _, match_type, _, _ in matched)),
                    "direct_match_count": len(matched),
                    "candidate_match_count": sum(
                        match_scope == "候选短语" for _, _, _, _, match_scope in matched
                    ),
                    "match_scopes": list(dict.fromkeys(match_scope for _, _, _, _, match_scope in matched)),
                    "evidence_examples": evidence_examples or [f"候选短语：{text}"],
                }
            )

        # 标签只保留为聚合后的背景证据，避免同一个“护肤”标签把大量具体条目全部召回。
        for label in entry["labels"]:
            normalized_label = clean_text(label)
            if len(normalized_label) < 2 or normalized_label not in normalized_text:
                continue
            if normalized_label in normalized_query:
                match_scope = "候选短语"
            elif normalized_product and normalized_label in normalized_product:
                match_scope = "产品信息"
            else:
                match_scope = "原始语境"
            key = (entry["lexicon_id"], normalized_label, match_scope)
            if key not in label_groups:
                label_groups[key] = {
                    "lexicon_id": entry["lexicon_id"],
                    "title": entry["title"],
                    "role": entry["role"],
                    "usage_note": entry["usage_note"],
                    "term": f"标签：{label}",
                    "labels": label,
                    "score": len(normalized_label) * 0.65,
                    "matched_surfaces": label,
                    "matched_details": f"{label}（标签·{match_scope}）",
                    "matched_types": ["标签"],
                    "direct_match_count": 0,
                    "candidate_match_count": 0,
                    "match_scopes": [match_scope],
                    "evidence_examples": extract_evidence_examples(
                        normalized_label, evidence_texts, limit=2
                    ),
                }

    matches.extend(label_groups.values())
    if semantic:
        semantic_matches = retrieve_semantic_lexicon_matches(
            text,
            entries,
            direct_entry_keys,
            extract_evidence_examples(text, evidence_texts, limit=3),
            product_context=product_context,
            top_k=max(2, min(4, top_k // 2)),
        )
        matches.extend(semantic_matches)

    return sorted(matches, key=lambda item: (item["score"], len(item["term"])), reverse=True)[:top_k]


def retrieve_hybrid_lexicon_matches(
    text: str,
    top_k: int = 8,
    source_texts: Optional[list[str]] = None,
    product_context: Optional[str] = None,
) -> tuple[list[dict], str]:
    """混合 RAG 的安全入口：语义层失败时回退到词面层，不阻断页面。"""
    try:
        return (
            retrieve_lexicon_matches(
                text,
                top_k=top_k,
                source_texts=source_texts,
                product_context=product_context,
                semantic=True,
            ),
            "",
        )
    except Exception as exc:
        fallback = retrieve_lexicon_matches(
            text,
            top_k=top_k,
            source_texts=source_texts,
            product_context=product_context,
            semantic=False,
        )
        return fallback, str(exc)


def classify_rag_confidence(matches: list[dict]) -> str:
    if not matches:
        return "低信度"
    source_count = len({item["lexicon_id"] for item in matches})
    candidate_match_count = sum(int(item.get("candidate_match_count", 0)) for item in matches)
    if candidate_match_count >= 3 and source_count >= 2:
        return "高信度"
    # 语义相似只能说明“值得参考”，不能和精确词面命中一起升级为高信度。
    # 它可以把原本没有直接命中的候选从低信度提升到中信度，但仍需人工核验。
    return "中信度"


def format_rag_context(matches: list[dict]) -> str:
    if not matches:
        return "暂无匹配的美妆词库条目，请主要依据热点统计和原始语境判断。"
    lines = []
    for item in matches:
        lines.append(
            f"- {item.get('role', '知识参考')} · {item['title']}：{item['term']}（标签：{item['labels']}；"
            f"命中依据：{item.get('matched_details', item.get('matched_surfaces', item['term']))}）"
        )
        lines.append(f"  - 使用边界：{item.get('usage_note', '仅作辅助参考，不代表事实。')}")
        for evidence in item.get("evidence_examples", []):
            lines.append(f"  - 原始证据：{evidence}")
    return "\n".join(lines)


def get_llm_settings() -> tuple[str, str]:
    """读取大模型配置；没有配置时保持纯本地 Prompt 模式。"""
    try:
        api_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
        model = str(st.secrets.get("OPENAI_MODEL", "gpt-5.6")).strip()
    except Exception:
        api_key = ""
        model = "gpt-5.6"
    return api_key, model or "gpt-5.6"


def call_llm(prompt: str) -> Tuple[bool, Union[dict, str]]:
    """调用 OpenAI Responses API，返回是否成功和结构化分析结果。"""
    api_key, model = get_llm_settings()
    if not api_key:
        return False, "尚未配置大模型 API Key。请先在本地或 Streamlit Cloud 的 Secrets 中配置。"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            instructions=LLM_SYSTEM_INSTRUCTIONS,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "hotspot_analysis",
                    "strict": True,
                    "schema": LLM_RESPONSE_SCHEMA,
                }
            },
        )
        raw_result = (response.output_text or "").strip()
        if not raw_result:
            return False, "大模型没有返回可展示的内容，请稍后重试。"
        result = json.loads(raw_result)
        if not isinstance(result, dict):
            return False, "大模型返回的结果格式不正确，请稍后重试。"
        return True, result
    except Exception as exc:
        return False, f"大模型调用失败，请检查 API Key、模型名称或账户额度。\n\n错误信息：{exc}"


def render_llm_result(result: dict) -> None:
    """将结构化的大模型结果拆成易读的内容卡片。"""
    st.markdown("#### 大模型分析结果")
    st.markdown(f"**语义解释**\n\n{result.get('semantic_explanation', '暂无')}")
    st.markdown(f"**情绪倾向**\n\n{result.get('emotion', '暂无')}")
    st.markdown(f"**商业价值**\n\n{result.get('business_value', '暂无')}")

    st.markdown("**内容行动**")
    actions = result.get("content_actions", [])
    if actions:
        for action in actions:
            st.markdown(f"- {action}")
    else:
        st.write("暂无内容行动建议")

    st.markdown("**风险提示**")
    risks = result.get("risk_tips", [])
    if risks:
        for risk in risks:
            st.markdown(f"- {risk}")
    else:
        st.write("暂无额外风险提示")

    st.markdown(f"**CTA 引导**\n\n{result.get('cta', '暂无')}")


def validate_and_prepare(df: pd.DataFrame, mapping: Optional[dict] = None) -> pd.DataFrame:
    if mapping is None:
        mapping, _ = infer_column_mapping(df)
    missing_core = [field for field in CORE_INPUT_COLUMNS if not mapping.get(field)]
    if missing_core:
        labels = "、".join(FIELD_LABELS[field] for field in missing_core)
        raise ValueError(f"无法识别核心字段：{labels}。请在字段映射中手动指定。")

    prepared = pd.DataFrame(index=df.index)
    defaults = {
        "platform": "未知平台",
        "brand": "未知品牌",
        "likes": 0,
        "comments": 0,
        "saves": 0,
        "category": "未分类",
    }
    for field in REQUIRED_COLUMNS:
        source = mapping.get(field)
        if source and source in df.columns:
            prepared[field] = df[source]
        else:
            prepared[field] = defaults.get(field, "")

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    for col in ["likes", "comments", "saves"]:
        prepared[col] = (
            pd.to_numeric(
                prepared[col].astype(str).str.replace(",", "", regex=False).str.replace("，", "", regex=False),
                errors="coerce",
            )
            .fillna(0)
            .clip(lower=0)
            .round()
            .astype(int)
        )
    prepared["text"] = prepared["text"].fillna("").astype(str).str.strip()
    for col in ["platform", "brand", "category"]:
        prepared[col] = prepared[col].fillna(defaults[col]).astype(str).str.strip()
        prepared.loc[prepared[col] == "", col] = defaults[col]
    prepared = prepared.dropna(subset=["date"])
    prepared = prepared[prepared["text"] != ""]
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
    if phrase in GENERIC_STANDALONE_TERMS:
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


def classify_trend_label(freq_recent: int, freq_prev: int) -> str:
    if freq_prev == 0 and freq_recent > 0:
        return "新出现"
    if freq_recent > freq_prev:
        return "增长"
    if freq_recent < freq_prev:
        return "衰减"
    return "稳定"


def calculate_dynamic_frequency_cutoff(recent_frequencies: list[int], min_freq: int) -> int:
    """用当前候选池的 P75 计算高频线，避免跨数据集硬编码绝对频次。"""
    if not recent_frequencies:
        return max(int(min_freq), 1)
    percentile_cutoff = int(np.ceil(np.quantile(recent_frequencies, 0.75)))
    return max(int(min_freq), percentile_cutoff, 1)


def build_ngram_statistics(texts: list[str], n_values: tuple[int, ...]) -> tuple[dict, dict, Counter, int]:
    ngram_counters = {n: Counter() for n in n_values}
    ngram_totals = {n: 0 for n in n_values}
    char_counter = Counter()
    total_chars = 0
    for text in texts:
        cleaned = clean_text(text)
        char_counter.update(cleaned)
        total_chars += len(cleaned)
        for n in n_values:
            windows = max(len(cleaned) - n + 1, 0)
            ngram_counters[n].update(cleaned[i : i + n] for i in range(windows))
            ngram_totals[n] += windows
    return ngram_counters, ngram_totals, char_counter, total_chars


def calculate_pmi_score(
    phrase: str,
    ngram_counters: dict,
    ngram_totals: dict,
    char_counter: Counter,
    total_chars: int,
) -> float:
    """计算严格的字符序列 PMI，而不是使用短语频次/总字符数的近似值。"""
    n = len(phrase)
    phrase_count = ngram_counters.get(n, {}).get(phrase, 0)
    total_ngrams = ngram_totals.get(n, 0)
    if not phrase_count or not total_ngrams or not total_chars:
        return 0.0

    log_probability = np.log2(phrase_count / total_ngrams)
    for char in phrase:
        char_count = char_counter.get(char, 0)
        if not char_count:
            return 0.0
        log_probability -= np.log2(char_count / total_chars)
    return round(float(log_probability), 3)


def collect_boundary_contexts(
    texts: list[str], candidate_phrases: set[str], n_values: tuple[int, ...]
) -> tuple[dict, dict]:
    left_contexts = defaultdict(Counter)
    right_contexts = defaultdict(Counter)
    if not candidate_phrases:
        return left_contexts, right_contexts

    for text in texts:
        cleaned = clean_text(text)
        for n in n_values:
            for index in range(max(len(cleaned) - n + 1, 0)):
                phrase = cleaned[index : index + n]
                if phrase not in candidate_phrases:
                    continue
                left = cleaned[index - 1] if index > 0 else "<START>"
                right_index = index + n
                right = cleaned[right_index] if right_index < len(cleaned) else "<END>"
                left_contexts[phrase][left] += 1
                right_contexts[phrase][right] += 1
    return left_contexts, right_contexts


def shannon_entropy(context_counts: Counter) -> float:
    total = sum(context_counts.values())
    if not total:
        return 0.0
    probabilities = [count / total for count in context_counts.values()]
    entropy = -sum(probability * np.log2(probability) for probability in probabilities)
    return round(max(float(entropy), 0.0), 3)


def classify_expression_quality(
    pmi_score: float,
    left_entropy: float,
    right_entropy: float,
    content_support: int,
) -> tuple[str, str]:
    reasons = []
    if content_support < 2:
        reasons.append("只出现在少量独立内容中")
    if min(left_entropy, right_entropy) <= 0.1:
        reasons.append("左右边界较固定，可能是更长表达中的片段")
    if pmi_score <= 0:
        reasons.append("非随机共现证据不足")

    if reasons:
        return "需复查", "；".join(reasons)
    return "自然", "PMI、边界变化和跨内容支持度基本满足要求"


def calculate_hotness_score(
    freq_recent: int,
    growth_rate: float,
    platform_coverage: int,
    platform_data_available: bool,
    engagement_score: float,
    high_freq_cutoff: int,
    growth_threshold: float,
) -> float:
    """只衡量热点趋势，不混入美妆适配和语义解释。"""
    frequency_scale = max(float(high_freq_cutoff) * 1.5, 1.0)
    frequency_component = min(max(freq_recent, 0) / frequency_scale, 1.0) * 40
    growth_scale = max(float(growth_threshold), 0.5) * 2
    growth_component = min(max(growth_rate, 0.0) / growth_scale, 1.0) * 30
    if platform_data_available:
        platform_component = min(max(platform_coverage, 0) / 3, 1.0) * 20
    else:
        platform_component = 10.0
    engagement_component = min(max(float(engagement_score), 0.0) / 40, 1.0) * 10
    return round(
        float(
            min(
                100,
                frequency_component
                + growth_component
                + platform_component
                + engagement_component,
            )
        ),
        2,
    )


def classify_hotness_level(hotness_score: float, trend_label: str, lifecycle: str) -> str:
    """把连续热度趋势分数转成前端使用的高/中/低。"""
    if lifecycle == "爆发中" or (
        hotness_score >= 70 and trend_label in {"增长", "新出现"}
    ):
        return "高"
    if lifecycle == "衰减期" and hotness_score < 70:
        return "低"
    if hotness_score >= 45 or lifecycle == "潜力期":
        return "中"
    return "低"


def calculate_beauty_usability_score(
    content_signal_score: float,
    category_values: list[str],
    rag_matches: list[dict],
    expression_quality: str,
    brand_profile: Optional[dict] = None,
) -> float:
    """在热点统计之上，补充美妆行业适配和表达质量，不把它当作热度分。"""
    brand_profile = brand_profile or DEFAULT_BRAND_PROFILE
    beauty_categories = {
        "美妆",
        "护肤",
        "彩妆",
        "清洁",
        "香氛",
        "个护身体/洗护",
        "个护",
    }
    normalized_categories = {str(value).strip() for value in category_values if str(value).strip()}
    target_category = str(brand_profile.get("category", "")).strip()
    if target_category and target_category in normalized_categories:
        category_fit_score = 20.0
    elif normalized_categories & beauty_categories:
        category_fit_score = 18.0
    else:
        # 跨行业热点保留迁移空间，但不能因为没有美妆分类就直接判为不可用。
        category_fit_score = 10.0

    direct_match_count = sum(int(item.get("candidate_match_count", 0)) for item in rag_matches)
    background_match_count = sum(
        1
        for item in rag_matches
        if item.get("candidate_match_count", 0) == 0
    )
    rag_fit_score = min(direct_match_count / 3, 1.0) * 10
    if direct_match_count == 0 and background_match_count > 0:
        rag_fit_score = 4.0
    expression_score = 10.0 if expression_quality == "自然" else 0.0
    score = float(content_signal_score) * 0.65 + category_fit_score + rag_fit_score + expression_score
    return round(min(max(score, 0.0), 100.0), 2)


def extract_evidence_examples(phrase: str, texts: list[str], limit: int = 3) -> list[str]:
    """从原始文本中提取短语所在的上下文，供人工复查和大模型参考。"""
    if not clean_text(phrase):
        return []
    examples = []
    seen = set()
    for text in texts:
        original = re.sub(r"\s+", " ", str(text)).strip()
        if not original or phrase not in clean_text(original):
            continue

        exact_index = original.find(phrase)
        if exact_index >= 0:
            window = 22
            start = max(0, exact_index - window)
            end = min(len(original), exact_index + len(phrase) + window)
            snippet = original[start:end]
            if start > 0:
                snippet = "…" + snippet
            if end < len(original):
                snippet = snippet + "…"
        else:
            # 原文中可能夹有标点或空格，无法直接定位时保留一段原文，避免误造连续表达。
            snippet = original if len(original) <= 90 else original[:90].rstrip() + "…"

        if snippet not in seen:
            examples.append(snippet)
            seen.add(snippet)
        if len(examples) >= limit:
            break
    return examples


def format_evidence_examples(examples: list[str]) -> str:
    if not examples:
        return "暂无原始语境证据"
    return "\n".join(f"- {example}" for example in examples)


@st.cache_data(show_spinner=False)
def build_candidates(
    df: pd.DataFrame,
    min_freq: int,
    min_brand_coverage: int,
    growth_threshold: float,
    n_values=(2, 3),
    brand_profile: Optional[dict] = None,
    product_context: Optional[str] = None,
) -> pd.DataFrame:
    working = df.copy()
    working["clean_text"] = working["text"].map(clean_text)
    max_date = working["date"].max()
    recent_start = max_date - timedelta(days=6)
    prev_start = max_date - timedelta(days=13)

    recent_df = working[working["date"] >= recent_start]
    prev_df = working[(working["date"] >= prev_start) & (working["date"] < recent_start)]

    phrase_rows = []
    ngram_counters, ngram_totals, char_counter, total_chars = build_ngram_statistics(
        working["text"].tolist(), n_values
    )

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
                    "source_text": row.text,
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

    brand_data_available = df["brand"].ne("未知品牌").any()
    platform_data_available = df["platform"].ne("未知平台").any()
    eligible_phrases = []
    for phrase, total_freq in all_counter.items():
        freq_recent = recent_counter.get(phrase, 0)
        freq_prev = prev_counter.get(phrase, 0)
        if freq_recent < min_freq:
            continue

        subset = phrase_df[phrase_df["phrase"] == phrase]
        brand_coverage = (
            subset.loc[subset["brand"] != "未知品牌", "brand"].nunique() if brand_data_available else 0
        )
        platform_coverage = (
            subset.loc[subset["platform"] != "未知平台", "platform"].nunique() if platform_data_available else 0
        )
        if brand_data_available and brand_coverage < min_brand_coverage:
            continue
        eligible_phrases.append(phrase)

    left_contexts, right_contexts = collect_boundary_contexts(
        working["clean_text"].tolist(), set(eligible_phrases), n_values
    )
    high_freq_cutoff = calculate_dynamic_frequency_cutoff(
        [recent_counter[phrase] for phrase in eligible_phrases], min_freq
    )

    rows = []
    for phrase in eligible_phrases:
        total_freq = all_counter[phrase]
        freq_recent = recent_counter.get(phrase, 0)
        freq_prev = prev_counter.get(phrase, 0)
        subset = phrase_df[phrase_df["phrase"] == phrase]
        brand_coverage = (
            subset.loc[subset["brand"] != "未知品牌", "brand"].nunique() if brand_data_available else 0
        )
        platform_coverage = (
            subset.loc[subset["platform"] != "未知平台", "platform"].nunique() if platform_data_available else 0
        )

        growth_rate = normalized_growth(freq_recent, freq_prev)
        trend_label = classify_trend_label(freq_recent, freq_prev)
        source_texts = subset["source_text"].dropna().astype(str).tolist()
        evidence_examples = extract_evidence_examples(phrase, source_texts)
        evidence_context = format_evidence_examples(evidence_examples)
        rag_matches = retrieve_lexicon_matches(
            phrase,
            source_texts=source_texts,
            product_context=product_context,
        )
        rag_context = format_rag_context(rag_matches)
        rag_confidence = classify_rag_confidence(rag_matches)
        engagement_raw = (
            subset["likes"].sum() * 1.0 + subset["comments"].sum() * 2.0 + subset["saves"].sum() * 1.5
        ) / max(total_freq, 1)
        pmi_score = calculate_pmi_score(phrase, ngram_counters, ngram_totals, char_counter, total_chars)
        left_entropy = shannon_entropy(left_contexts[phrase])
        right_entropy = shannon_entropy(right_contexts[phrase])
        content_support = int(subset["source_text"].nunique())
        expression_quality, review_reason = classify_expression_quality(
            pmi_score,
            left_entropy,
            right_entropy,
            content_support,
        )
        support_factor = min(np.log1p(content_support) / np.log1p(5), 1.0)
        boundary_factor = 1 - np.exp(-max(min(left_entropy, right_entropy), 0.0))
        phrase_quality_score = round(float(max(pmi_score, 0) * support_factor * boundary_factor), 3)
        coverage_score = min(brand_coverage / 4, 1) * 15 + min(platform_coverage / 3, 1) * 10
        growth_cap = growth_threshold * 4 if growth_threshold > 0 else 4
        growth_score = min(max(growth_rate, 0), growth_cap) * 10
        engagement_score = min(np.log1p(engagement_raw) * 6, 40)
        pmi_component = min(max(phrase_quality_score, 0) * 3, 25)
        content_signal_score = round(
            float(min(100, coverage_score + growth_score + engagement_score + pmi_component)),
            2,
        )
        beauty_usability_score = calculate_beauty_usability_score(
            content_signal_score,
            subset["category"].tolist(),
            rag_matches,
            expression_quality,
            brand_profile=brand_profile,
        )
        hotness_score = calculate_hotness_score(
            freq_recent,
            growth_rate,
            platform_coverage,
            platform_data_available,
            engagement_score,
            high_freq_cutoff,
            growth_threshold,
        )

        lifecycle = classify_lifecycle(
            freq_recent,
            freq_prev,
            growth_rate,
            brand_coverage,
            growth_threshold,
            high_freq_cutoff=high_freq_cutoff,
            content_support=content_support,
        )
        hotness_level = classify_hotness_level(hotness_score, trend_label, lifecycle)
        risk = make_risk_tip(
            phrase,
            lifecycle,
            brand_coverage,
            platform_coverage,
            rag_confidence,
            brand_coverage_known=brand_data_available,
            platform_coverage_known=platform_data_available,
            expression_quality=expression_quality,
            review_reason=review_reason,
            trend_label=trend_label,
            rag_lexicon_ids={
                item["lexicon_id"]
                for item in rag_matches
                if item.get("candidate_match_count", 0) > 0
            },
        )

        rows.append(
            {
                "phrase": phrase,
                "freq_recent": freq_recent,
                "freq_prev": freq_prev,
                "growth_rate": round(growth_rate, 3),
                "trend_label": trend_label,
                "hotness_score": hotness_score,
                "hotness_level": hotness_level,
                "brand_coverage": brand_coverage,
                "platform_coverage": platform_coverage,
                "pmi_score": pmi_score,
                "left_entropy": left_entropy,
                "right_entropy": right_entropy,
                "phrase_quality_score": phrase_quality_score,
                "content_support": content_support,
                "expression_quality": expression_quality,
                "review_reason": review_reason,
                "engagement_score": round(float(engagement_score), 2),
                "beauty_usability_score": beauty_usability_score,
                # 保留旧字段，避免已有导出文件和外部引用失效。
                "usability_score": beauty_usability_score,
                "lifecycle": lifecycle,
                "risk_tip": risk,
                "high_freq_cutoff": high_freq_cutoff,
                "rag_confidence": rag_confidence,
                "llm_prompt": generate_llm_prompt(
                    {
                        "phrase": phrase,
                        "freq_recent": freq_recent,
                        "freq_prev": freq_prev,
                        "growth_rate": round(growth_rate, 3),
                        "trend_label": trend_label,
                        "hotness_score": hotness_score,
                        "hotness_level": hotness_level,
                        "brand_coverage": brand_coverage,
                        "platform_coverage": platform_coverage,
                        "lifecycle": lifecycle,
                        "high_freq_cutoff": high_freq_cutoff,
                        "risk_tip": risk,
                        "beauty_usability_score": beauty_usability_score,
                        "usability_score": beauty_usability_score,
                        "pmi_score": pmi_score,
                        "left_entropy": left_entropy,
                        "right_entropy": right_entropy,
                        "phrase_quality_score": phrase_quality_score,
                        "expression_quality": expression_quality,
                        "review_reason": review_reason,
                        "evidence_context": evidence_context,
                        "engagement_score": round(float(engagement_score), 2),
                        "rag_confidence": rag_confidence,
                        "rag_context": rag_context,
                    },
                    brand_profile=brand_profile,
                ),
                "rag_match_count": len(rag_matches),
                "rag_context": rag_context,
                "evidence_context": evidence_context,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        ["beauty_usability_score", "hotness_score", "freq_recent"],
        ascending=False,
    ).reset_index(drop=True)


def classify_lifecycle(
    freq_recent: int,
    freq_prev: int,
    growth_rate: float,
    brand_coverage: int,
    growth_threshold: float = 0.5,
    high_freq_cutoff: Optional[int] = None,
    content_support: int = 0,
) -> str:
    threshold = max(float(growth_threshold), 0.1)
    if freq_recent < freq_prev:
        return "衰减期"
    frequency_line = max(int(high_freq_cutoff or 1), 1)
    has_positive_growth = growth_rate > 0 and freq_recent > freq_prev
    if (
        freq_recent >= frequency_line
        and has_positive_growth
        and growth_rate >= threshold
        and content_support >= 2
    ):
        return "爆发中"
    if has_positive_growth and growth_rate >= threshold and brand_coverage >= 2:
        return "潜力期"
    return "低热观察"


def make_risk_tip(
    phrase: str,
    lifecycle: str,
    brand_coverage: int,
    platform_coverage: int,
    rag_confidence: str = "中信度",
    brand_coverage_known: bool = True,
    platform_coverage_known: bool = True,
    expression_quality: str = "自然",
    review_reason: str = "",
    trend_label: str = "",
    rag_lexicon_ids: Optional[set[str]] = None,
) -> str:
    risks = []
    rag_lexicon_ids = rag_lexicon_ids or set()
    if rag_confidence == "低信度":
        risks.append(LOW_CONFIDENCE_NOTICE)
    if "efficacy" in rag_lexicon_ids:
        risks.append("命中功效词库：仅作内容方向，需核验产品证据与合规边界")
    if trend_label == "新出现":
        risks.append("新出现：前7天无记录，需观察是否持续")
    if lifecycle == "衰减期":
        risks.append("热度可能已经衰减")
    if len(phrase) <= 2:
        risks.append("语义不完整，需人工复核")
    if expression_quality != "自然":
        risks.append(f"{expression_quality}：{review_reason or '建议结合原始语境人工复查'}")
    if not brand_coverage_known:
        risks.append("品牌覆盖数据缺失，需人工复核")
    elif brand_coverage <= 1:
        risks.append("可能与品牌调性不匹配")
    if not platform_coverage_known:
        risks.append("平台覆盖数据缺失，需人工复核")
    elif platform_coverage <= 1:
        risks.append("跨平台扩散不足")
    if any(word in phrase for word in ["笑", "疯", "躺", "摆"]):
        risks.append("可能过度娱乐化")
    return "；".join(risks) if risks else "风险较低，但仍需人工复核语境"


def build_brand_strategy_context(brand_profile: Optional[dict]) -> dict:
    profile = brand_profile or DEFAULT_BRAND_PROFILE
    industry = profile.get("industry", "美妆")
    goals = [goal.strip() for goal in profile.get("goal", "提升互动/种草").split("、") if goal.strip()]
    goal_focuses = [GOAL_GUIDANCE.get(goal) for goal in goals if GOAL_GUIDANCE.get(goal)]
    return {
        "fit_focus": INDUSTRY_GUIDANCE.get(
            industry,
            "请结合用户填写的美妆品类、功效目标和品牌补充信息，判断热点是否适合真实内容场景。",
        ),
        "goal_focus": "；".join(goal_focuses) or "请结合用户填写的营销目标，给出可执行且可衡量的行动建议。",
    }


def build_product_context(brand_profile: Optional[dict]) -> str:
    """把用户填写的产品事实整理成 RAG 与 Prompt 共用的上下文。"""
    profile = brand_profile or DEFAULT_BRAND_PROFILE
    product_fields = [
        ("产品名称", "product_name", "未填写具体产品"),
        ("产品定位与核心卖点", "product_positioning", "未补充产品定位和核心卖点"),
        ("已确认产品证据", "product_evidence", "未补充已确认的成分、功效或使用证据"),
        ("禁用/规避表达", "prohibited_claims", "未补充禁用或规避表达"),
    ]
    provided_fields = []
    for label, key, placeholder in product_fields:
        value = str(profile.get(key, "")).strip()
        if value and value != placeholder:
            provided_fields.append(f"{label}：{value}")
    return "\n".join(provided_fields) or "未提供产品信息"


def select_with_other(container, label: str, options: list[str], default: int, key: str, placeholder: str) -> str:
    choice = container.selectbox(label, options, index=default, key=key)
    if choice == "其他":
        custom_value = container.text_input(f"{label}·自定义", placeholder=placeholder, key=f"{key}_custom")
        return custom_value.strip() or "其他"
    return choice


def multiselect_with_other(
    container,
    label: str,
    options: list[str],
    default: list[str],
    key: str,
    placeholder: str,
    max_selections: Optional[int] = None,
) -> list[str]:
    kwargs = {"default": default, "key": key}
    if max_selections is not None:
        kwargs["max_selections"] = max_selections
    values = container.multiselect(label, options, **kwargs)
    if "其他" in values:
        custom_value = container.text_input(f"{label}·自定义", placeholder=placeholder, key=f"{key}_custom")
        values = [value for value in values if value != "其他"]
        if custom_value.strip():
            values.append(custom_value.strip())
    return values


def generate_llm_prompt(row: dict, brand_profile: Optional[dict] = None) -> str:
    brand_profile = brand_profile or DEFAULT_BRAND_PROFILE
    strategy_context = build_brand_strategy_context(brand_profile)
    return f"""# 任务
请评估下面这个来自榜单的热点候选，判断它是否值得被当前美妆品牌借用，并给出可执行的内容行动建议。

这不是让你重新计算热点分数，也不是让你把词库内容当成事实。请严格区分：
- 统计证据：频次、增长、覆盖、互动、热度趋势、表达质量、美妆可用性和本地规则判断。
- RAG 参考：美妆词库检索到的可能相关条目，只用于辅助理解。
- 策略推断：你基于前两类信息和品牌画像做出的解释与建议。

# 一、当前品牌画像
- 行业：{brand_profile['industry']}
- 品牌层级：{brand_profile['brand_tier']}
- 品牌来源：{brand_profile['brand_origin']}
- 产品阶段：{brand_profile['product_stage']}
- 主推品类：{brand_profile['category']}
- 品牌调性：{brand_profile['tone']}
- 本次营销目标：{brand_profile['goal']}
- 目标需求类型：{brand_profile['audience_need']}
- 功效目标：{brand_profile['efficacy_goal']}
- 目标人群：{brand_profile['target_group']}
- 肤质：{brand_profile['skin_type']}
- 品牌补充信息：{brand_profile['notes']}
- 行业适配重点：{strategy_context['fit_focus']}
- 目标策略重点：{strategy_context['goal_focus']}

# 二、当前产品信息
- 产品名称：{brand_profile.get('product_name', '未填写具体产品')}
- 产品定位与核心卖点：{brand_profile.get('product_positioning', '未补充产品定位和核心卖点')}
- 已确认产品证据：{brand_profile.get('product_evidence', '未补充已确认的成分、功效或使用证据')}
- 禁用/规避表达：{brand_profile.get('prohibited_claims', '未补充禁用或规避表达')}
- 产品信息使用边界：以上内容只用于品牌适配、词库检索和建议生成；未提供证据的功效不能写成产品承诺。

# 三、统计证据
- 候选短语：{row['phrase']}
- 近 7 天频次：{row['freq_recent']}
- 前 7 天频次：{row['freq_prev']}
- 增长率：{row['growth_rate']}
- 趋势判断：{row.get('trend_label', '未提供')}
- 当前候选池动态爆发频次线：{row.get('high_freq_cutoff', '未提供')}（候选池近 7 天频次 P75）
- 品牌覆盖数：{row['brand_coverage']}
- 平台覆盖数：{row['platform_coverage']}
- 热度趋势：{row.get('hotness_level', '未提供')}（热度趋势分：{row.get('hotness_score', '未提供')}）
- 表达质量：{row.get('expression_quality', '未提供')}
- 美妆可用性：{row.get('beauty_usability_score', row.get('usability_score', '未提供'))} / 100
- PMI 非随机共现分数：{row.get('pmi_score', '未提供')}
- 左边界熵：{row.get('left_entropy', '未提供')}
- 右边界熵：{row.get('right_entropy', '未提供')}
- 短语质量分数：{row.get('phrase_quality_score', '未提供')}
- 复查原因：{row.get('review_reason', '无')}
- 原始语境证据：
{row.get('evidence_context', '暂无原始语境证据')}
- 互动强度：{row.get('engagement_score', '未提供')}
- 生命周期/趋势判断：{row['lifecycle']}
- 本地规则风险提示：{row['risk_tip']}

# 四、RAG 参考
- RAG 检索信度：{row.get('rag_confidence', '低信度')}
- 美妆词库检索结果：
{row.get('rag_context', '暂无匹配的美妆词库条目')}
- 阅读说明：命中范围为“候选短语”的词条/同义词关联更直接；仅“原始语境”或“标签”命中时，只能作为行业或场景背景，不能直接证明候选短语含义。

# 五、请按以下顺序完成判断
1. 语义解释：用普通中文解释这个候选短语可能表达什么；如果词义不完整或无法确定，明确说出不确定性。
2. 情绪倾向：说明它可能承载的情绪和使用场景，但不要把推断写成用户调研结论。
3. 美妆适配判断：结合品牌行业、品类、调性、目标人群和营销目标，说明适合迁移、不适合迁移，还是只能小范围测试。
4. 商业价值：解释它能连接哪类内容机会；如果无法自然连接产品或品牌，不要强行解释。
5. 内容行动：给出至少 2 条、最多 3 条建议。每条必须包含“内容形式、切入角度、品牌连接方式、观察指标”。
6. 风险提示：只写真实存在的风险。若 RAG 检索信度为低信度，必须保留“低信度，代人工核验”；若表达不自然或语义不完整，必须提示人工复查。
7. CTA 引导：给出一句与品牌调性和当前目标匹配的用户行动引导；如果不建议跟进，CTA 应改为验证性动作或不输出强转化引导。

判断表达质量时，请优先结合“原始语境证据”判断。PMI 和边界熵只是统计信号，不足以单独证明一个中文短语自然；如果原始语境仍无法解释词义，请明确保留不确定性，并建议人工核验。

# 六、输出要求
- 输出面向品牌运营团队，使用清晰、具体、少空话的中文。
- 不得声称这些数据来自真实平台、真实品牌或真实用户调研。
- 不得编造搜索量、用户画像、市场趋势、产品成分和功效证据。
- 词库条目只能作为辅助参考，不得直接当作产品功效或用户事实。
- 如果热点显著性、语义或品牌适配证据不足，宁可给出“暂不建议跟进”或“先小范围验证”。
- 最终严格按接口要求的 JSON 结构输出。"""


def build_markdown_report(candidates: pd.DataFrame, brand_profile: Optional[dict] = None) -> str:
    brand_profile = brand_profile or DEFAULT_BRAND_PROFILE
    top = candidates.head(10)
    lines = [
        "# AI 辅助热点洞察与趋势决策报告",
        "",
        "说明：本报告基于本地模拟数据生成，不连接真实平台，不包含公司内部数据。",
        "",
        "## 本次分析品牌画像",
        "",
        f"- 品牌行业：{brand_profile['industry']}",
        f"- 品牌层级：{brand_profile['brand_tier']}",
        f"- 品牌来源：{brand_profile['brand_origin']}",
        f"- 产品阶段：{brand_profile['product_stage']}",
        f"- 主推品类：{brand_profile['category']}",
        f"- 品牌调性：{brand_profile['tone']}",
        f"- 本次目标：{brand_profile['goal']}",
        f"- 目标需求类型：{brand_profile['audience_need']}",
        f"- 功效目标：{brand_profile['efficacy_goal']}",
        f"- 目标人群：{brand_profile['target_group']}",
        f"- 肤质：{brand_profile['skin_type']}",
        f"- 品牌补充信息：{brand_profile['notes']}",
        f"- 产品名称：{brand_profile.get('product_name', '未填写具体产品')}",
        f"- 产品定位与核心卖点：{brand_profile.get('product_positioning', '未补充产品定位和核心卖点')}",
        f"- 已确认产品证据：{brand_profile.get('product_evidence', '未补充已确认的成分、功效或使用证据')}",
        f"- 禁用/规避表达：{brand_profile.get('prohibited_claims', '未补充禁用或规避表达')}",
        "",
        "## Top 10 高可用性热点",
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
                f"- 趋势判断：{row.trend_label}",
                f"- 热度趋势：{row.hotness_level}（热度趋势分：{row.hotness_score}）",
                f"- 动态爆发频次线：{row.high_freq_cutoff}（候选池近 7 天频次 P75）",
                f"- 品牌覆盖：{row.brand_coverage}",
                f"- 平台覆盖：{row.platform_coverage}",
                f"- 独立内容支持数：{row.content_support}",
                f"- 生命周期：{row.lifecycle}",
                f"- 表达质量：{row.expression_quality}",
                f"- 美妆可用性：{row.beauty_usability_score} / 100",
                f"- PMI 非随机共现分数：{row.pmi_score}",
                f"- 左/右边界熵：{row.left_entropy} / {row.right_entropy}",
                f"- 复查原因：{row.review_reason}",
                "- 原始语境证据：",
                row.evidence_context,
                f"- 风险提示：{row.risk_tip}",
                f"- RAG置信度：{row.rag_confidence}",
                "- 词库检索及原始证据：",
                row.rag_context,
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
        .chart-shell {
            background: #FFFFFF;
            border: 1px solid #E7E9F2;
            border-radius: 16px;
            padding: 18px 18px 14px;
            margin-top: 8px;
        }
        .chart-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
            color: #697386;
            font-size: 0.78rem;
        }
        .chart-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .chart-legend-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }
        .chart-rows {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .chart-row {
            display: grid;
            grid-template-columns: 72px minmax(0, 1fr) 48px;
            gap: 10px;
            align-items: center;
        }
        .chart-row-label {
            color: #252A3A;
            font-size: 0.9rem;
            text-align: right;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .chart-row-track {
            height: 16px;
            border-radius: 999px;
            background: repeating-linear-gradient(
                to right,
                #F1F3F8 0,
                #F1F3F8 calc(20% - 1px),
                #DDE1EA calc(20% - 1px),
                #DDE1EA 20%
            );
            overflow: hidden;
        }
        .chart-row-fill {
            height: 100%;
            border-radius: 999px;
        }
        .chart-row-value {
            color: #252A3A;
            font-size: 0.88rem;
            font-variant-numeric: tabular-nums;
        }
        .chart-scale {
            display: grid;
            grid-template-columns: 72px minmax(0, 1fr) 48px;
            gap: 10px;
            margin-top: 8px;
            color: #9AA4B2;
            font-size: 0.72rem;
        }
        .chart-scale-values {
            display: flex;
            justify-content: space-between;
        }
        .distribution-row {
            display: grid;
            grid-template-columns: 64px minmax(0, 1fr) 36px;
            gap: 10px;
            align-items: center;
        }
        @media (max-width: 700px) {
            .chart-row {
                grid-template-columns: 56px minmax(0, 1fr) 42px;
                gap: 8px;
            }
            .chart-scale {
                grid-template-columns: 56px minmax(0, 1fr) 42px;
                gap: 8px;
            }
            .distribution-row {
                grid-template-columns: 54px minmax(0, 1fr) 32px;
                gap: 8px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_bar_chart(top_candidates: pd.DataFrame) -> None:
    if top_candidates.empty:
        st.info("暂无可展示的候选热梗。")
        return

    chart_data = top_candidates.sort_values("beauty_usability_score", ascending=True)
    legend = "".join(
        f'<span class="chart-legend-item"><span class="chart-legend-dot" style="background:{HOTNESS_COLORS[level]}"></span>热度趋势：{escape(level)}</span>'
        for level in ("高", "中", "低")
    )
    rows = []
    for row in chart_data.itertuples(index=False):
        value = float(row.beauty_usability_score)
        width = max(0, min(value, 100))
        color = HOTNESS_COLORS.get(row.hotness_level, "#9AA4B2")
        rows.append(
            f'<div class="chart-row"><div class="chart-row-label" title="{escape(str(row.phrase))}">{escape(str(row.phrase))}</div>'
            f'<div class="chart-row-track"><div class="chart-row-fill" style="width:{width:.2f}%;background:{color};"></div></div>'
            f'<div class="chart-row-value">{value:.1f}</div></div>'
        )
    st.markdown(
        f'<div class="chart-shell"><div class="chart-legend">{legend}</div>'
        f'<div class="chart-rows">{"".join(rows)}</div>'
        '<div class="chart-scale"><span></span>'
        '<span class="chart-scale-values"><span>0</span><span>20</span><span>40</span><span>60</span><span>80</span><span>100</span></span>'
        '<span></span></div></div>',
        unsafe_allow_html=True,
    )


def render_lifecycle_distribution(candidates: pd.DataFrame) -> None:
    counts = candidates["lifecycle"].value_counts().reindex(LIFECYCLE_ORDER, fill_value=0)
    max_count = max(int(counts.max()), 1)
    rows = []
    for lifecycle in LIFECYCLE_ORDER:
        value = int(counts[lifecycle])
        width = value / max_count * 100
        color = LIFECYCLE_COLORS[lifecycle]
        rows.append(
            f'<div class="distribution-row"><div class="chart-row-label">{escape(lifecycle)}</div>'
            f'<div class="chart-row-track"><div class="chart-row-fill" style="width:{width:.2f}%;background:{color};"></div></div>'
            f'<div class="chart-row-value">{value}</div></div>'
        )
    scale_max = max_count if max_count <= 5 else int(np.ceil(max_count / 5) * 5)
    st.markdown(
        f'<div class="chart-shell"><div class="chart-rows">{"".join(rows)}</div>'
        f'<div class="chart-scale"><span></span><span class="chart-scale-values"><span>0</span>'
        f'<span>{scale_max // 4}</span><span>{scale_max // 2}</span><span>{scale_max * 3 // 4}</span><span>{scale_max}</span></span>'
        '<span></span></div></div>',
        unsafe_allow_html=True,
    )


def render_scoring_explanation() -> None:
    with st.expander("查看热度与美妆可用性计算口径", expanded=False):
        st.caption(
            "热度回答“这个表达是否正在升温”，美妆可用性回答“它是否适合当前美妆品牌和产品使用”。"
            "PMI、左右边界熵和原始语境用于验证短语是否自然、完整，不直接计入热度趋势分。"
        )
        heat_col, usability_col = st.columns(2)
        with heat_col:
            st.markdown("#### 热度趋势分（0—100）")
            st.markdown(
                "- **近 7 天频次 40%**：min(近7天频次 / (P75 × 1.5), 1) × 40。\n"
                "- **增长率 30%**：min(max(增长率, 0) / (max(增长阈值, 0.5) × 2), 1) × 30。\n"
                "- **平台覆盖 20%**：min(平台数 / 3, 1) × 20；缺少平台字段时固定记 10 分。\n"
                "- **互动强度 10%**：min(互动强度 / 40, 1) × 10；互动强度由点赞、评论、收藏汇总。"
            )
            st.caption("热度趋势高/中/低，是上述分数与当前趋势、生命周期共同得出的快捷标签。")
        with usability_col:
            st.markdown("#### 美妆可用性（0—100）")
            st.markdown(
                "- **内容信号 65%**：(品牌覆盖≤15 + 平台覆盖≤10 + 增长≤10 + 互动≤40 + 短语质量≤25) × 0.65。\n"
                "- **美妆品类匹配 10—20 分**：命中当前主推品类记 20；命中其他美妆品类记 18；跨行业热点保留 10 分迁移空间。\n"
                "- **RAG 词库关联 0—10 分**：候选短语/同义词直接命中按最多 3 条计满 10 分；只有语境或标签背景命中时记 4 分。\n"
                "- **表达质量 +10 分**：自然表达加分；需复查不加分，并在风险提示中标记。"
            )
            st.caption("RAG 只帮助判断行业与产品相关性，不会证明一个词本身“热”。")
        st.markdown("#### 生命周期怎么判")
        st.markdown(
            "系统在通过最低频次、品牌覆盖与降噪校验的候选池中，计算近 7 天频次的 **P75** 作为动态高频线。"
            "频次达到 P75、增长率达到当前阈值、且至少有 2 条独立内容支持时为“爆发中”；"
            "增长达标但未达到动态高频线且品牌覆盖不少于 2 时为“潜力期”；"
            "近 7 天少于前 7 天为“衰减期”；其余为“低热观察”。"
        )


def main() -> None:
    st.set_page_config(page_title="AI 辅助热点洞察与趋势决策工作流", layout="wide")
    apply_ui_styles()
    st.title("AI 辅助热点洞察与趋势决策工作流")

    with st.expander("项目说明", expanded=True):
        st.markdown(
            """
            这是一个面向美妆内容运营的 AI 热点洞察 Agent：把社交媒体内容输入，自动清洗、提取候选短语、计算增长与覆盖指标，输出热度趋势、生命周期、表达质量与美妆可用性，并结合品牌与产品上下文、词库 RAG 和可选大模型生成语义解释和内容建议。

            **搭建原理：** 本地规则负责数据处理、热梗提取、热度与可用性计算；大模型负责理解热梗语义，并把分析结果转成运营团队可以直接阅读和执行的建议。

            **使用场景：** 内容运营做日常热点监测、品牌 campaign 选题、社交媒体趋势复盘，以及从热点发现到内容测试的快速决策。

            **可以解决的问题：** 热点信息分散、人工筛选耗时、判断口径不一致，以及发现热点后难以继续转化为具体内容动作。

            当前 Demo 使用脱敏模拟数据；配置 API Key 后，可以在选择具体热梗时调用大模型生成分析。
            """
        )

    uploaded_file = st.sidebar.file_uploader("上传自己的 CSV", type=["csv"])
    with st.sidebar.expander("分析参数", expanded=True):
        min_freq = st.slider(
            "min_freq：近 7 天最低频次",
            min_value=1,
            max_value=20,
            value=3,
            help="只有近 7 天出现次数达到这个数的短语，才进入候选池。",
        )
        min_brand_coverage = st.slider(
            "min_brand_coverage：最低品牌覆盖",
            1,
            5,
            2,
            help="至少被多少个品牌的内容提及，才认为具有跨品牌参考价值。",
        )
        growth_threshold = st.slider(
            "growth_threshold：增长判断阈值",
            0.0,
            3.0,
            0.5,
            0.1,
            format="%.2f",
            help="增长率=(近7天频次-前7天频次)/前7天频次。0.50表示至少增长50%。前7天为0时，增长率按近7天频次计算。",
        )
        ngram_mode = st.radio("候选短语长度", ["2-gram + 3-gram", "仅 2-gram", "仅 3-gram"], index=0)
        st.caption("增长阈值越高，只有增长更明显的词条才会被标记为潜力或爆发。")

    n_values = {"2-gram + 3-gram": (2, 3), "仅 2-gram": (2,), "仅 3-gram": (3,)}[ngram_mode]

    try:
        raw_df = load_raw_data(uploaded_file)
        default_mapping, default_methods = infer_column_mapping(raw_df)
        with st.expander("CSV 字段识别与清洗", expanded=uploaded_file is not None):
            mapping, mapping_methods = render_column_mapping(raw_df, default_mapping, default_methods)
            for warning in mapping_warnings(mapping):
                st.warning(warning)
        df = validate_and_prepare(raw_df, mapping=mapping)
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

    st.subheader("品牌上下文")
    st.caption("输入数据可以来自跨行业榜单；这里用美妆品牌视角判断热点是否值得迁移和使用。")
    with st.expander("品牌上下文 · 美妆行业", expanded=True):
        context_left, context_right = st.columns(2)
        with context_left:
            brand_tier = select_with_other(
                context_left,
                "品牌层级",
                BEAUTY_BRAND_TIERS,
                default=2,
                key="brand_tier",
                placeholder="例如：专业线、平价国货",
            )
            brand_origin = select_with_other(
                context_left,
                "品牌来源",
                BEAUTY_BRAND_ORIGINS,
                default=1,
                key="brand_origin",
                placeholder="例如：新锐国货",
            )
            product_stage = select_with_other(
                context_left,
                "产品阶段",
                BEAUTY_PRODUCT_STAGES,
                default=2,
                key="product_stage",
                placeholder="例如：礼赠节点主推",
            )
            beauty_category = select_with_other(
                context_left,
                "主推品类",
                BEAUTY_CATEGORIES,
                default=1,
                key="beauty_category",
                placeholder="例如：美容仪",
            )
            audience_need = select_with_other(
                context_left,
                "目标需求类型",
                NEED_ARCHETYPES,
                default=0,
                key="audience_need",
                placeholder="例如：成分党尝鲜",
            )
        with context_right:
            tone_values = multiselect_with_other(
                context_right,
                "品牌调性（可多选）",
                BEAUTY_TONES,
                default=["年轻活泼"],
                key="beauty_tones",
                placeholder="例如：松弛、极简",
                max_selections=3,
            )
            goal_values = multiselect_with_other(
                context_right,
                "营销目标（最多3项）",
                CAMPAIGN_GOALS,
                default=["提升互动/种草"],
                key="campaign_goals",
                placeholder="例如：新品种草、口碑维护",
                max_selections=3,
            )
            efficacy_values = multiselect_with_other(
                context_right,
                "功效目标（最多3项）",
                EFFICACY_GOALS,
                default=["修护"],
                key="efficacy_goals",
                placeholder="例如：妆效持久",
                max_selections=3,
            )
            target_group_values = multiselect_with_other(
                context_right,
                "目标人群（最多2项）",
                TARGET_GROUPS,
                default=["职场"],
                key="target_groups",
                placeholder="例如：轻熟龄女性",
                max_selections=2,
            )
            skin_type_values = multiselect_with_other(
                context_right,
                "肤质（最多2项）",
                SKIN_TYPES,
                default=["未知"],
                key="skin_types",
                placeholder="例如：混合偏油",
                max_selections=2,
            )
            brand_notes = context_right.text_area(
                "品牌补充信息（可选）",
                placeholder="例如：主要面向学生群体，强调性价比；避免过度夸张表达。",
                height=90,
                key="brand_notes",
            ).strip() or "未补充其他品牌信息"

    brand_profile = {
        "industry": "美妆",
        "brand_tier": brand_tier,
        "brand_origin": brand_origin,
        "product_stage": product_stage,
        "category": beauty_category,
        "tone": "、".join(tone_values) or "未指定",
        "goal": "、".join(goal_values) or "未指定",
        "audience_need": audience_need,
        "efficacy_goal": "、".join(efficacy_values) or "未指定",
        "target_group": "、".join(target_group_values) or "未指定",
        "skin_type": "、".join(skin_type_values) or "未指定",
        "notes": brand_notes,
    }
    st.caption(
        f"当前分析画像：{brand_profile['industry']} · {brand_profile['category']} · "
        f"{brand_profile['tone']} · 功效目标：{brand_profile['efficacy_goal']} · "
        f"目标为{brand_profile['goal']}"
    )

    st.subheader("产品信息")
    st.caption("可选但建议填写。产品信息会用于美妆适配、词库检索和 Prompt；没有证据的功效不会被系统自动当成事实。")
    with st.expander("产品信息 · 用于当前品牌适配", expanded=False):
        product_left, product_right = st.columns(2)
        with product_left:
            product_name = product_left.text_input(
                "产品名称",
                placeholder="例如：积雪草舒缓修护精华",
                key="product_name",
            ).strip() or "未填写具体产品"
            product_positioning = product_left.text_area(
                "产品定位与核心卖点",
                placeholder="例如：面向敏感肌的轻薄修护精华；强调清爽、低负担。",
                height=110,
                key="product_positioning",
            ).strip() or "未补充产品定位和核心卖点"
        with product_right:
            product_evidence = product_right.text_area(
                "已确认的产品证据",
                placeholder="例如：已确认成分、测试结果、真实使用信息；没有就写“未补充”。",
                height=110,
                key="product_evidence",
            ).strip() or "未补充已确认的成分、功效或使用证据"
            prohibited_claims = product_right.text_area(
                "禁用/规避表达",
                placeholder="例如：不说“根治”“百分百修复”；避免未经证实的功效承诺。",
                height=110,
                key="prohibited_claims",
            ).strip() or "未补充禁用或规避表达"
    brand_profile.update(
        {
            "product_name": product_name,
            "product_positioning": product_positioning,
            "product_evidence": product_evidence,
            "prohibited_claims": prohibited_claims,
        }
    )
    product_context = build_product_context(brand_profile)

    st.subheader("候选词条结果表")
    candidates = build_candidates(
        df,
        min_freq,
        min_brand_coverage,
        growth_threshold,
        n_values=n_values,
        brand_profile=brand_profile,
        product_context=product_context,
    )
    if candidates.empty:
        st.warning("当前筛选条件下没有识别到候选热梗。建议先把 min_freq 调到 3、品牌覆盖调到 2，再逐步收紧条件。")
        return

    top_candidate = candidates.iloc[0]
    lifecycle_counts = candidates["lifecycle"].value_counts()
    st.info(
        f"当前识别出 {len(candidates)} 个候选短语。当前美妆可用性最高的是「{top_candidate['phrase']}」："
        f"热度趋势为「{top_candidate['hotness_level']}」，生命周期为「{top_candidate['lifecycle']}」，"
        f"表达质量为「{top_candidate['expression_quality']}」，美妆可用性 {top_candidate['beauty_usability_score']}。"
    )
    st.caption(
        f"本次动态爆发频次线：近 7 天频次 ≥ {int(top_candidate['high_freq_cutoff'])}。"
        f"该数值取当前候选池近 7 天频次的 P75，不是固定阈值；同时还需要增长率 ≥ {growth_threshold:.2f}。"
    )
    render_scoring_explanation()
    summary_a, summary_b, summary_c, summary_d, summary_e = st.columns(5)
    summary_a.metric("候选短语数", len(candidates))
    summary_b.metric("爆发中", int(lifecycle_counts.get("爆发中", 0)))
    summary_c.metric("潜力期", int(lifecycle_counts.get("潜力期", 0)))
    summary_d.metric("需复查", int((candidates["expression_quality"] == "需复查").sum()))
    summary_e.metric("平均美妆可用性", round(float(candidates["beauty_usability_score"].mean()), 1))

    compact_cols = [
        "phrase",
        "hotness_level",
        "expression_quality",
        "beauty_usability_score",
        "lifecycle",
        "growth_rate",
        "risk_tip",
    ]
    display_cols = [
        "phrase",
        "freq_recent",
        "freq_prev",
        "growth_rate",
        "trend_label",
        "hotness_score",
        "hotness_level",
        "high_freq_cutoff",
        "brand_coverage",
        "platform_coverage",
        "content_support",
        "pmi_score",
        "left_entropy",
        "right_entropy",
        "phrase_quality_score",
        "expression_quality",
        "review_reason",
        "engagement_score",
        "rag_confidence",
        "rag_match_count",
        "beauty_usability_score",
        "lifecycle",
        "risk_tip",
    ]
    st.caption("系统已过滤缺少语境的通用行业名词；默认展示最适合快速判断的关键指标，完整评分明细已收起。")
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
        st.subheader("Top 10 美妆可用性排序")
        st.caption("按美妆可用性排序；热度趋势和表达质量在表格中单独展示。")
        render_bar_chart(top10)
    with chart_right:
        st.subheader("生命周期分布")
        st.caption("看整体候选池中，词条处于爆发、潜力、观察或衰减阶段的数量。")
        render_lifecycle_distribution(candidates)

    st.subheader("选择热梗，查看详细结果")
    st.info("请选择一个候选热梗，下面的热度趋势、生命周期、表达质量、美妆可用性、风险提示和 AI Prompt 会同步更新。")
    selected_phrase = st.selectbox(
        "选择候选热梗",
        candidates["phrase"].tolist(),
        key="selected_hotspot_phrase",
    )
    selected = candidates[candidates["phrase"] == selected_phrase].iloc[0]

    st.subheader("所选热梗分析")
    api_key, _ = get_llm_settings()
    dimension_a, dimension_b, dimension_c, dimension_d = st.columns(4)
    dimension_a.metric("热度趋势", selected["hotness_level"])
    dimension_b.metric("表达质量", selected["expression_quality"])
    dimension_c.metric("美妆可用性", selected["beauty_usability_score"])
    dimension_d.metric("生命周期", selected["lifecycle"])
    left, right = st.columns([1, 1])
    use_semantic_rag = False
    selected_rag_matches = []
    selected_rag_context = selected["rag_context"]
    selected_rag_confidence = selected["rag_confidence"]
    with left:
        st.markdown("#### 规则判断结果")
        st.caption("这些结果由本地规则根据频次、增长、覆盖和互动数据计算。")
        st.write(f"热度趋势分：**{selected['hotness_score']}**")
        st.write(f"趋势判断：**{selected['trend_label']}**")
        st.caption(
            f"动态爆发频次线：近 7 天 ≥ {int(selected['high_freq_cutoff'])}；"
            "仅当同时满足增长阈值和独立内容支持条件时，才会标记为“爆发中”。"
        )
        st.write(f"生命周期：**{selected['lifecycle']}**")
        if selected["expression_quality"] != "自然":
            st.caption(f"复查原因：{selected['review_reason']}")
        with st.expander(
            "查看原始语境证据",
            expanded=selected["expression_quality"] != "自然",
        ):
            st.caption("以下片段来自上传 CSV 的原始文本，仅用于判断这个候选是否是人类自然表达。")
            st.code(selected["evidence_context"], language=None)
        st.write(f"风险提示：{selected['risk_tip']}")
        with st.expander(
            "查看美妆词库检索参考",
            expanded=False,
        ):
            st.caption(
                "词面 RAG 默认开启：保留候选短语、原始语境和产品信息中的可解释命中。"
                "语义 RAG 只补充近义条目，不会直接改变热度趋势或美妆可用性评分。"
            )
            if api_key:
                use_semantic_rag = st.checkbox(
                    "启用语义 RAG（调用 Embeddings API）",
                    value=False,
                    key=f"semantic_rag_{selected_phrase}",
                    help="只对当前选中的词条检索一次，并缓存结果；适合检查“词库没写同样的词，但业务含义可能接近”的情况。",
                )
            else:
                use_semantic_rag = False
                st.caption("未配置 API Key，当前仅使用词面 RAG；配置后可手动开启语义 RAG。")

            semantic_rag_error = ""
            if use_semantic_rag:
                with st.spinner("正在检索与当前热梗语义相关的美妆词库条目……"):
                    selected_rag_matches, semantic_rag_error = retrieve_hybrid_lexicon_matches(
                        selected_phrase,
                        source_texts=[selected["evidence_context"]],
                        product_context=product_context,
                    )
                selected_rag_context = format_rag_context(selected_rag_matches)
                selected_rag_confidence = classify_rag_confidence(selected_rag_matches)
                if semantic_rag_error:
                    st.warning(
                        "语义 RAG 调用失败，已自动回退到词面 RAG；"
                        f"可继续使用当前结果。错误信息：{semantic_rag_error}"
                    )
                else:
                    st.success(
                        f"已完成混合 RAG：词面命中 + 语义相似条目；当前共展示 {len(selected_rag_matches)} 条参考。"
                    )
            else:
                selected_rag_matches = retrieve_lexicon_matches(
                    selected_phrase,
                    source_texts=[selected["evidence_context"]],
                    product_context=product_context,
                )
                selected_rag_context = selected["rag_context"]

            st.caption(
                f"当前 RAG：{('混合检索' if use_semantic_rag else '词面检索')} · "
                f"{len(selected_rag_matches) or int(selected['rag_match_count'])} 条 · {selected_rag_confidence}"
            )
            st.markdown(selected_rag_context)
    selected_state_key = (
        f"{selected_phrase}_{selected['beauty_usability_score']}_{selected['hotness_level']}_{selected['lifecycle']}_"
        f"{'semantic' if use_semantic_rag else 'lexical'}_{abs(hash(product_context))}"
    )
    selected_prompt = selected["llm_prompt"]
    if use_semantic_rag:
        prompt_row = selected.to_dict()
        prompt_row.update(
            {
                "rag_context": selected_rag_context,
                "rag_confidence": selected_rag_confidence,
                "rag_match_count": len(selected_rag_matches),
            }
        )
        selected_prompt = generate_llm_prompt(prompt_row, brand_profile=brand_profile)
    with right:
        st.markdown("#### 大模型分析")
        st.caption(
            f"大模型将从{brand_profile['industry']}行业、{brand_profile['tone']}调性和{brand_profile['goal']}目标出发，"
            "根据当前热点数据生成内容建议。"
        )
        strategy_context = build_brand_strategy_context(brand_profile)
        with st.expander("查看本次品牌适配策略", expanded=True):
            st.write(f"**行业关注：** {strategy_context['fit_focus']}")
            st.write(f"**目标关注：** {strategy_context['goal_focus']}")
            if brand_profile["notes"] != "未补充其他品牌信息":
                st.write(f"**品牌补充：** {brand_profile['notes']}")
        if api_key:
            st.success("大模型 API 已配置")
        else:
            st.info("当前为本地 Prompt 模式：配置 API Key 后即可直接生成分析结果。")

        with st.expander("查看本次发送给大模型的 Prompt", expanded=False):
            st.text_area(
                "提示词内容",
                selected_prompt,
                height=300,
                key=f"llm_prompt_{selected_state_key}",
            )

        result_key = f"llm_result_{selected_state_key}"
        if st.button("调用大模型生成分析", type="primary", key=f"call_llm_{selected_state_key}"):
            with st.spinner("大模型正在分析这个热点，请稍等……"):
                success, result = call_llm(selected_prompt)
            st.session_state[result_key] = {"success": success, "text": result}

        if result_key in st.session_state:
            llm_result = st.session_state[result_key]
            if llm_result["success"]:
                if isinstance(llm_result["text"], dict):
                    render_llm_result(llm_result["text"])
                else:
                    st.markdown("#### 大模型分析结果")
                    st.markdown(llm_result["text"])
            else:
                st.warning(llm_result["text"])

    st.subheader("导出报告")
    csv_bytes = candidates.to_csv(index=False).encode("utf-8-sig")
    markdown_report = build_markdown_report(candidates, brand_profile=brand_profile)
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
