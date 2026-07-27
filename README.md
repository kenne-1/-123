# AI 辅助热点洞察与趋势决策工作流 Demo

> 一个面向内容运营的 AI 热点洞察 Agent 原型：把社交媒体热点数据转化为可执行的热点等级判断，并可调用大模型生成内容策略建议。

## 在线体验

- 在线 Demo：部署后补充
- GitHub 源码：部署后补充

这是一个可本地运行的 Streamlit 作品集 demo，用模拟数据展示“热点监测 -> 候选筛选 -> 趋势判断 -> 生成 AI Prompt -> 可选调用大模型 -> 运营建议输出”的自动化流程。

## 这是脱敏 demo

- 数据全部来自 `sample_hotspot_data.csv`，为虚构的小红书 / 微博 / 抖音风格内容。
- 默认不连接真实平台、不爬取真实网站；未配置 API Key 时，项目只运行本地规则和 Prompt 模式。
- 品牌名均为 BrandA、BrandB、BrandC、BrandD 等模拟名称。
- 不包含任何公司内部数据、真实账号权限、源代码或业务机密。

## 如何安装

```bash
pip install -r requirements.txt
```

## 如何运行

```bash
streamlit run app.py
```

运行后浏览器会打开本地页面。默认读取 `sample_hotspot_data.csv`，也可以上传同字段结构的 CSV。

## 可选：接入大模型 API

项目支持通过 OpenAI API 对选中的热点生成结构化的语义解释、商业价值、内容建议、风险提示和 CTA。

1. 复制配置模板：

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

2. 在 `.streamlit/secrets.toml` 中填写 API Key：

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_MODEL = "gpt-5.6"
```

3. 重新运行应用，在“所选热梗分析”区域点击“调用大模型生成分析”。

调用成功后，结果会按语义解释、情绪倾向、商业价值、内容行动、风险提示和 CTA 分模块展示，而不是直接堆成一段文本。

`.streamlit/secrets.toml` 仅用于本地配置，已被 `.gitignore` 忽略，不应提交到 GitHub。部署到 Streamlit Community Cloud 时，请在应用的 Secrets 设置中填写同样的配置。

## 品牌画像配置

页面左侧支持配置本次分析的品牌画像：

- 品牌行业：提供通用、美妆、食品饮料、家清、数码等选项，也可以填写其他行业。
- 品牌调性：支持选择多个常用调性，并补充自定义描述。
- 本次目标：曝光、互动、转化、内容测试，也可以填写自定义目标。
- 品牌补充信息：可填写目标人群、品牌偏好或需要规避的表达。

这些信息会进入当前热点的 AI Prompt，并写入导出报告，用于让内容建议更贴近具体品牌场景。

## 文件结构

```text
.
├── app.py
├── requirements.txt
├── sample_hotspot_data.csv
├── .streamlit/secrets.toml.example
├── README.md
├── workflow说明.md
└── demo_output_sample.md
```

## 工作流说明

1. 数据输入：读取默认模拟数据或用户上传 CSV。
2. 数据清洗：去除 URL、@提及、特殊符号和纯数字。
3. 候选短语提取：用中文字符 2-gram / 3-gram 生成候选短语，并过滤泛化营销词。
4. 趋势计算：比较近 7 天和前 7 天频次，计算增长率、品牌覆盖、平台覆盖、简化 PMI 分数和互动强度。
5. 生命周期判断：标注潜力期、爆发中、衰减期、低热观察。
6. 热点等级：输出“夯 / 人上人 / NPC / 拉完了”和风险提示。
7. 品牌画像：结合行业、品牌调性、本次目标和补充信息，生成更贴合品牌场景的分析上下文。
8. AI Prompt：根据当前候选词条、判断结果和品牌画像，生成结构化的大模型分析指令。
9. 大模型调用：配置 API Key 后，可在页面中点击按钮直接获得内容策略建议；未配置时仍可复制 Prompt 手动使用。
10. 报告导出：支持导出候选结果 CSV 和包含品牌画像的 Markdown 报告。


## 免责声明

本项目仅使用模拟数据，不包含任何公司内部数据、真实账号权限、源代码或业务机密。项目只用于作品集展示和本地流程演示。
