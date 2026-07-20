# AI 辅助热点洞察与趋势决策工作流 Demo

> 一个面向内容运营的 AI 热点洞察 Agent 原型：把社交媒体热点数据转化为可执行的热点等级判断，并生成可交给大模型的分析 Prompt。

## 在线体验

- 在线 Demo：部署后补充
- GitHub 源码：部署后补充

这是一个可本地运行的 Streamlit 作品集 demo，用模拟数据展示“热点监测 -> 候选筛选 -> 趋势判断 -> 生成 AI Prompt -> 运营建议输出”的自动化流程。

## 这是脱敏 demo

- 数据全部来自 `sample_hotspot_data.csv`，为虚构的小红书 / 微博 / 抖音风格内容。
- 不连接真实平台，不爬取真实网站，不调用外部 API。
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

## 文件结构

```text
.
├── app.py
├── requirements.txt
├── sample_hotspot_data.csv
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
7. AI Prompt：根据当前候选词条和判断结果，生成一段可复制到 Kimi / 通义 / ChatGPT 的分析指令。
8. 报告导出：支持导出候选结果 CSV 和 Markdown 报告。


## 免责声明

本项目仅使用模拟数据，不包含任何公司内部数据、真实账号权限、源代码或业务机密。项目只用于作品集展示和本地流程演示。
