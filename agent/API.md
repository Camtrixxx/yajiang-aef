# Yajiang Report Agent API

面向前端的统一入口是 **Agent 服务**。前端只需要访问 Agent，不需要直接访问 AEF 模型推理服务。

## 服务拓扑

```text
Frontend
  -> Agent API :7870
      -> AEF inference service :7862
      -> agent/reports/*.html, *.md, assets/*.png
```

- 对前端暴露：`http://112.111.7.74:1112`
- 内部依赖：`http://127.0.0.1:7862`
- 默认已开启 CORS：`AGENT_CORS_ORIGINS=*`

当前公网访问通过 EIP DNAT 转发：

```text
112.111.7.74:1112 -> 实例 7870
```

生产或联调环境建议只暴露 Agent 对外端口，不要把 `7862` 暴露给前端或公网。

## 任务和地区

当前支持任务：

| 中文任务 | 说明 |
| --- | --- |
| `地物分类` | 输出地物分类报告、分类对比图、叠加图和分类指标 |
| `水体分布` | 输出水体分类报告、水体概率/掩膜对比图和水体指标 |
| `高程地形` | 输出高程地形报告、地形总览图、DEM 验证图和回归指标 |

当前可用地区：

| 地区 | 说明 |
| --- | --- |
| `雅江区域` | 当前 AEF 闭环验证区域 |

时间格式：

- 推荐由用户自然语言输入，例如：`去年九月份`、`2025年9月`
- 前端也可以显式传 `time_range: "2025-09"`

## 状态机

`POST /api/report` 的响应字段 `status` 决定前端如何展示：

| status | 含义 | 前端行为 |
| --- | --- | --- |
| `ok` | 报告已生成 | 展示 `message`，并渲染 `report` 卡片和右侧预览 |
| `needs_input` | 缺少必要槽位，通常是月份 | 展示 `message`，等待用户继续输入 |
| `needs_confirmation` | 需要确认是否沿用历史槽位 | 展示 `message` 和确认按钮/文本输入 |
| `chat` | 普通自然语言对话 | 只展示 `message`，不展示报告卡片 |

## Endpoints

### `GET /api/health`

健康检查。

响应示例：

```json
{
  "status": "ok",
  "service": "yajiang-report-agent",
  "backend": "fastapi"
}
```

### `POST /api/report`

主接口。用于自然语言对话、补槽、报告生成。

请求体：

```json
{
  "session_id": "frontend-session-001",
  "task": "地物分类",
  "region": "雅江区域",
  "prompt": "给我一份去年九月份的地物分类报告",
  "time_range": ""
}
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `session_id` | 否 | 会话 ID。前端应为每个聊天窗口生成稳定 ID。默认 `default` |
| `task` | 否 | 前端选择的任务。默认 `地物分类` |
| `region` | 否 | 前端选择的地区。默认 `雅江区域` |
| `prompt` | 是 | 用户自然语言输入 |
| `time_range` | 否 | `YYYY-MM`，前端已知月份时可直接传 |

成功生成报告响应节选：

```json
{
  "status": "ok",
  "message": "报告已生成。",
  "session_id": "frontend-session-001",
  "request": {
    "task": "地物分类",
    "region": "雅江区域",
    "prompt": "给我一份去年九月份的地物分类报告",
    "time_range": "2025-09",
    "session_id": "frontend-session-001"
  },
  "report": {
    "title": "雅江区域2025-09地物分类遥感分析报告",
    "html_url": "/reports/2025-09-aef_inference-xxxx.html",
    "markdown_url": "/reports/2025-09-aef_inference-xxxx.md",
    "llm_provider": "template:missing_api_key",
    "reused": false
  },
  "analysis": {
    "data_source": "aef_inference",
    "metrics": [
      {"label": "任务", "value": "地物分类", "description": "Agent 识别后的标准任务"},
      {"label": "总体精度", "value": "98.3%", "description": "与 WorldCover 真值对比得到的平均精度"}
    ],
    "charts": [
      {
        "title": "地物分类真值与推理对比",
        "kind": "image",
        "url": "/reports/assets/aef_landcover_compare_xxxx.png",
        "caption": "展示地物分类真值、模型预测、正确/错误区域和置信度。"
      }
    ],
    "aef_payload": {
      "service": "http://127.0.0.1:7862",
      "task": "landcover",
      "region": "雅江区域",
      "time_range": "2025-09",
      "sample_indices": [300],
      "selector": "temporary_deterministic_patch_selector"
    }
  }
}
```

缺月份响应示例：

```json
{
  "status": "needs_input",
  "message": "请在需求里补充要分析的月份，例如：去年十月份、2025年9月。",
  "report": null
}
```

自然语言聊天响应示例：

```json
{
  "status": "chat",
  "message": "我是雅江遥感报告助手，主要帮你把自然语言需求整理成标准化遥感任务，调用 AEF 模型完成地物分类、水体分类或高程地形分析，然后生成带图表的报告。",
  "report": null
}
```

前端 URL 拼接规则：

```text
AGENT_BASE_URL = http://112.111.7.74:1112
absolute_html_url = AGENT_BASE_URL + response.report.html_url
absolute_image_url = AGENT_BASE_URL + response.analysis.charts[0].url
```

### `GET /api/sessions?limit=30`

获取最近会话列表。

响应示例：

```json
{
  "status": "ok",
  "sessions": [
    {
      "session_id": "frontend-session-001",
      "title": "雅江区域 地物分类 2025-09",
      "summary": "最近一次报告任务：雅江区域，地物分类，2025-09。",
      "mode": "ok",
      "updated_at": "2026-06-26T06:30:00+00:00"
    }
  ]
}
```

### `GET /api/session/{session_id}`

获取单个会话详情、最近消息、记忆和最近报告。

响应示例结构：

```json
{
  "status": "ok",
  "session": {
    "session_id": "frontend-session-001",
    "messages": [],
    "reports": []
  },
  "memory": {
    "current_intent": {},
    "pending_slots": [],
    "recent_messages": [],
    "reports": []
  }
}
```

### `POST /api/session/reset`

清空指定会话的记忆、消息和报告索引。

请求体：

```json
{
  "session_id": "frontend-session-001"
}
```

响应：

```json
{
  "status": "ok",
  "session_id": "frontend-session-001"
}
```

### `GET /reports/{filename}`

静态报告和图片文件。

常见类型：

- `.html`: 完整 HTML 报告
- `.md`: Markdown 原文
- `.png`: 报告图像资产

## 前端最小接入流程

1. 前端启动时调用 `GET /api/health`。
2. 用户发送消息时调用 `POST /api/report`。
3. 把 `message` 渲染为助手回复。
4. 当 `status=ok && report` 时，渲染报告卡片。
5. 点击报告卡片时，在右侧面板加载 `AGENT_BASE_URL + report.html_url`。
6. Markdown 按钮加载 `AGENT_BASE_URL + report.markdown_url`。
7. 左侧历史会话调用 `GET /api/sessions` 和 `GET /api/session/{session_id}`。

## Curl 示例

```bash
BASE=http://112.111.7.74:1112

curl --noproxy '*' "$BASE/api/health"

curl --noproxy '*' -X POST "$BASE/api/report" \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-001",
    "task": "地物分类",
    "region": "雅江区域",
    "prompt": "给我一份去年九月份的地物分类报告"
  }'

curl --noproxy '*' -X POST "$BASE/api/report" \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-001",
    "task": "水体分布",
    "region": "雅江区域",
    "prompt": "给我一份2025年9月的水体分类报告"
  }'

curl --noproxy '*' -X POST "$BASE/api/report" \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-001",
    "task": "高程地形",
    "region": "雅江区域",
    "prompt": "生成一份2025年9月雅江区域高程地形分析报告"
  }'
```
