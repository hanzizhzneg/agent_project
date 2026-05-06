# Enterprise QA Project 2 - Agentic RAG (LangGraph + Ollama)

基于你上一版 LangChain RAG 升级为 Agentic RAG：

- 使用 LangGraph 编排检索决策流程
- 自动评估检索相关性
- 若相关性不足，自动重写查询并重试
- 最多迭代 3 轮后输出答案
- 支持 Chroma / FAISS 向量库
- 使用 Ollama 本地模型（企业内网更友好）

## 1. 环境准备

1. 安装 Python 3.10+
2. 安装并启动 Ollama
3. 拉取模型（示例）

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. 配置环境变量

复制 `.env.example` 为 `.env`，根据本机情况修改：

- `OLLAMA_BASE_URL`
- `OLLAMA_CHAT_MODEL`
- `OLLAMA_EMBED_MODEL`
- `OLLAMA_TIMEOUT_SEC`
- `INGEST_BATCH_SIZE`（大批量入库建议 16~64）
- `VECTOR_DB_DIR`
- `KNOWLEDGE_BASE_DIR`
- `API_KEY`（生产环境必须修改）
- 499-499-499-499
- `ENABLE_AUTH`
- `CORS_ALLOW_ORIGINS`

## 4. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 5. API 使用

当 `ENABLE_AUTH=true` 时，请在请求头中携带：

- `X-API-Key: <your-api-key>`
- `X-Request-Id: <optional-trace-id>`

前端页面入口：

- `GET /`：ChatGPT 风格主页（问答 + ingest + 上传即入库）
- `GET /test`：测试控制台页面

### 5.1 文档入库

`POST /ingest`

示例请求：

```json
{
  "docs_dir": "./knowledge_base",
  "store_type": "chroma",
  "chunk_size": 800,
  "chunk_overlap": 120
}
```

### 5.1.1 上传并入库（免手工放文件）

`POST /upload_and_ingest`（`multipart/form-data`）

- `files`: 可多文件上传（.pdf/.docx/.doc）
- `store_type`: `chroma` 或 `faiss`
- `chunk_size`
- `chunk_overlap`

### 5.1.2 异步入库与进度查询（推荐大批量）

- `POST /ingest_async`：提交异步入库任务
- `GET /ingest_tasks/{task_id}`：查询任务状态和进度

状态字段说明：

- `status`: `queued` / `running` / `completed` / `failed`
- `progress`: 0-100
- `stage`: 当前阶段
- `detail`: 当前阶段说明或错误信息

### 5.2 Agentic RAG 问答

`POST /ask`

示例请求：

```json
{
  "question": "公司采购审批流程是什么？",
  "store_type": "chroma",
  "top_k": 4,
  "max_iterations": 3
}
```

示例响应字段说明：

- `final_query`: 最终查询语句（可能被重写）
- `iterations_used`: 实际迭代轮次
- `trace`: 每轮的 query + 相关性评估结论
- `sources`: 最终使用的文档来源

## 6. 推荐企业级增强（下一步）

- 增加 Prompt/回答安全护栏（敏感词、越权过滤）
- 引入 reranker 与评测集（离线评估召回率/正确率）
- 扩展测试覆盖（Agent流程、异常场景、回归集）
- 接入可观测平台（Prometheus/Grafana/OpenTelemetry）

## 7. Docker 部署

```bash
docker build -t enterprise-qa-agentic:latest .
docker run --rm -p 8000:8000 --env-file .env enterprise-qa-agentic:latest
```

