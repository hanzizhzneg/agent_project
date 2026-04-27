# Enterprise QA 一页纸（简版）

## 项目目标

基于 LangChain，从 PDF/Word 文档构建向量数据库（Chroma/FAISS），并提供可直接使用的问答能力（含前端页面）。

## 里程碑

1. **MVP 完成**
   - 实现 PDF + DOCX 文档读取
   - 实现向量化与持久化（Chroma/FAISS）
   - 提供 `POST /ingest` 与 `POST /ask`
   - 提供 CLI 入口

2. **升级为 RAG 生成式问答**
   - 接入 `ChatOpenAI`（OpenAI 兼容）
   - 支持环境变量配置：`LLM_API_KEY`、`LLM_MODEL`、`LLM_BASE_URL`、`LLM_TEMPERATURE`
   - `/ask` 从检索摘要升级为“检索 + LLM 生成”

3. **增强可用性**
   - 增加 `sources` 来源溯源（文件路径 + 页码）
   - 支持 `chat_history` 多轮上下文
   - 新增 `POST /ask_stream`（SSE 流式输出）

4. **提供前端页面**
   - 新增 `GET /` 页面入口
   - 页面支持：建库、普通问答、流式问答、来源展示
   - 降低使用门槛，无需手写 `curl`

## 当前已具备能力

- PDF / DOCX 文档入库
- Chroma / FAISS 双存储可切换
- RAG 生成式回答
- 来源引用与可追溯性
- 多轮对话
- 流式问答
- 浏览器可视化测试页

## 核心文件

- `app.py`：后端 API 与 RAG 主逻辑
- `web/index.html`：前端测试页
- `requirements.txt`：依赖清单
- `README.md`：完整使用说明
- `WORKLOG.md`：详细全过程记录

## 快速启动（5 步）

1. 安装依赖：`pip install -r requirements.txt`
2. 设置密钥：`LLM_API_KEY`
3. 启动服务：`uvicorn app:app --reload`
4. 打开页面：`http://127.0.0.1:8000`
5. 先建库，再问答

## 当前注意事项

- 若看到“无法解析导入”的 linter 警告，通常是本地环境未安装依赖，不是代码语法错误。
- 多次 `python -m py_compile app.py` 校验已通过。
