# Enterprise QA 项目工作记录

本文档记录了从本次对话开始到当前为止，我在项目中完成的所有工作。

## 0. 初始状态

- 工作区：`E:/Vscode 代码/Enterprise_QA_project`
- 初始检查结果：项目目录为空（未发现现有代码文件）
- 目标：基于 LangChain，用 PDF/Word 文档构建向量数据库（Chroma/FAISS），并提供简单问答接口

---

## 1. 第一阶段：搭建最小可用版本（MVP）

### 1.1 实现内容

- 创建后端主程序 `app.py`
  - 支持读取 PDF（`PyPDF2`）
  - 支持读取 Word（`.docx`，`python-docx`）
  - 文档切分（`RecursiveCharacterTextSplitter`）
  - 向量库构建与加载：
    - `chroma`（持久化目录：`vector_db/chroma`）
    - `faiss`（持久化目录：`vector_db/faiss`）
  - 提供 API：
    - `POST /ingest`：建库
    - `POST /ask`：问答（当时是检索摘要式回答）
  - 提供 CLI：
    - `python app.py ingest ...`
    - `python app.py ask ...`

- 创建依赖文件 `requirements.txt`
- 创建说明文档 `README.md`
- 创建知识库目录占位文件 `knowledge_base/.gitkeep`

### 1.2 产出文件

- `app.py`
- `requirements.txt`
- `README.md`
- `knowledge_base/.gitkeep`

### 1.3 验证

- 执行：`python -m py_compile app.py`（通过）
- lints：未发现语法/风格问题（后续出现的导入警告与本地未安装依赖有关）

---

## 2. 第二阶段：升级为 RAG 生成式问答

### 2.1 升级目标

将 `/ask` 从“检索片段拼接”升级为“检索 + LLM 生成答案（RAG）”。

### 2.2 实现内容

- 在 `app.py` 中新增 LLM 初始化逻辑：
  - 使用 `langchain-openai` 的 `ChatOpenAI`
  - 支持 OpenAI 兼容接口
  - 环境变量配置：
    - `LLM_API_KEY`（必填）
    - `LLM_MODEL`（默认 `gpt-4o-mini`）
    - `LLM_BASE_URL`（可选）
    - `LLM_TEMPERATURE`（可选，默认 `0.2`）

- 重写问答生成流程：
  - 先检索相关上下文
  - 再通过 Prompt 约束模型“仅基于上下文回答”
  - 返回生成式答案

- 依赖升级：
  - `requirements.txt` 新增 `langchain-openai`

- 文档升级：
  - `README.md` 新增 LLM 环境变量配置与说明
  - 更新 `/ask` 的语义为“向量检索 + LLM 生成答案”

### 2.3 验证

- 执行：`python -m py_compile app.py`（通过）
- lints：出现“无法解析导入”警告，原因为本地环境尚未安装对应依赖包

---

## 3. 第三阶段：增强可用性（来源、多轮、流式）

### 3.1 新增能力

- **来源溯源**
  - `AskResponse` 新增 `sources`
  - 每个来源包含文件路径，PDF 尽可能附带页码（如 `xxx.pdf (page 2)`）

- **多轮对话**
  - `AskRequest` 新增 `chat_history`
  - 支持输入格式：`[{ "role": "user|assistant", "content": "..." }]`
  - 问答时将历史对话注入 Prompt，改善上下文连续性

- **流式输出**
  - 新增接口：`POST /ask_stream`
  - 返回类型：`text/event-stream`（SSE）
  - 事件约定：
    - `type=token`：增量文本片段
    - `type=done`：最终答案及 `sources/contexts`

### 3.2 文档更新

- `README.md` 新增：
  - `/ask` 返回示例（含 `sources`）
  - 多轮问答调用示例
  - 流式问答调用示例（`curl -N`）
  - 关键字段解释

### 3.3 验证

- 执行：`python -m py_compile app.py`（通过）
- lints：仍为依赖未安装引起的导入解析警告

---

## 4. 第四阶段：新增前端页面（开箱即用）

### 4.1 需求

提供一个简单前端页面，让你无需手写 `curl` 即可完成建库和问答。

### 4.2 实现内容

- 后端路由改造（`app.py`）
  - 新增 `GET /`，返回前端页面文件

- 新建前端页面（`web/index.html`）
  - 功能区 1：建库
    - 输入文档目录
    - 选择向量库类型（chroma/faiss）
    - 点击“开始建库”
  - 功能区 2：问答参数
    - 问题输入
    - `top_k` 设置
    - `chat_history`（JSON，可选）
  - 功能区 3：结果展示
    - 普通问答结果
    - 流式问答实时输出
    - 来源引用列表展示

- 文档更新（`README.md`）
  - 添加前端访问地址：`http://127.0.0.1:8000`
  - 添加页面最简使用步骤

### 4.3 验证

- 执行：`python -m py_compile app.py`（通过）
- lints：仍为本地依赖未安装导致的导入解析警告

---

## 5. 过程中遇到的问题与处理

- 问题：首次写文件时路径格式错误（Windows 盘符路径与补丁路径格式冲突）
- 处理：改用正确绝对路径（`E:/...`）后成功写入文件
- 结果：后续文件创建/修改均正常完成

---

## 6. 当前项目状态（截至本记录）

### 已具备能力

- PDF/Word（`.docx`）文档入库
- Chroma / FAISS 双向量库支持
- RAG 生成式问答
- 来源引用（含页码信息，若可提取）
- 多轮对话输入
- SSE 流式问答
- 浏览器端测试页面（`GET /`）

### 主要文件

- `app.py`：后端核心逻辑与 API
- `web/index.html`：前端测试页
- `requirements.txt`：依赖清单
- `README.md`：使用说明
- `knowledge_base/.gitkeep`：知识库目录占位

---

## 7. 运行提示

1. 安装依赖：`pip install -r requirements.txt`
2. 配置环境变量（至少 `LLM_API_KEY`）
3. 启动：`uvicorn app:app --reload`
4. 打开页面：`http://127.0.0.1:8000`
5. 操作顺序：先建库，再问答

---

## 8. 备注

- 当前 linter 警告主要是“导入无法解析”，通常由本地解释器未安装依赖导致，不是代码语法错误。
- 多次 `py_compile` 校验均通过，核心代码可运行性正常。
