# Enterprise QA Demo (LangChain + Chroma/FAISS)

一个最小可用示例：基于 LangChain，从 PDF/Word (`.docx`) 文档构建向量数据库，并提供简单问答接口（RAG 生成式回答）。

## 1. 安装依赖
安装部分依赖时会出现报错，具体原因是因为镜像源的版本更新不对，可以选用官方源进行下载
不要用清华源和其他镜像源，就用最直接最稳定的官方源
pip install -i https://pypi.org/simple -r requirements.txt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

```

## 2. 配置 LLM（百炼 / Qwen，OpenAI 兼容）

在启动前设置环境变量（PowerShell）：

```powershell
# 百炼控制台创建的 API Key
$env:LLM_API_KEY="你的百炼APIKey"
$env:LLM_MODEL="qwen3.6-35b-a3b"
# 百炼 OpenAI 兼容根地址
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
# 可选：默认 0.2
$env:LLM_TEMPERATURE="0.2"
```

> 注意：`LLM_API_KEY` 和 `LLM_BASE_URL` 必须来自同一平台。  
> 如果把百炼 Key 配在 OpenAI 官方地址（或反过来），会报 `401 invalid_api_key`。

## 3. 准备文档

把你的 PDF / DOCX 文件放到：

`knowledge_base/`

你也可以在接口或命令行里传自定义目录。

## 4. 启动 API

每次打开前都需要配置

```bash
$env:LLM_API_KEY="APIKey"
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen3.6-35b-a3b"
$env:LLM_TEMPERATURE="0.2"
uvicorn app:app --reload
```

启动后可直接打开前端页面：

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

接口说明：

- `POST /ingest`：加载文档并建立向量库
- `POST /ask`：向量检索 + LLM 生成答案（返回 `sources`）
- `POST /ask_stream`：流式返回答案（SSE）

### 4.0 前端页面使用说明（最简单）

1. 在页面上先点“开始建库”
2. 填问题，点“普通问答”或“流式问答”
3. 在“来源引用”里查看答案依据

### 4.1 建库示例

```bash
curl -X POST "http://127.0.0.1:8000/ingest" ^
  -H "Content-Type: application/json" ^
  -d "{\"docs_dir\":\"knowledge_base\",\"store_type\":\"chroma\"}"
```

`store_type` 支持：

- `chroma`（本地持久化到 `vector_db/chroma`）
- `faiss`（本地持久化到 `vector_db/faiss`）

### 4.2 问答示例

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"合同付款节点是什么？\",\"store_type\":\"chroma\"}"
```

返回示例（简化）：

```json
{
  "answer": "...... [片段1][片段3]",
  "contexts": ["片段正文1", "片段正文2"],
  "sources": [
    "E:/Vscode 代码/Enterprise_QA_project/knowledge_base/a.pdf (page 2)",
    "E:/Vscode 代码/Enterprise_QA_project/knowledge_base/b.docx"
  ]
}
```

### 4.3 多轮对话示例（更好用）

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"那违约责任呢？\",\"store_type\":\"chroma\",\"chat_history\":[{\"role\":\"user\",\"content\":\"合同付款节点是什么？\"},{\"role\":\"assistant\",\"content\":\"分三期支付...\"}]}"
```

`chat_history` 只需要传 `role` 和 `content`：

- `role`: `user` 或 `assistant`
- `content`: 文本内容

### 4.4 流式回答示例（SSE）

```bash
curl -N -X POST "http://127.0.0.1:8000/ask_stream" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"总结下本合同关键风险点\",\"store_type\":\"chroma\"}"
```

你会持续收到 `data: {...}`，其中：

- `type=token`：增量文本
- `type=done`：最终结果（含 `answer/sources/contexts`）

## 5. 命令行方式（可选）

先建库：

```bash
python app.py ingest --docs-dir knowledge_base --store-type chroma
```

再提问：

```bash
python app.py ask --question "项目验收标准是什么？" --store-type chroma
```

## 6. 说明

- 默认 Embedding 模型为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，首次运行会自动下载模型。
- 当前 Word 解析支持 `.docx`。如果你有 `.doc`，建议先转成 `.docx`。
- 未设置 `LLM_API_KEY` 时，`/ask` 会报错并提示缺少配置。
- `sources` 会给出引用来源（文件路径 + 页码，若可用），方便溯源。
