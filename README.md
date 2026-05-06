# agent_project

### Enterprise_QA_project
对企业知识库RAG的初步使用

用FastAPI提供HTTP服务,用 LangChain把PDF/Word（.docx）读入、切分、向量化，再存进Chroma或FAISS，最后结合Qwen（OpenAI 兼容接口）做检索增强生成.

### Enterprise_QA_project2


在普通 RAG 基础上做成了 Agentic RAG：

技术栈：FastAPI + LangGraph 编排 + LangChain 检索 + Ollama 本地大模型与嵌入。

流程特点：检索后会评估相关性；不够相关时会自动改写查询并重试，重复多轮再给出答案。

向量库：支持 Chroma 或 FAISS，文档切块后写入 VECTOR_DB_DIR。

知识来源：从 KNOWLEDGE_BASE_DIR读文档；也支持 上传 PDF/Word 即入库。

对外能力：提供/ingest、/upload_and_ingest、/ingest_async、/ask 等接口；带 API Key 鉴权可选；另有 GET/聊天式页面和GET/测试页说明里的/test。

目前缺点：

1、上传文件速度较慢，大批量文件上传时会有超时报错，使用了异步策略会节省时间，但上传速度还是太慢。

​2、使用的是本地模型，因硬件配置限制，处理速度较慢，在知识库储量过大时，回答问题很慢。
