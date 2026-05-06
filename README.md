# agent_project

Enterprise_QA_project
对企业知识库RAG的初步使用
用FastAPI提供HTTP服务,用 LangChain把PDF/Word（.docx）读入、切分、向量化，再存进Chroma或FAISS，最后结合Qwen（OpenAI 兼容接口）做检索增强生成.
