from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from app.config import settings


class AgenticRagState(TypedDict):
    question: str
    query: str
    top_k: int
    max_iterations: int
    iteration: int
    retrieved_docs: list[Document]
    relevance_decision: str
    relevance_reason: str
    trace: list[dict]
    answer: str


def _llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=0.1,
        client_kwargs={"timeout": settings.ollama_timeout_sec},
    )


def _docs_to_text(docs: list[Document]) -> str:
    chunks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "n/a")
        chunks.append(f"[{i}] source={source}, page={page}\n{d.page_content}")
    return "\n\n".join(chunks)


def build_agentic_rag_graph(retriever):
    llm = _llm()

    def retrieve_node(state: AgenticRagState) -> AgenticRagState:
        docs = retriever.invoke(state["query"])
        state["retrieved_docs"] = docs
        return state

    def grade_relevance_node(state: AgenticRagState) -> AgenticRagState:
        docs_text = _docs_to_text(state["retrieved_docs"])
        prompt = (
            "你是企业问答系统的检索评估器。请判断检索结果是否足以回答用户问题。"
            "只输出一行JSON，格式："
            '{"decision":"sufficient|insufficient","reason":"..."}。\n'
            "若文档与问题高度相关且信息完整，decision=sufficient；否则insufficient。"
        )
        result = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        f"问题：{state['question']}\n"
                        f"当前查询：{state['query']}\n"
                        f"检索内容：\n{docs_text[:settings.rag_relevance_max_chars]}"
                    )
                ),
            ]
        ).content
        text = str(result).strip()

        decision = "insufficient"
        reason = text
        if '"decision":"sufficient"' in text or '"decision": "sufficient"' in text:
            decision = "sufficient"

        state["relevance_decision"] = decision
        state["relevance_reason"] = reason
        state["trace"] = state["trace"] + [
            {
                "iteration": state["iteration"],
                "query": state["query"],
                "relevance_decision": decision,
                "relevance_reason": reason,
            }
        ]
        return state

    def rewrite_query_node(state: AgenticRagState) -> AgenticRagState:
        prompt = (
            "你是企业知识库检索查询重写器。"
            "根据用户问题和上一轮检索不足的原因，输出一个更精确的新查询。"
            "只输出重写后的查询文本，不要解释。"
        )
        rewritten = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        f"用户问题：{state['question']}\n"
                        f"上一轮查询：{state['query']}\n"
                        f"不足原因：{state['relevance_reason']}"
                    )
                ),
            ]
        ).content
        state["query"] = str(rewritten).strip()
        state["iteration"] = state["iteration"] + 1
        return state

    def generate_answer_node(state: AgenticRagState) -> AgenticRagState:
        docs_text = _docs_to_text(state["retrieved_docs"])
        prompt = (
            "你是企业知识库问答助手。仅基于给定上下文回答问题。"
            "若上下文不足，请明确说'根据现有知识库内容无法确定'，不要编造。"
        )
        answer = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        f"问题：{state['question']}\n"
                        f"上下文：\n{docs_text[:settings.rag_answer_max_chars]}"
                    )
                ),
            ]
        ).content
        state["answer"] = str(answer).strip()
        return state

    def route_after_grading(
        state: AgenticRagState,
    ) -> Literal["generate_answer", "rewrite_query"]:
        if state["relevance_decision"] == "sufficient":
            return "generate_answer"
        if state["iteration"] >= state["max_iterations"]:
            return "generate_answer"
        return "rewrite_query"

    graph = StateGraph(AgenticRagState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_answer", generate_answer_node)

    if settings.rag_fast_mode:
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate_answer")
        graph.add_edge("generate_answer", END)
        return graph.compile()

    graph.add_node("grade_relevance", grade_relevance_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges(
        "grade_relevance",
        route_after_grading,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)
    return graph.compile()
