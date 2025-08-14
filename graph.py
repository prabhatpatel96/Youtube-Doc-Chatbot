import os
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompts import (
    SUMMARY_PROMPT, QA_PROMPT,
    KEY_INSIGHTS_PROMPT, TOPIC_CLASSIFIER_PROMPT, CONVERSATIONAL_REWRITE_PROMPT
)

parser = StrOutputParser()

def _llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        model="z-ai/glm-4.5-air:free",
        temperature=0.2,
    )

def retrieve_node(state: Dict, vector_index) -> Dict:
    k = state.get("k", 4)
    query = state["query"]
    docs = vector_index.retrieve(query, k=k)
    state["retrieved"] = [d.page_content for d in docs]
    state["sources"] = [
        {
            "text": d.page_content,
            "preview": d.page_content[:100].replace("\n", " ") + "..."
        }
        for d in docs
    ]
    return state

def summarize_node(state: Dict, vector_index=None) -> Dict:
    llm = _llm()
    context = "\n\n".join(state.get("retrieved", []))[:12000]
    state["summary"] = "" if not context.strip() else (SUMMARY_PROMPT | llm | parser).invoke({"context": context})
    return state

def qa_node(state: Dict, vector_index=None) -> Dict:
    llm = _llm()
    context_segments = state.get("retrieved", [])
    if not context_segments:
        state["answer"] = "this is not present in video/document"
    else:
        context = "\n\n".join(context_segments)[:12000]
        state["answer"] = (QA_PROMPT | llm | parser).invoke({"question": state["query"], "context": context})
    return state

def key_insights_node(state: Dict, vector_index=None) -> Dict:
    llm = _llm()
    context = "\n\n".join(state.get("retrieved", []))[:12000]
    state["insights"] = "" if not context.strip() else (KEY_INSIGHTS_PROMPT | llm | parser).invoke({"context": context})
    return state

def classify_node(state: Dict, vector_index=None) -> Dict:
    llm = _llm()
    context = "\n\n".join(state.get("retrieved", []))[:8000]
    state["classification"] = "" if not context.strip() else (TOPIC_CLASSIFIER_PROMPT | llm | parser).invoke({"context": context})
    return state

def friendly_rewrite_node(state: Dict, vector_index=None) -> Dict:
    llm = _llm()
    context = "\n\n".join(state.get("retrieved", []))[:12000]
    state["rewrite"] = "" if not context.strip() else (CONVERSATIONAL_REWRITE_PROMPT | llm | parser).invoke({"context": context})
    return state

def respond_node(state: Dict, vector_index=None) -> Dict:
    mode = state.get("mode")
    if mode == "summarize_qa":
        final = f"{state.get('summary','')}\n\n{state.get('answer','')}"
    elif mode == "qa":
        final = state.get("answer", "")
    elif mode == "key_insights":
        final = state.get("insights", "")
    elif mode == "classify":
        final = state.get("classification", "")
    elif mode == "friendly_rewrite":
        final = state.get("rewrite", "")
    else:
        final = "[ERROR] Unknown mode."
    state["last_response"] = final.strip()
    hist: List[Tuple[str,str]] = state.get("history", [])
    hist.append(("user", state["query"]))
    hist.append(("assistant", state["last_response"]))
    state["history"] = hist[-50:]
    return state

def run_graph_step(state: Dict, vector_index):
    s = retrieve_node(state, vector_index)
    mode = state.get("mode")
    if mode == "summarize_qa":
        s = summarize_node(s, vector_index)
        s = qa_node(s, vector_index)
    elif mode == "qa":
        s = qa_node(s, vector_index)
    elif mode == "key_insights":
        s = key_insights_node(s, vector_index)
    elif mode == "classify":
        s = classify_node(s, vector_index)
    elif mode == "friendly_rewrite":
        s = friendly_rewrite_node(s, vector_index)
    s = respond_node(s, vector_index)
    return s