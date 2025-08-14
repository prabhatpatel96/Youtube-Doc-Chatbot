import os
import streamlit as st
from dotenv import load_dotenv
from graph import run_graph_step
from utils.vector_store import VectorIndex
from services.youtube_service import fetch_youtube_transcript
from services.doc_service import extract_text_from_files

load_dotenv()

st.set_page_config(page_title="LangGraph YouTube & Document Chatbot", layout="wide")
st.title("🎛️ LangGraph-powered Q&A over YouTube & Documents")

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "history" not in st.session_state:
    st.session_state.history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk size", 300, 2000, 800, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, 10)
    k = st.slider("Retriever top-k", 1, 10, 4, 1)
    if st.session_state.history:
        st.download_button("💾 Download Chat History", "\n".join([f"{r}: {m}" for r,m in st.session_state.history]), "chat_history.txt")

tabs = st.tabs(["🎥 YouTube", "📄 Documents", "💬 Chat"])

with tabs[0]:
    url = st.text_input("YouTube URL")
    if st.button("Fetch & Index", disabled=not url):
        with st.spinner("Fetching transcript and building vector index..."):
            text = fetch_youtube_transcript(url)
            if text.startswith("[ERROR]"):
                st.error(text)
            else:
                st.session_state.vector_index = VectorIndex.from_texts(
                    texts=[text], chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                st.success("Index built! Switch to Chat tab.")

with tabs[1]:
    up = st.file_uploader("Upload documents", type=["pdf","docx","txt"], accept_multiple_files=True)
    if st.button("Process & Index", disabled=not up):
        with st.spinner("Extracting text and building vector index..."):
            text = extract_text_from_files(up)
            if not text.strip():
                st.error("No text extracted.")
            else:
                st.session_state.vector_index = VectorIndex.from_texts(
                    texts=[text], chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                st.success("Index built! Switch to Chat tab.")

with tabs[2]:
    if st.session_state.vector_index is None:
        st.info("Ingest a YouTube video or upload documents first.")
    else:
        user_input = st.text_area("Your input", height=100)
        mode = st.selectbox("Mode", ["qa", "summarize_qa", "key_insights", "classify", "friendly_rewrite"],
                            format_func=lambda m: {
                                "qa": "Q&A",
                                "summarize_qa": "Summarize + Q&A",
                                "key_insights": "Key Insights",
                                "classify": "Topic Classification",
                                "friendly_rewrite": "Friendly Rewrite"
                            }[m])
        if st.button("Run"):
            with st.spinner("Processing..."):
                state = {
                    "history": st.session_state.history,
                    "query": user_input,
                    "mode": mode,
                    "k": k
                }
                result_state = run_graph_step(state, st.session_state.vector_index)
                st.session_state.history = result_state["history"]
                st.session_state.last_sources = result_state.get("sources", [])
                st.markdown("### Assistant")
                st.write(result_state["last_response"])
                if st.session_state.last_sources:
                    st.markdown("### Sources")
                    for i, src in enumerate(st.session_state.last_sources, 1):
                        st.write(f"**{i}.** {src['preview']}")

        if st.session_state.history:
            st.markdown("### Conversation History")
            for role, msg in st.session_state.history[-10:]:
                st.markdown(f"**{role}:** {msg}")