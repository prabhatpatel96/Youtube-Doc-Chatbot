# LangGraph YouTube & Document Q&A Chatbot

End-to-end chatbot that ingests **YouTube transcripts** or **documents (PDF/DOCX/TXT)** and performs **retrieval-augmented summarization and Q&A** using **LangGraph + LangChain** and **OpenRouter**.

## Features
- 🔗 YouTube transcript fetch via `yt-dlp` + `webvtt`
- 📄 PDF/DOCX/TXT ingestion
- 🔍 Chunking + MiniLM embeddings + FAISS
- 🧠 LangGraph workflow (retrieve → [summarize] → QA → respond)
- 🗣️ Two-paragraph summaries and Q&A constrained to context
- 🖥️ Streamlit UI
- 🧰 Clean repo, ready for GitHub

## Quickstart

1. **Clone & install**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API key**
   Create `.env`:
   ```bash
   OPENROUTER_API_KEY=sk-or-...
   ```

3. **Run**
   ```bash
   streamlit run app.py
   ```

4. **Use**
   - Ingest a YouTube URL on the **YouTube** tab or upload PDFs/DOCX/TXT on **Documents**.
   - Switch to **Chat** to ask questions or get a summary + Q&A.

## Notes
- Model: `z-ai/glm-4.5-air:free` via OpenRouter (can be changed in `graph.py`).
- All answers follow strict rules; if content is missing from context, the assistant replies with `this is not present in video/document`.
- Memory: simple chat history kept in session state.

## Structure
```
youtube-doc-bot/
│── app.py
│── graph.py
│── prompts.py
│── requirements.txt
│── README.md
│── .env.example
│
├── services/
│   ├── youtube_service.py
│   ├── doc_service.py
│
└── utils/
    ├── text_utils.py
    └── vector_store.py
```

## License
MIT
