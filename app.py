"""app.py — RAG Search with multi-turn chat interface and conversational memory"""
import os

# Gemini to use v1beta
os.environ["GOOGLE_GENAI_API_VERSION"] = "v1beta"

# Silence USER_AGENT warning
os.environ["USER_AGENT"] = "CollegeRAG/1.0 (student project)"

import streamlit as st
from pathlib import Path
import sys
import time
import uuid
import json

# Ensure the repo src is importable
sys.path.append(str(Path(__file__).parent))

# Your RAG components
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

st.set_page_config(page_title="RAG Search", layout="centered")

# Hide default chat-message avatar icons
st.markdown(
    """
    <style>
    [data-testid^="stChatMessageAvatar"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state keys (do this before creating widgets)."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    # Multi-turn chat messages: list of {"role": "user"|"assistant", "content": str}
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    # Unique thread_id per chat session — ties into LangGraph MemorySaver
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


# Path to save the FAISS index
VECTORSTORE_PATH = "faiss_index"

@st.cache_resource
def initialize_rag(rebuild_index: bool = False):
    """
    Cached initialization of RAG components.
    
    Args:
        rebuild_index: If True, force re-embedding of documents.
    """
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
    )
    vector_store = VectorStore()
    
    # Check if index exists and we don't want to rebuild
    if os.path.exists(VECTORSTORE_PATH) and not rebuild_index:
        try:
            vector_store.load_index(VECTORSTORE_PATH)
            doc_count = "Loaded from disk"
        except Exception as e:
            st.warning(f"Failed to load existing index: {e}. Rebuilding...")
            rebuild_index = True
            
    # If we need to build (fresh start or rebuild requested)
    if not os.path.exists(VECTORSTORE_PATH) or rebuild_index:
        sources = Config.DEFAULT_URLS
        if os.path.exists("data"):
            sources.append("data")
            
        documents = doc_processor.process_urls(sources)
        
        progress_bar = st.progress(0, text="Starting ingestion...")
        status_text = st.empty()
        
        def update_progress(msg, percent):
            p = min(max(percent, 0.0), 1.0)
            progress_bar.progress(p, text=msg)

        try:
            vector_store.create_vectorstore(documents, progress_callback=update_progress)
            vector_store.save_index(VECTORSTORE_PATH)
            doc_count = len(documents)
            
            progress_bar.progress(1.0, text="Ingestion complete!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            raise e

    graph_builder = GraphBuilder(retriever=vector_store.get_retriever(), llm=llm)
    graph_builder.build()
    return graph_builder, doc_count


def normalize_answer(ans) -> str:
    """
    Convert various types of 'answer' that the RAG/LLM might return to a single string.
    """
    if ans is None:
        return ""
    if isinstance(ans, str):
        return ans
    if isinstance(ans, (list, tuple)):
        try:
            if all(isinstance(x, str) for x in ans):
                return "\n\n".join(ans)
            return "\n\n".join(
                json.dumps(x, ensure_ascii=False)
                if isinstance(x, (dict, list))
                else str(x)
                for x in ans
            )
        except Exception:
            return str(ans)
    if isinstance(ans, dict):
        try:
            return json.dumps(ans, indent=2, ensure_ascii=False)
        except Exception:
            return str(ans)
    if isinstance(ans, (bytes, bytearray)):
        try:
            return ans.decode("utf-8", errors="replace")
        except Exception:
            return str(ans)
    return str(ans)


def main():
    init_session_state()

    st.title("🔍 Agentic RAG: Docs, Web & Wiki")
    st.markdown("Ask questions about your **Documents**, **Wikipedia**, or **Web Search**.")

    # ── Initialize RAG once (cached) ──────────────────────────
    if not st.session_state.initialized:
        with st.spinner("Loading RAG system..."):
            try:
                rag_system, num_chunks = initialize_rag()
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"System ready! ({num_chunks} document chunks loaded)")
            except Exception as e:
                st.session_state.initialized = False
                st.error(f"Failed to initialize RAG: {e}")
                return


    # ── Chat history display ──────────────────────────────────
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question..."):
        if not st.session_state.initialized or st.session_state.rag_system is None:
            st.error("RAG system is not ready.")
            return

        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                try:
                    result = st.session_state.rag_system.run(
                        prompt,
                        thread_id=st.session_state.thread_id,
                        chat_history=st.session_state.chat_messages
                    )
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    result = None
                elapsed = time.time() - start_time

            if result:
                raw_answer = result.get("answer")
                answer = normalize_answer(raw_answer)

                st.markdown(answer)
                st.caption(f"Response time: {elapsed:.2f}s")

                # Show source documents in an expander
                docs = result.get("retrieved_docs", [])
                if docs:
                    with st.expander("📄 Source Documents"):
                        for i, doc in enumerate(docs, 1):
                            content = getattr(doc, "page_content", str(doc))
                            preview = content[:1200] + ("..." if len(content) > 1200 else "")
                            st.text_area(
                                f"Document {i}", value=preview, height=140, disabled=True
                            )

                # Save assistant message to chat history
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            else:
                fallback = "Sorry, I couldn't generate an answer. Please try again."
                st.markdown(fallback)
                st.session_state.chat_messages.append({"role": "assistant", "content": fallback})


if __name__ == "__main__":
    main()
