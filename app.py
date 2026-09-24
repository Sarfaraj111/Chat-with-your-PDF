import hashlib
import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# On Streamlit Community Cloud there is no .env file; secrets come from st.secrets.
# Inject them into os.environ so LangChain picks them up the same way in both environments.
if "GOOGLE_API_KEY" in st.secrets:
    os.environ.setdefault("GOOGLE_API_KEY", st.secrets["GOOGLE_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-2"
CHAT_MODEL = "gemini-3.6-flash"
DB_ROOT = "chroma_db"  # each PDF gets its own sub-folder: chroma_db/<file-hash>

# Rate-limit settings: lower BATCH_SIZE / raise PAUSE if you still hit 429s
BATCH_SIZE = 10           # chunks embedded per API request
PAUSE_BETWEEN_BATCHES = 1.0  # seconds to wait between requests
MAX_RETRIES = 6           # retries per batch on 429 (waits 5s, 10s, 20s, 40s, 60s...)

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


# ---------- cached models ----------
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=CHAT_MODEL)


# ---------- helpers ----------
def msg_text(msg):
    """Works whether .text is a property or a method (differs across langchain versions)."""
    t = msg.text
    return t() if callable(t) else t


def add_batch_with_retry(vs, docs, ids):
    """Embed + store one batch, backing off and retrying if Google returns 429."""
    delay = 5
    for attempt in range(MAX_RETRIES):
        try:
            vs.add_documents(docs, ids=ids)
            return
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise


def build_vectorstore(file_bytes: bytes, file_name: str, progress=None):
    """Load -> split -> embed in small batches -> store.

    Chunks get deterministic IDs, so if a run is interrupted (e.g. quota hit),
    the next attempt resumes where it stopped instead of re-embedding everything.
    """
    doc_id = hashlib.md5(file_bytes).hexdigest()
    persist_dir = os.path.join(DB_ROOT, doc_id)

    # PyPDFLoader needs a real file path, so write the upload to a temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(file_bytes)
        tmp.close()
        docs = PyPDFLoader(tmp.name).load()
    finally:
        os.remove(tmp.name)

    if not any(d.page_content.strip() for d in docs):
        raise ValueError(
            "No text could be extracted. This looks like a scanned/image-only PDF (needs OCR)."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = file_name  # replace temp path with the real file name
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]

    vs = Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())

    # Skip chunks that were already embedded in an earlier (possibly interrupted) run
    existing = set(vs.get()["ids"])
    pending = [(i, c) for i, c in zip(ids, chunks) if i not in existing]
    total = len(pending)

    for start in range(0, total, BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        add_batch_with_retry(vs, [c for _, c in batch], [i for i, _ in batch])
        done = start + len(batch)
        if progress:
            progress.progress(done / total, text=f"Embedding chunks… {done}/{total}")
        time.sleep(PAUSE_BETWEEN_BATCHES)

    return vs, doc_id, len(chunks)


def get_retriever(vs):
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )


def show_sources(sources):
    if not sources:
        return
    pages = sorted({s["page"] for s in sources})
    with st.expander(f"📚 Sources (pages {', '.join(map(str, pages))})"):
        for s in sources:
            st.markdown(f"**Page {s['page']}**")
            st.caption(s["text"])


# ---------- session state ----------
for key, default in {
    "vectorstore": None,
    "doc_id": None,
    "file_name": None,
    "num_chunks": 0,
    "messages": [],
    "failed_id": None,   # hash of a PDF that failed, so we don't auto-retry on every rerun
    "error_msg": "",
}.items():
    st.session_state.setdefault(key, default)


# ---------- sidebar ----------
with st.sidebar:
    st.header("📄 Your document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        new_id = hashlib.md5(file_bytes).hexdigest()

        if new_id != st.session_state.doc_id:
            if new_id == st.session_state.failed_id:
                # Already failed once: show the error and let the user decide when to retry
                st.error(f"Could not process this PDF: {st.session_state.error_msg}")
                st.caption("Progress is saved, so retrying resumes where it stopped.")
                if st.button("🔄 Retry", use_container_width=True):
                    st.session_state.failed_id = None
                    st.rerun()
            else:
                progress = st.progress(0.0, text="Reading PDF…")
                try:
                    vs, doc_id, n = build_vectorstore(file_bytes, uploaded.name, progress)
                    progress.empty()
                    st.session_state.update(
                        vectorstore=vs,
                        doc_id=doc_id,
                        file_name=uploaded.name,
                        num_chunks=n,
                        messages=[],  # new PDF -> fresh chat
                        failed_id=None,
                    )
                except Exception as e:
                    progress.empty()
                    st.session_state.update(
                        vectorstore=None,
                        doc_id=None,
                        failed_id=new_id,
                        error_msg=str(e),
                    )
                    st.rerun()
    else:
        # file removed from the uploader -> reset
        if st.session_state.doc_id is not None or st.session_state.failed_id is not None:
            st.session_state.update(
                vectorstore=None,
                doc_id=None,
                file_name=None,
                num_chunks=0,
                messages=[],
                failed_id=None,
            )

    if st.session_state.vectorstore is not None:
        st.success(f"Ready: **{st.session_state.file_name}**")
        st.caption(f"{st.session_state.num_chunks} chunks indexed")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`  \nLLM: `{CHAT_MODEL}`")


# ---------- main chat area ----------
st.title("Chat with your PDF")

if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF in the sidebar to get started.")

# replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"))

query = st.chat_input(
    "Ask a question about the document…",
    disabled=st.session_state.vectorstore is None,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document…"):
                retriever = get_retriever(st.session_state.vectorstore)
                docs = retriever.invoke(query)
                context = "\n\n".join(d.page_content for d in docs)
                final_prompt = PROMPT.invoke({"context": context, "question": query})

            def stream():
                for chunk in get_llm().stream(final_prompt):
                    yield msg_text(chunk)

            answer = st.write_stream(stream())

            sources = [
                {
                    "page": d.metadata.get("page", 0) + 1,  # PyPDF pages are 0-indexed
                    "text": d.page_content[:400].strip() + "…",
                }
                for d in docs
            ]
            show_sources(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")