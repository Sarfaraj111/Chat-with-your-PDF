# 📄 Chat with your PDF

A Streamlit app that lets you upload any PDF and ask questions about it. It uses **Retrieval-Augmented Generation (RAG)**, so answers come only from your document, and every answer shows the pages it was drawn from.

## ✨ Features

- **Upload any PDF** from the sidebar and start chatting
- **Grounded answers**: responds only from the document, and says so when the answer isn't there
- **Source citations**: page numbers and text snippets for every answer
- **Streaming responses** in a familiar chat interface
- **Separate index per PDF**: each file gets its own vector store, and re-uploading the same file reuses it instantly
- **Rate-limit friendly**: chunks are embedded in small batches with automatic retry, and an interrupted run resumes where it stopped

## 🧠 How it works

```
PDF → Split into chunks → Embed → Store in ChromaDB
                                        │
Question → Embed → Retrieve top chunks (MMR) → Gemini → Answer + sources
```

1. **Load** the PDF with `PyPDFLoader`
2. **Split** it into 1000-character chunks with 200 characters of overlap
3. **Embed** the chunks with Google Gemini embeddings
4. **Store** the vectors in a local ChromaDB database (`chroma_db/<file-hash>/`)
5. **Retrieve** the most relevant, diverse chunks for each question using MMR search
6. **Generate** an answer with Gemini using only the retrieved context

## 🛠️ Tech stack

Python · Streamlit · LangChain · ChromaDB · Google Gemini

## 🚀 Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Sarfaraj111/Chat-with-your-PDF.git
cd Chat-with-your-PDF
```

### 2. Install dependencies

Using a virtual environment is recommended:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

You can get a key from [Google AI Studio](https://aistudio.google.com/apikey). The `.env` file is git-ignored, so your key is never committed.

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`), upload a PDF in the sidebar, and ask away.

## 📁 Project structure

```
.
├── app.py              # Streamlit app (indexing + chat UI)
├── requirements.txt    # Python dependencies
├── .gitignore          # Keeps .env and chroma_db out of git
├── .env                # Your API key (not committed)
└── chroma_db/          # Generated vector stores (not committed)
```

## ⚙️ Configuration

Settings live at the top of `app.py`:

| Setting | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Model used to embed chunks and questions |
| `CHAT_MODEL` | `gemini-3.6-flash` | Model used to generate answers |
| `BATCH_SIZE` | `10` | Chunks embedded per API request |
| `PAUSE_BETWEEN_BATCHES` | `1.0` | Seconds to wait between requests |
| `MAX_RETRIES` | `6` | Retries per batch when rate-limited |

Chunk size and overlap are set in `build_vectorstore()`, and the retrieval settings (`k`, `fetch_k`, `lambda_mult`) are in `get_retriever()`.

## 🩺 Troubleshooting

**`429 RESOURCE_EXHAUSTED` while processing a PDF**
You've hit Google's embedding quota. Lower `BATCH_SIZE`, raise `PAUSE_BETWEEN_BATCHES`, or wait a minute and click **Retry**. Progress is saved, so it resumes instead of starting over. If it keeps failing, check your quota in Google AI Studio or try a different embedding model.

**"No text could be extracted"**
The PDF is a scan or image-only file. Text extraction needs OCR, which isn't included.

**Blank page in the browser**
Check the terminal running `streamlit run` for the actual error. Most often it's a missing package, so re-run `pip install -r requirements.txt` in the same environment.

## ⚠️ Limitations

- Scanned or image-only PDFs aren't supported (no OCR)
- Each question is answered independently, so the app doesn't use earlier chat turns as context
- Very large PDFs take longer to index because embedding is rate-limited

## 🔮 Ideas for the future

- Conversation memory for follow-up questions
- OCR support for scanned PDFs
- Multiple PDFs in one chat
- Support for other file types (DOCX, TXT, web pages)

## 📄 License

Add a license of your choice (for example MIT) if you'd like others to reuse this project.
