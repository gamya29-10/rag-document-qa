import streamlit as st
import os
import re
import numpy as np

st.set_page_config(page_title="Paracetamol RAG QA System", page_icon="@", layout="wide")
st.title("Document based Question Answering System")
st.divider()

# ── Session State ──

    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ──
with st.sidebar:
    st.header("Configuration")
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 100, 10)
    top_k         = st.slider("Top-K Retrieval", 1, 5, 3, 1)
    chunk_size    = st.slider("Chunk Size", 200, 1000, 500, 50)
    show_scores   = st.checkbox("Show Similarity Scores", value=True)



    st.divider()
    st.markdown("### Chunking Strategy")
    st.markdown(f"""
- **chunk_size={chunk_size}**
- **chunk_overlap={chunk_overlap}**
- Split order: paragraph → sentence → word → character
    """)
    st.divider()
    if st.button("🗑 Clear Database"):
        st.session_state.chunks = None
        st.session_state.faiss_index = None
        st.session_state.model = None
        st.success("Database cleared.")
# ── PDF Extraction ──
def extract_pages(file_bytes):
    import pypdf, io
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return [{"page": i+1, "text": (p.extract_text() or "")} for i, p in enumerate(reader.pages)]

# ── Recursive Splitter ──
def recursive_split(text, size, overlap):
    separators = ["\n\n", "\n", ". ", " ", ""]
    def _split(t, seps):
        if not t.strip(): return []
        if len(t) <= size: return [t.strip()]
        sep = seps[0] if seps else ""
        parts = t.split(sep) if sep else list(t)

        chunks, buf, buf_len = [], [], 0
        for p in parts:
            if buf_len + len(p) > size and buf:
                chunks.append(sep.join(buf))
                while buf and buf_len > overlap:
                    buf_len -= len(buf.pop(0)) + len(sep)
            buf.append(p)
            buf_len += len(p) + len(sep)

        if buf:
            chunks.append(sep.join(buf))

        result = []
        for c in chunks:
            if len(c) > size and len(seps) > 1:
                result.extend(_split(c, seps[1:]))
            elif c.strip():
                result.append(c.strip())
        return result

    return _split(text, separators)

# ── Step 1: Upload ──
st.header("Step 1: Upload PDF Documents")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing and chunking documents..."):
        chunks = []
        total_pages = 0

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            pages = extract_pages(file_bytes)
            total_pages += len(pages)

            for p in pages:
                for c in recursive_split(p["text"], chunk_size, chunk_overlap):
                    if len(c) > 30:
                        chunks.append({
                            "source": uploaded_file.name,
                            "page": p["page"],
                            "text": c
                        })

        st.session_state.chunks = chunks

    st.success(f"{len(uploaded_files)} PDFs | {total_pages} total pages → {len(chunks)} chunks created")

    # ── Embeddings + FAISS ──
    with st.spinner("Creating embeddings..."):
        from sentence_transformers import SentenceTransformer
        import faiss

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [c["text"] for c in chunks]
        embs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embs  = np.array(embs, dtype="float32")

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        st.session_state.faiss_index = index
        st.session_state.model = model

    st.success("Embeddings stored in FAISS vector database.")

    with st.expander("Preview First 3 Chunks"):
        for i, c in enumerate(chunks[:3]):
            st.markdown(f"**Chunk {i+1}** — {c['source']} | Page {c['page']}")
            st.text(c["text"][:300] + ("…" if len(c["text"]) > 300 else ""))
            st.divider()

# ── Step 2: QA ──
st.header("Step 2: Ask a Question")

user_query = st.text_input("Enter your question:")

if st.button("Get Answer", type="primary") and user_query:
    if not st.session_state.chunks:
        st.warning("Upload PDF(s) first.")
    else:
        with st.spinner("Searching and generating answer..."):
            model  = st.session_state.model
            index  = st.session_state.faiss_index
            chunks = st.session_state.chunks

            q_vec = np.array(model.encode([user_query], normalize_embeddings=True), dtype="float32")
            scores, idxs = index.search(q_vec, top_k)

            retrieved = [(chunks[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]

        context   = " ".join([r[0]["text"] for r in retrieved])
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if len(s.strip()) > 30]
        keywords  = [w.lower() for w in re.findall(r'\w+', user_query) if len(w) > 3]

        scored = sorted(
            [(sum(1 for k in keywords if k in s.lower()), s) for s in sentences],
            reverse=True
        )

        top_sents = [s for sc, s in scored if sc > 0][:4]
        answer    = " ".join(top_sents) if top_sents else context[:600]

        st.subheader("Answer")
        st.markdown(
            f'<div style="background:#f0f8ff;padding:16px;border-radius:10px;border-left:4px solid #1e90ff">{answer}</div>',
            unsafe_allow_html=True
        )

        st.subheader(f"Top-{top_k} Retrieved Source Chunks")
        for i, (chunk, score) in enumerate(retrieved):
            label = f"Chunk {i+1} — {chunk['source']} | Page {chunk['page']}"
            if show_scores:
                label += f"  |  Score: {score:.4f}"

            with st.expander(label):
                st.write(chunk["text"])