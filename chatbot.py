import streamlit as st
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from difflib import SequenceMatcher

# ---------- CONFIG ----------
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-large"
TOP_K = 3
THRESHOLD = 1.5
# ----------------------------

# ---------- Page Setup ----------
st.set_page_config(page_title="Conversational Chatbot", layout="wide")

st.markdown(
    """
    <h1 style="text-align: center; color: #f5f5f5;">
        💬 Conversational Document Chatbot
    </h1>
    <p style="text-align: center; color: #cfcfcf;">
        Ask questions about the User Manual.
    </p>
    """,
    unsafe_allow_html=True,
)

# Control buttons with alignment
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()
with col3:
    if st.button("❌ Exit Chatbot", use_container_width=True):
        st.stop()

# ---------- Custom Chat Bubble Styling ----------
st.markdown(
    """
    <style>
    /* User message bubble */
    .user-bubble {
        background-color: #2b5278;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 75%;
        float: right;
        clear: both;
    }

    /* Assistant message bubble */
    .bot-bubble {
        background-color: #3c3c3c;
        color: #f5f5f5;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 75%;
        float: left;
        clear: both;
    }

    /* Chat area cleanup */
    .chat-container {
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Utilities ----------

@st.cache_resource
def load_resources():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    gen_pipeline = pipeline("text2text-generation", model=GEN_MODEL)
    return index, chunks, model, gen_pipeline

index, chunks, model, gen_pipeline = load_resources()

def expand_query(query: str) -> str:
    q = query.lower()
    q = q.replace("leaked", "leak")
    q = q.replace("insider", "inside")
    q = q.replace("water", "flood")
    return q

def expand_query_with_synonyms(query: str) -> list:
    expansions = [query]
    if "safety" in query.lower():
        expansions += ["precautions", "guidelines", "protective measures"]
    if "water" in query.lower() and "leak" in query.lower():
        expansions += ["flood sensors", "liquid detected"]
    if "alarm" in query.lower():
        expansions += ["fire alarm", "emergency stop"]
    return expansions

def is_toc_like(text: str) -> bool:
    words = text.split()
    if not words:
        return True
    numeric_ratio = sum(w.isdigit() or w.replace(".", "").isdigit() for w in words) / len(words)
    return numeric_ratio > 0.3 and len(words) < 30

def retrieve(query, index, chunks, model, top_k=TOP_K, threshold=THRESHOLD):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        score = float(distances[0][i])
        if score < threshold:
            chunk = chunks[idx]
            if not is_toc_like(chunk):
                results.append({"chunk": chunk, "distance": score})
    return results

def clean_answer(text: str) -> str:
    bad_phrases = [
        "Answer the question", "Answer in 1–2 sentences",
        "Use only the manual context", "based only on the manual",
        "If you are a helpful assistant"
    ]
    for phrase in bad_phrases:
        text = text.replace(phrase, "")
    return text.strip()

def deduplicate_sentences_fuzzy(text: str, threshold: float = 0.8) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
    unique = []
    for s in sentences:
        if not any(SequenceMatcher(None, s, u).ratio() > threshold for u in unique):
            unique.append(s)
    return " ".join(unique)

def remove_repetitions(text: str) -> str:
    words = text.split()
    cleaned, prev = [], None
    for w in words:
        if w == prev:
            continue
        cleaned.append(w)
        prev = w
    return " ".join(cleaned)

def trim_incomplete(text: str) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text)
    if not text.endswith(('.', '?', '!')) and len(sentences) > 1:
        return " ".join(sentences[:-1]).strip()
    return text

def remove_toc_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned = [ln for ln in lines if not re.match(r'^\d+(\.\d+)*\s', ln.strip())]
    return "\n".join(cleaned)

def format_answer(answer: str) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', answer) if s.strip()]
    if len(sentences) > 3:
        bullets = "\n".join([f"- {s}" for s in sentences])
        return bullets
    return answer

def get_dynamic_max_tokens(query: str, context_text: str) -> tuple:
    query_len = len(query.split())
    context_len = len(context_text.split())
    base = min(350, int(context_len * 1.5))
    if query_len < 4:
        return int(base * 0.3), min(150, base + 50)
    elif query_len > 12:
        return int(base * 0.6), min(450, base + 100)
    else:
        return int(base * 0.5), min(300, base + 50)

# ---------- Chatbot Logic ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history with chat bubbles
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if query := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="user-bubble">{query}</div>', unsafe_allow_html=True)

    chitchat = ["how are you", "hello", "hi", "what's up"]
    if query.lower().strip() in chitchat:
        answer = "👋 I’m here to help with the manual. Could you ask me something about the system?"
    else:
        expanded = expand_query(query)
        results = []
        for q in expand_query_with_synonyms(expanded):
            results = retrieve(q, index, chunks, model, top_k=TOP_K, threshold=THRESHOLD)
            if results:
                break

        if not results:
            answer = "I couldn’t find that in the manual. Please contact support at support@ampd.energy."
        else:
            context_text = " ".join([res["chunk"] for res in results])
            prompt = f"""
            Context: {context_text}
            Question: {query}
            Provide a clear and helpful answer:
            """
            min_len, max_tokens = get_dynamic_max_tokens(query, context_text)
            with st.spinner("🤔 Thinking..."):
                result = gen_pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    min_length=min_len,
                    do_sample=False
                )
            raw_answer = result[0].get("generated_text", "").strip()
            answer = clean_answer(raw_answer)
            answer = deduplicate_sentences_fuzzy(answer)
            answer = remove_repetitions(answer)
            answer = trim_incomplete(answer)
            answer = remove_toc_lines(answer)
            answer = format_answer(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown(f'<div class="bot-bubble">{answer}</div>', unsafe_allow_html=True)
