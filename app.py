# City Explorer Tours — Hybrid RAG Chatbot
# Offline retrieval + optional LLM synthesis (OpenAI), vector DB (Chroma), secure secrets,
# caching, rate limiting, logging, admin dashboard, and professional UI.

import os
import time
import uuid
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import numpy as np

# Optional OpenAI (used only if key is present)
try:
    import openai
except Exception:
    openai = None

# Vector DB + local embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="City Explorer Tours", page_icon="🗺️", layout="wide")

# ==================== STYLING (PROFESSIONAL & COLORFUL) ====================
st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
  color: #f2f6f9;
}
.card {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 14px;
  box-shadow: 0 10px 20px rgba(0,0,0,0.15);
}
.hero-title { font-size: 36px; font-weight: 700; letter-spacing: 0.3px; color: #eaf2f8; text-align: center; margin-top: 10px; }
.hero-sub { font-size: 18px; color: #d6e6f2; text-align: center; margin-bottom: 24px; }
.stTextInput > div > input {
  border: 2px solid #50c9c3;
  border-radius: 10px;
  padding: 10px 12px;
  background-color: rgba(255,255,255,0.9);
  color: #0b1f29;
}
.stForm button[type="submit"] {
  background: linear-gradient(135deg, #50c9c3, #96deda);
  color: #0b1f29;
  font-weight: 700;
  border-radius: 10px;
  padding: 10px 18px;
  border: none;
  box-shadow: 0 6px 12px rgba(0,0,0,0.2);
}
.stForm button[type="submit"]:hover { filter: brightness(1.05); }
.answer {
  background: rgba(255,255,255,0.12);
  border-left: 4px solid #96deda;
  border-radius: 12px;
  padding: 14px 16px;
}
.footer { text-align: center; color: #cfe4ed; margin-top: 18px; }
.stSidebar { background: rgba(15,32,39,0.55) !important; }
.badge { display: inline-block; padding: 4px 8px; border-radius: 999px; background: #96deda; color: #0b1f29; margin-right: 6px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🗺️ City Explorer Tours</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Hybrid RAG bot: Offline retrieval + optional LLM. Ask about cities, packages, schedules, prices, booking, and policies.</div>', unsafe_allow_html=True)

# ==================== CONFIG & SECRETS ====================
def get_api_key() -> str:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")

OPENAI_API_KEY = get_api_key()
if openai and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Caching and rate limits
CACHE_TTL = 600  # seconds
RATE_LIMIT = 10
RATE_LIMIT_WINDOW = 60  # seconds
DEBOUNCE_SECONDS = 3

# ==================== LOAD CATALOG ====================
CATALOG_PATH = Path(__file__).parent / "tours.json"

def load_catalog() -> Dict[str, Any]:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

try:
    catalog = load_catalog()
except Exception as e:
    st.error(f"Could not load tours.json: {e}")
    st.stop()

# ==================== SESSION STATE ====================
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}
if "logs" not in st.session_state:
    st.session_state.logs = []
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "last_call_ts" not in st.session_state:
    st.session_state.last_call_ts = 0.0
if "rate_limit_tracker" not in st.session_state:
    st.session_state.rate_limit_tracker = []

def cache_key(q: str) -> str:
    return f"q::{hash(q.strip().lower())}"

def log_event(kind: str, **kwargs):
    st.session_state.logs.append({"time": datetime.utcnow().isoformat(), "kind": kind, **kwargs})

def check_rate_limit() -> bool:
    now = time.time()
    st.session_state.rate_limit_tracker = [t for t in st.session_state.rate_limit_tracker if now - t <= RATE_LIMIT_WINDOW]
    if len(st.session_state.rate_limit_tracker) >= RATE_LIMIT:
        return False
    st.session_state.rate_limit_tracker.append(now)
    return True

# ==================== VECTOR DB & EMBEDDINGS ====================
# Persistent local Chroma (offline)
PERSIST_DIR = str(Path(__file__).parent / ".chroma")
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR))

# Use a local embedding model (no API)
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = get_embedder()

def embed_texts(texts: List[str]) -> List[List[float]]:
    embs = embedder.encode(texts, normalize_embeddings=True)
    return [list(map(float, v)) for v in embs]  # JSON-serializable

# Build a single collection for all chunks
collection_name = "city_explorer_tours"
try:
    collection = client.get_collection(collection_name)
except Exception:
    collection = client.create_collection(collection_name)

def build_chunks(cat: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    # Contact & policies
    chunks.append({"id": "contact", "text": json.dumps(cat.get("contact", {}), ensure_ascii=False), "meta": {"type": "contact"}})
    for key, val in cat.get("policies", {}).items():
        chunks.append({"id": f"policy_{key}", "text": f"{key}: {val}", "meta": {"type": "policy", "key": key}})
    # Cities & packages
    for city in cat.get("cities", []):
        city_meta = {"type": "city", "city": city["name"]}
        chunks.append({"id": f"city_{city['name']}", "text": f"City: {city['name']}", "meta": city_meta})
        for pkg in city.get("packages", []):
            text = (
                f"City: {city['name']}\n"
                f"Package: {pkg.get('title')} ({pkg.get('code','')})\n"
                f"Price: ${pkg.get('price')}\n"
                f"Duration: {pkg.get('duration')}\n"
                f"Schedule: {pkg.get('schedule')}\n"
                f"Includes: {', '.join(pkg.get('includes', []))}\n"
                f"Notes: {pkg.get('notes','')}"
            )
            chunks.append({"id": f"pkg_{city['name']}_{pkg.get('code','') or pkg.get('title')}", "text": text,
                           "meta": {"type": "package", "city": city["name"], "code": pkg.get("code",""), "title": pkg.get("title","")}})
    # FAQs
    for i, faq in enumerate(cat.get("faqs", [])):
        chunks.append({"id": f"faq_{i}", "text": f"Q: {faq['q']}\nA: {faq['a']}", "meta": {"type": "faq"}})
    return chunks

# Populate collection (idempotent: checks ids)
def populate_collection():
    chunks = build_chunks(catalog)
    existing_ids = set()
    try:
        existing = collection.get()
        existing_ids = set(existing["ids"])
    except Exception:
        pass
    new_texts, new_ids, new_metas = [], [], []
    for ch in chunks:
        if ch["id"] not in existing_ids:
            new_ids.append(ch["id"])
            new_texts.append(ch["text"])
            new_metas.append(ch["meta"])
    if new_ids:
        embs = embed_texts(new_texts)
        collection.add(documents=new_texts, metadatas=new_metas, ids=new_ids, embeddings=embs)
        client.persist()

populate_collection()

def retrieve(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=n_results)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    return [{"id": i, "text": d, "meta": m} for i, d, m in zip(ids, docs, metas)]

# ==================== ANSWER SYNTHESIS ====================
SYSTEM_PROMPT = """You are City Explorer Tours' helpful assistant.
Use only the provided context (catalog, packages, policies, FAQs).
Be clear, concise, and friendly. Include package codes when relevant.
Offer next steps (booking, contact). If info isn't in context, say so.
Limit responses to 4–8 sentences unless listing package details."""

def build_prompt(user_q: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    ctx_text = "\n\n".join([c["text"] for c in contexts])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{user_q}\n\nContext:\n{ctx_text}"}
    ]

def answer_offline(contexts: List[Dict[str, Any]], user_q: str) -> str:
    # Simple heuristic: show best package details or policy/FAQ answers
    text_blocks = []
    # Policy hit
    for c in contexts:
        if c["meta"].get("type") == "policy":
            text_blocks.append(c["text"])
    # FAQ hit
    for c in contexts:
        if c["meta"].get("type") == "faq":
            text_blocks.append(c["text"])
    # Package summaries
    for c in contexts:
        if c["meta"].get("type") == "package":
            text_blocks.append(c["text"])

    if text_blocks:
        answer = "Here’s what I found:\n\n" + "\n\n".join(text_blocks[:3])
        return answer + "\n\nFor bookings or more details, contact: " + json.dumps(catalog["contact"])
    return ("I couldn’t find a precise match in our catalog. Try referencing a city or package code/title. "
            "You can also ask about refunds, weather, or booking.")

def answer_with_openai(contexts: List[Dict[str, Any]], user_q: str) -> str:
    if not (openai and OPENAI_API_KEY and OPENAI_API_KEY.startswith(("sk-", "sk-proj-"))):
        return answer_offline(contexts, user_q)
    messages = build_prompt(user_q, contexts)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        return resp.choices[0].message["content"]
    except openai.error.RateLimitError:
        return answer_offline(contexts, user_q)
    except openai.error.AuthenticationError:
        return answer_offline(contexts, user_q)
    except Exception:
        return answer_offline(contexts, user_q)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title(catalog["company"])
    st.caption("Hybrid RAG bot (offline + optional LLM).")
    st.divider()
    st.subheader("Contact")
    st.write(catalog["contact"]["address"])
    st.write(f"{catalog['contact']['phone']} • {catalog['contact']['email']}")
    if catalog["contact"].get("website"):
        st.write(catalog["contact"]["website"])
    st.divider()
    ai_on = st.toggle("AI synthesis (OpenAI)", value=bool(OPENAI_API_KEY))
    st.caption("Turn off to run purely offline.")
    mode = st.radio("Mode", ["Chat", "Admin"], horizontal=True)

# ==================== CHAT MODE ====================
def chat_mode():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Ask about our tours")
    st.write("Cities, packages, schedules, prices, booking, and policies.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input("Your question", placeholder="e.g., Price and schedule for Golden Gate Highlights?")
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        # Debounce
        now = time.time()
        if now - st.session_state.last_call_ts < DEBOUNCE_SECONDS:
            st.info("Please wait a few seconds before asking again.")
            return
        st.session_state.last_call_ts = now

        # Rate limit
        if not check_rate_limit():
            st.warning("Too many requests. Please wait a minute and try again.")
            return

        key = cache_key(question)
        cached = st.session_state.response_cache.get(key)
        if cached and (time.time() - cached["time"] <= CACHE_TTL):
            st.success("Answer served from cache")
            st.markdown(f'<div class="answer">{cached["content"]}</div>', unsafe_allow_html=True)
            return

        with st.spinner("Searching our catalog..."):
            contexts = retrieve(question, n_results=6)
            # Answer
            content = answer_with_openai(contexts, question) if ai_on else answer_offline(contexts, question)
            st.markdown(f'<div class="answer">{content}</div>', unsafe_allow_html=True)
            st.session_state.response_cache[key] = {"content": content, "time": time.time()}
            log_event("answer", question=question, ai=ai_on, hits=len(contexts))

    # Showcase highlights
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Popular packages")
    for city in catalog["cities"]:
        for pkg in city["packages"][:2]:
            st.write(f"• {city['name']} — {pkg['title']} ({pkg.get('code','')}), {pkg['duration']}, ${pkg['price']}")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== ADMIN MODE ====================
def admin_mode():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Admin dashboard")
    logs = st.session_state.logs
    total_answers = sum(1 for l in logs if l["kind"] == "answer")
    latencies = []  # placeholder if you add timing metrics
    avg_latency = f"{(sum(latencies) / len(latencies)):.0f}" if latencies else "n/a"
    st.metric("Total answers", total_answers)
    st.metric("Average latency (ms)", avg_latency)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Logs")
    if logs: st.json(logs)
    else: st.write("No logs yet.")
    st.download_button(
        "Download logs (JSON)",
        data=json.dumps(logs, indent=2),
        file_name=f"city_explorer_logs_{datetime.utcnow().date()}.json",
        mime="application/json"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== RENDER ====================
if mode == "Chat":
    chat_mode()
else:
    admin_mode()

# ==================== FOOTER ====================
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown(
    f"**{catalog['company']}** • {catalog['contact']['address']} • "
    f"{catalog['contact']['phone']} • {catalog['contact']['email']}"
)
st.markdown('</div>', unsafe_allow_html=True)
