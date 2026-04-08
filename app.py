import streamlit as st
import json
import re
import os
from PIL import Image
import pytesseract

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HS Code Classifier",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root theme */
:root {
    --bg: #0d0f14;
    --surface: #161a23;
    --surface2: #1e2330;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --text: #e8eaf0;
    --muted: #7a8099;
    --border: #2a2f3d;
    --success: #00c896;
    --warn: #f59e0b;
}

/* Global overrides */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Hide default streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d0f14 0%, #161a23 50%, #1a1040 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,229,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 12px;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 10px 0;
    line-height: 1.2;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-accent {
    border-left: 3px solid var(--accent);
}
.card-success {
    border-left: 3px solid var(--success);
}
.card-warn {
    border-left: 3px solid var(--warn);
}

/* ── Result item ── */
.result-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.result-item:hover { border-color: var(--accent); }
.hs-code {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 2px;
}
.confidence-bar-wrap {
    background: var(--surface);
    border-radius: 4px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
}
.confidence-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 20px;
    background: rgba(0,229,255,0.1);
    color: var(--accent);
    border: 1px solid rgba(0,229,255,0.2);
    margin-right: 6px;
}
.badge-purple {
    background: rgba(124,58,237,0.15);
    color: #a78bfa;
    border-color: rgba(124,58,237,0.3);
}
.badge-green {
    background: rgba(0,200,150,0.1);
    color: var(--success);
    border-color: rgba(0,200,150,0.2);
}

/* ── Metric boxes ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}
.metric-box {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── Input overrides ── */
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important;
}
.stButton button {
    background: linear-gradient(135deg, #00c8ff, #7c3aed) !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(0,229,255,0.25) !important;
}
.stSlider [data-testid="stSlider"] { color: var(--accent); }
.stSelectbox select {
    background: var(--surface2) !important;
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Alert / info */
.stAlert {
    background: var(--surface2) !important;
    border-radius: 10px !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Tab style */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: lazy pipeline loader ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(h6_path, faiss_path, meta_path, alpha):
    from pipelines.pipeline_main import ICCARAGPipeline
    return ICCARAGPipeline(h6_path, faiss_path, meta_path, alpha=alpha)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Space Mono",monospace; font-size:11px;
                letter-spacing:3px; color:#00e5ff; margin-bottom:8px;'>
    CONFIGURATION
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Paths")
    h6_path = st.text_input("H6 JSON path", value="data/H6.json")
    faiss_path = st.text_input("FAISS index path", value="indexing/vector_store/h6.faiss")
    meta_path = st.text_input("Metadata JSON path", value="indexing/vector_store/h6_meta.json")

    st.markdown("---")
    st.markdown("### Retrieval Settings")
    alpha = st.slider(
        "Hybrid Alpha (0 = pure BM25, 1 = pure Vector)",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Controls the blend between sparse (BM25) and dense (vector) retrieval."
    )
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#7a8099; line-height:1.6;'>
    <b style='color:#e8eaf0;'>Ablation Results</b><br>
    BM25 Recall@1: <b style='color:#00e5ff;'>71.0%</b><br>
    Vector Recall@1: <b style='color:#00e5ff;'>72.8%</b><br>
    Best Hybrid α=0.8: <b style='color:#00c896;'>72.4%</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    pipeline_ready = st.button("🔌 Load / Reload Pipeline", use_container_width=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "results" not in st.session_state:
    st.session_state.results = []

if pipeline_ready:
    with st.spinner("Loading models and index…"):
        try:
            st.session_state.pipeline = load_pipeline(h6_path, faiss_path, meta_path, alpha)
            st.success("✅ Pipeline loaded!")
        except Exception as e:
            st.error(f"❌ Failed to load pipeline: {e}")


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-tag'>📦 Hybrid RAG · HS Code Intelligence</div>
    <h1>HS Code Classifier</h1>
    <p>Paste receipt or invoice text — the pipeline cleans, retrieves, and classifies
       each product line into a 6-digit Harmonized System code.</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  CLASSIFY", "📊  ABLATION RESULTS", "ℹ️  HOW IT WORKS"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_out = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("""
        <div class='card card-accent'>
            <div style='font-family:"Space Mono",monospace; font-size:11px;
                        letter-spacing:2px; color:#00e5ff; margin-bottom:12px;'>
            INPUT TEXT
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="Upload receipt/invoice image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to extract text via OCR."
        )

        image_text = ""
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)
            with st.spinner("Extracting text from image…"):
                try:
                    image_text = pytesseract.image_to_string(image)
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    image_text = ""

            if image_text.strip():
                st.success("✅ OCR extracted text from image.")
            else:
                st.warning("⚠️ OCR did not extract text. Please enter receipt text manually.")

        sample_text = """ENGINE OIL 4L
MOBILE PHONE CHARGER FAST 20W
TRACTOR DIESEL ENGINE PART
LED BULB 9W COOL WHITE
AIR FILTER FOR DIESEL ENGINE"""

        invoice_text = st.text_area(
            label="Receipt / Invoice text",
            value=image_text.strip() or sample_text,
            height=220,
            label_visibility="collapsed",
            placeholder="Paste your invoice or receipt text here…"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        run_btn = st.button("⚡ CLASSIFY PRODUCTS", use_container_width=True)

        if st.session_state.pipeline is None:
            st.markdown("""
            <div style='background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
                        border-radius:8px; padding:12px 16px; margin-top:12px;
                        font-size:13px; color:#f59e0b;'>
            ⚠️  Pipeline not loaded — click <b>"Load / Reload Pipeline"</b> in the sidebar first.
            </div>
            """, unsafe_allow_html=True)

    # ── Run classification ──
    if run_btn:
        if st.session_state.pipeline is None:
            st.error("Load the pipeline first using the sidebar button.")
        elif not invoice_text.strip():
            st.warning("Please enter some invoice text.")
        else:
            st.session_state.results = []
            with st.spinner("🔄 Parsing, cleaning & retrieving…"):
                try:
                    results = st.session_state.pipeline.predict(invoice_text, top_k=top_k)
                    st.session_state.results = results
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.session_state.results = []

    # ── Output panel ──
    with col_out:
        results = st.session_state.results
        unique_results = []
        seen = set()
        for item in results:
            key = (item.get("raw_line", "").strip(), item.get("cleaned_line", "").strip())
            if key in seen:
                continue
            seen.add(key)
            unique_results.append(item)
        results = unique_results

        if results:
            n_products = len(results)

            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-box'>
                    <div class='metric-val'>{n_products}</div>
                    <div class='metric-label'>Products Found</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for i, item in enumerate(results, 1):
                pred = item.get("prediction") or "—"
                raw = item.get("raw_line", "")
                cleaned = item.get("cleaned_line", "")
                reasoning = item.get("reasoning", "")
                candidates = item.get("retrieved_candidates", [])

                st.markdown(f"""
                <div class='result-item'>
                    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                        <div>
                            <div style='font-size:12px; color:#7a8099; margin-bottom:4px;'>
                                Product {i}
                            </div>
                            <div style='font-weight:500; margin-bottom:6px;'>{raw[:80]}{"…" if len(raw)>80 else ""}</div>
                            <span class='badge'>Cleaned</span>
                            <span style='font-size:13px; color:#a0aab8;'>{cleaned[:60]}{"…" if len(cleaned)>60 else ""}</span>
                        </div>
                        <div style='text-align:right; min-width:110px;'>
                            <div class='hs-code'>{pred}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"🔎 Candidates & reasoning — Product {i}"):
                    if candidates:
                        st.markdown("**Top retrieved candidates:**")
                        for doc in candidates[:5]:
                            rank = doc.get("rank", "?")
                            code = doc.get("doc_id", "")
                            score = doc.get("score", 0)
                            text = doc.get("text", "")
                            st.markdown(f"""
                            <div style='background:#1e2330; border-radius:8px; padding:10px 14px; margin-bottom:6px;'>
                                <span class='badge'>#{rank}</span>
                                <span style='font-family:"Space Mono",monospace; color:#00e5ff;'>{code}</span>
                                <span style='color:#7a8099; font-size:12px; float:right;'>score: {score:.4f}</span>
                                <div style='font-size:13px; color:#a0aab8; margin-top:4px;'>{text[:120]}{"…" if len(text)>120 else ""}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    if reasoning:
                        st.markdown("**LLM Reasoning:**")
                        st.markdown(f"""
                        <div style='background:#1e2330; border-radius:8px; padding:12px 14px;
                                    font-size:13px; color:#a0aab8; line-height:1.6;'>
                        {reasoning[:400]}{"…" if len(reasoning)>400 else ""}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#7a8099;'>
                <div style='font-size:48px; margin-bottom:16px;'>📦</div>
                <div style='font-size:16px; font-weight:500; color:#e8eaf0; margin-bottom:8px;'>
                    No results yet
                </div>
                <div style='font-size:14px;'>
                    Load the pipeline and click <b style='color:#00e5ff;'>Classify Products</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABLATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='card card-accent'>
        <div style='font-family:"Space Mono",monospace; font-size:11px;
                    letter-spacing:2px; color:#00e5ff; margin-bottom:16px;'>
        ABLATION STUDY — RETRIEVAL CONFIGURATIONS
        </div>
    """, unsafe_allow_html=True)

    ablation_data = [
        {"Configuration": "Pure BM25 (α=0.0)", "Recall@1": 0.710, "Recall@5": 0.744, "MRR": 0.726},
        {"Configuration": "Hybrid (α=0.2)",     "Recall@1": 0.720, "Recall@5": 0.744, "MRR": 0.730},
        {"Configuration": "Hybrid (α=0.5)",     "Recall@1": 0.722, "Recall@5": 0.744, "MRR": 0.732},
        {"Configuration": "Hybrid (α=0.8)",     "Recall@1": 0.724, "Recall@5": 0.744, "MRR": 0.733},
        {"Configuration": "Pure Vector (α=1.0)","Recall@1": 0.728, "Recall@5": 0.748, "MRR": 0.738},
    ]

    for row in ablation_data:
        best = row["Configuration"] == "Pure Vector (α=1.0)"
        border = "var(--success)" if best else "var(--border)"
        st.markdown(f"""
        <div style='background:var(--surface2); border:1px solid {border};
                    border-radius:10px; padding:16px 20px; margin-bottom:10px;
                    display:flex; align-items:center; gap:24px;'>
            <div style='min-width:200px; font-weight:500;'>
                {"🏆 " if best else ""}{row["Configuration"]}
            </div>
            <div style='flex:1;'>
                <div style='font-size:11px; color:#7a8099; margin-bottom:4px;'>Recall@1</div>
                <div style='background:var(--surface); border-radius:4px; height:8px; overflow:hidden;'>
                    <div style='width:{row["Recall@1"]*100:.1f}%;height:100%;
                                background:linear-gradient(90deg,#7c3aed,#00e5ff); border-radius:4px;'></div>
                </div>
                <div style='font-size:12px; color:#00e5ff; margin-top:2px;'>{row["Recall@1"]:.3f}</div>
            </div>
            <div style='flex:1;'>
                <div style='font-size:11px; color:#7a8099; margin-bottom:4px;'>Recall@5</div>
                <div style='background:var(--surface); border-radius:4px; height:8px; overflow:hidden;'>
                    <div style='width:{row["Recall@5"]*100:.1f}%;height:100%;
                                background:linear-gradient(90deg,#7c3aed,#00c896); border-radius:4px;'></div>
                </div>
                <div style='font-size:12px; color:#00c896; margin-top:2px;'>{row["Recall@5"]:.3f}</div>
            </div>
            <div style='flex:1;'>
                <div style='font-size:11px; color:#7a8099; margin-bottom:4px;'>MRR</div>
                <div style='background:var(--surface); border-radius:4px; height:8px; overflow:hidden;'>
                    <div style='width:{row["MRR"]*100:.1f}%;height:100%;
                                background:linear-gradient(90deg,#f59e0b,#ef4444); border-radius:4px;'></div>
                </div>
                <div style='font-size:12px; color:#f59e0b; margin-top:2px;'>{row["MRR"]:.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    steps = [
        ("01", "PARSE", "ReceiptParser splits raw OCR text into candidate product lines, filtering out addresses, totals, and tax IDs."),
        ("02", "CLEAN", "GroqCleaner (LLaMA-3.1 via Groq API) semantically normalises each line into a short trade description."),
        ("03", "RETRIEVE", "HybridRetriever fuses BM25 (sparse) and BGE-M3 + FAISS (dense) scores using a tunable α parameter."),
        ("04", "AUGMENT", "ContextAugmenter formats the top-K retrieved HS code descriptions into a prompt-ready context block."),
        ("05", "GENERATE", "Flan-T5 selects the best 6-digit HS code, outputs a confidence score and a reasoning trace."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex; gap:20px; margin-bottom:20px; align-items:flex-start;'>
            <div style='font-family:"Space Mono",monospace; font-size:1.6rem;
                        font-weight:700; color:rgba(0,229,255,0.2); min-width:40px;'>
                {num}
            </div>
            <div style='background:var(--surface); border:1px solid var(--border);
                        border-radius:10px; padding:18px 20px; flex:1;'>
                <div style='font-family:"Space Mono",monospace; font-size:11px;
                            letter-spacing:2px; color:#00e5ff; margin-bottom:8px;'>{title}</div>
                <div style='color:#a0aab8; font-size:14px; line-height:1.6;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
