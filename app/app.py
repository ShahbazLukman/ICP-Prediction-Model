import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from predict import predict_lead, load_model, FEATURES


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ICP Insight AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --bg: #080B12;
    --card: #111827;
    --surface: #161F2E;
    --accent: #6C63FF;
    --cyan: #22D3EE;
    --success: #22C55E;
    --warning: #FBBF24;
    --hot: #FF5A5F;
    --text: #F8FAFC;
    --muted: #94A3B8;
    --border: rgba(148,163,184,.14);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% -10%, rgba(108,99,255,.14), transparent 28%),
        radial-gradient(circle at 0% 20%, rgba(34,211,238,.06), transparent 25%),
        var(--bg);
    color: var(--text);
}

/* ------------------------------------------------------------
   Top header: fully transparent (no visible strip), but keep the
   sidebar open/close control usable and clearly visible.
------------------------------------------------------------ */
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stDecoration"] { display: none !important; } /* removes the rainbow accent line */
[data-testid="stToolbar"] { display: none; }

/* Sidebar OPEN control (visible when sidebar is collapsed) */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    top: 14px;
    left: 14px;
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 25px rgba(0,0,0,.3);
    z-index: 999999 !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
    fill: var(--text) !important;
    stroke: var(--text) !important;
}

/* Sidebar CLOSE control (visible inside the open sidebar) */
[data-testid="stSidebarCollapseButton"] button svg,
section[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] svg {
    fill: var(--text) !important;
    stroke: var(--text) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.025em;
}

p, label, span, div { color: inherit; }

.block-container {
    max-width: 1500px;
    padding: 2rem 3rem 6rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(11,16,27,.96);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div { padding: 1.25rem .9rem; }

.nav-brand {
    display:flex; align-items:center; gap:12px; padding: 8px 10px 22px;
}
.brand-mark {
    width:38px; height:38px; border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg, #6C63FF, #22D3EE);
    color:#fff; font-weight:800; font-family:Manrope;
    box-shadow: 0 10px 30px rgba(108,99,255,.25);
}
.brand-name { font-family:Manrope; font-weight:800; font-size:17px; color:#F8FAFC; }
.brand-sub { color:#64748B; font-size:11px; margin-top:2px; }

/* Cards */
.card {
    background: linear-gradient(180deg, rgba(17,24,39,.96), rgba(17,24,39,.82));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 16px 40px rgba(0,0,0,.18);
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.card:hover { transform: translateY(-2px); border-color: rgba(108,99,255,.35); box-shadow:0 20px 50px rgba(0,0,0,.25); }
.card-title { font-family:Manrope; font-weight:700; color:#F8FAFC; font-size:15px; }
.card-sub { color:#64748B; font-size:12px; margin-top:4px; }

.metric-value { font-family:Manrope; font-size:30px; font-weight:800; margin-top:13px; }
.metric-label { color:#94A3B8; font-size:12px; }
.metric-delta { font-size:11px; margin-top:8px; color:#22C55E; }

.hero {
    border:1px solid rgba(108,99,255,.22);
    border-radius:22px;
    padding:30px;
    background: radial-gradient(circle at 90% 10%, rgba(108,99,255,.18), transparent 32%),
                linear-gradient(135deg, rgba(17,24,39,.98), rgba(15,23,42,.9));
    box-shadow:0 24px 70px rgba(0,0,0,.25);
}
.eyebrow { color:#8B83FF; font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.hero-title { font-family:Manrope; font-size:36px; line-height:1.08; font-weight:800; margin:8px 0 10px; }
.hero-copy { color:#94A3B8; max-width:700px; font-size:14px; line-height:1.7; }

.score-ring {
    width:190px; height:190px; border-radius:50%;
    margin:auto;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    background: radial-gradient(circle, #111827 57%, transparent 58%),
                conic-gradient(#6C63FF 0deg, #22D3EE 260deg, #243047 260deg 360deg);
    box-shadow: 0 0 50px rgba(108,99,255,.18);
}
.score-number { font-family:Manrope; font-size:42px; font-weight:800; color:#F8FAFC; }
.score-label { color:#94A3B8; font-size:11px; }

.badge { display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; font-size:11px; font-weight:700; }
.badge-hot { background:rgba(255,90,95,.12); color:#FF7B7F; border:1px solid rgba(255,90,95,.2); }
.badge-warm { background:rgba(251,191,36,.10); color:#FBBF24; border:1px solid rgba(251,191,36,.2); }
.badge-medium { background:rgba(108,99,255,.12); color:#A9A3FF; border:1px solid rgba(108,99,255,.2); }
.badge-cold { background:rgba(100,116,139,.12); color:#CBD5E1; border:1px solid rgba(100,116,139,.2); }

.insight {
    border:1px solid rgba(34,211,238,.16);
    background:linear-gradient(135deg, rgba(34,211,238,.06), rgba(108,99,255,.05));
    border-radius:16px; padding:18px;
}
.recommendation { border-left:3px solid #6C63FF; background:#161F2E; border-radius:14px; padding:17px 18px; }

.stepper { display:flex; gap:8px; margin:8px 0 25px; }
.step { flex:1; border:1px solid var(--border); background:#111827; border-radius:12px; padding:11px 12px; }
.step.active { border-color:rgba(108,99,255,.65); background:rgba(108,99,255,.10); }
.step.done { border-color:rgba(34,197,94,.35); }
.step-num { color:#64748B; font-size:10px; font-weight:700; }
.step-name { color:#F8FAFC; font-size:12px; margin-top:4px; font-weight:600; }

.toggle-card { border:1px solid var(--border); border-radius:15px; padding:15px; background:#111827; }
.small-muted { color:#64748B; font-size:11px; }

[data-testid="stMetric"] {
    background: #111827; border:1px solid var(--border); padding:18px; border-radius:16px;
}
[data-testid="stMetricLabel"] { color:#94A3B8 !important; }
[data-testid="stMetricValue"] { color:#F8FAFC !important; font-family:Manrope; }

button[kind="primary"] {
    background:linear-gradient(135deg,#6C63FF,#5A52E6) !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-radius:12px !important;
    min-height:44px;
    box-shadow:0 10px 25px rgba(108,99,255,.18);
}
button[kind="secondary"] { border-radius:12px !important; min-height:44px; }

input, textarea, [data-baseweb="select"] > div {
    background:#0F172A !important; border-color:rgba(148,163,184,.16) !important;
    border-radius:11px !important; color:#F8FAFC !important;
}

[data-testid="stFileUploaderDropzone"] {
    background:#0F172A !important; border:1px dashed rgba(108,99,255,.45) !important;
    border-radius:18px !important; padding:28px !important;
}

[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:16px; overflow:hidden; }

hr { border-color:var(--border); }

.mobile-only { display:none; }

@media (max-width: 900px) {
    .block-container { padding: 1.2rem 1rem 6rem; }
    .hero-title { font-size:28px; }
    .score-ring { width:160px; height:160px; }
    .score-number { font-size:34px; }
    .step-name { font-size:10px; }
    section[data-testid="stSidebar"] { min-width:250px; }
}

@media (max-width: 640px) {
    .block-container { padding: .8rem .7rem 6.5rem; }
    .card, .hero { padding:17px; border-radius:16px; }
    .metric-value { font-size:24px; }
    .step { padding:9px 7px; }
    .step-name { display:none; }
    .mobile-only { display:block; }
    [data-testid="stHorizontalBlock"] { gap: .55rem; }
    button { min-height:44px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR TOGGLE (works regardless of Streamlit-version internals)
# ============================================================

components.html(
    """
    <script>
    (function() {
        const doc = window.parent.document;
        if (doc.getElementById('icp-sidebar-toggle')) return;

        const btn = doc.createElement('button');
        btn.id = 'icp-sidebar-toggle';
        btn.innerHTML = '☰';
        btn.title = 'Toggle sidebar';
        Object.assign(btn.style, {
            position: 'fixed',
            top: '14px',
            left: '14px',
            zIndex: 999999,
            width: '42px',
            height: '42px',
            borderRadius: '12px',
            border: '1px solid rgba(148,163,184,.25)',
            background: '#111827',
            color: '#F8FAFC',
            fontSize: '18px',
            cursor: 'pointer',
            boxShadow: '0 10px 25px rgba(0,0,0,.35)',
        });

        btn.onclick = function() {
            const selectors = [
                '[data-testid="stSidebarCollapsedControl"] button',
                '[data-testid="stSidebarCollapsedControl"]',
                '[data-testid="collapsedControl"] button',
                '[data-testid="collapsedControl"]',
                'button[aria-label="Open sidebar"]',
                'button[aria-label="Close sidebar"]',
                '[data-testid="stSidebarCollapseButton"] button',
                '[data-testid="baseButton-headerNoPadding"]',
            ];
            for (const sel of selectors) {
                const el = doc.querySelector(sel);
                if (el) { el.click(); return; }
            }
            // Last-resort manual fallback if no native control is found at all
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            if (sidebar) {
                const isHidden = sidebar.style.width === '0px' || sidebar.style.display === 'none';
                sidebar.style.display = isHidden ? '' : 'none';
            }
        };

        doc.body.appendChild(btn);
    })();
    </script>
    """,
    height=0,
)


# ============================================================
# STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "batch" not in st.session_state:
    st.session_state.batch = None


# ============================================================
# HELPERS
# ============================================================


def category(score):
    if score >= 80:
        return "Hot"
    if score >= 60:
        return "Warm"
    if score >= 40:
        return "Medium"
    return "Cold"


def badge_html(cat):
    cls = {"Hot": "badge-hot", "Warm": "badge-warm", "Medium": "badge-medium", "Cold": "badge-cold"}[cat]
    icon = {"Hot": "🔥", "Warm": "◉", "Medium": "◐", "Cold": "○"}[cat]
    return f'<span class="badge {cls}">{icon} {cat}</span>'


def action_for(cat):
    return {
        "Hot": "Contact within 24 hours",
        "Warm": "Personalized follow-up",
        "Medium": "Add to nurture campaign",
        "Cold": "Low-cost automated nurturing",
    }[cat]






def render_metric_cards(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.markdown(
                f'''<div class="card"><div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta">{delta}</div></div>''',
                unsafe_allow_html=True,
            )


def lead_from_state():
    return {
        "age": st.session_state.age,
        "education": st.session_state.education,
        "occupation": st.session_state.occupation,
        "city": st.session_state.city,
        "course_enquiry": st.session_state.course,
        "lead_source": st.session_state.source,
        "website_visits": st.session_state.website_visits,
        "course_page_views": st.session_state.course_page_views,
        "pricing_page_views": st.session_state.pricing_page_views,
        "brochure_downloaded": int(st.session_state.brochure),
        "email_opens": st.session_state.email_opens,
        "email_clicks": st.session_state.email_clicks,
        "whatsapp_responses": st.session_state.whatsapp_responses,
        "calls_answered": st.session_state.calls_answered,
        "followup_count": st.session_state.followup_count,
        "webinar_attended": int(st.session_state.webinar),
        "demo_attended": int(st.session_state.demo),
        "video_watches": st.session_state.video_watches,
        "days_since_last_activity": st.session_state.days_since_last_activity,
    }




# ============================================================
# MODEL CHECK
# ============================================================

MODEL_PATH = ROOT / "models" / "purchase_model.joblib"
if not MODEL_PATH.exists():
    st.error("Model not found. Train the model first with `python src/train.py`.")
    st.stop()


# ============================================================
# NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        '''<div class="nav-brand"><div class="brand-mark">◈</div>
        <div><div class="brand-name">ICP Insight AI</div><div class="brand-sub">Customer Intelligence Platform</div></div></div>''',
        unsafe_allow_html=True,
    )
    st.caption("INTELLIGENCE")
    nav_items = ["Dashboard", "Batch Analysis", "Analytics", "Data Guide"]
    for item in nav_items:
        if st.button(item, key=f"nav_{item}", use_container_width=True):
            st.session_state.page = item
            st.rerun()
    st.divider()
    st.markdown("<div class='small-muted'>MODEL STATUS</div><div style='margin-top:6px'>🟢 Online · Purchase Prediction</div>", unsafe_allow_html=True)


# Mobile-style compact navigation at bottom. Streamlit's sidebar remains available as a drawer.
if st.session_state.page in ["Dashboard", "Batch Analysis", "Analytics"]:
    st.markdown("<div class='mobile-only'><div class='small-muted'>Quick navigation</div></div>", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

h1, h2 = st.columns([4, 1])
with h1:
    st.markdown("<div class='eyebrow'>AI-POWERED CUSTOMER INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:Manrope;font-size:28px;font-weight:800;margin-top:5px'>{st.session_state.page}</div>", unsafe_allow_html=True)
with h2:
    if st.button("＋ Upload Leads", type="primary", use_container_width=True):
        st.session_state.page = "Batch Analysis"
        st.rerun()


# ============================================================
# DASHBOARD
# ============================================================


def render_dashboard():
    st.markdown(
        """<div class="hero"><div class="eyebrow">WELCOME BACK</div>
        <div class="hero-title">Know your highest-value customers before you call.</div>
        <div class="hero-copy">ICP Insight AI turns customer profile, acquisition, digital behaviour, engagement, and activity signals into an actionable conversion intelligence score.</div></div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    if st.session_state.batch is not None and len(st.session_state.batch):
        b = st.session_state.batch
        total = len(b)
        high = int((b["lead_score"] >= 80).sum())
        avg = b["lead_score"].mean()
        conv = b["purchase_probability"].mean()
    else:
        total, high, avg, conv = 0, 0, 0, 0

    render_metric_cards([
        ("Total Leads", f"{total:,}", "Live dataset" if total else "Upload a dataset"),
        ("High-Value Leads", f"{high:,}", "ICP score ≥ 80"),
        ("Average ICP Score", f"{avg:.1f}", "Across scored leads"),
        ("Conversion Rate", f"{conv:.1f}%", "Mean model probability"),
    ])

    st.write("")
    c1, c2 = st.columns([1.45, 1])
    with c1:
        st.markdown('<div class="card"><div class="card-title">Lead Quality Trend</div><div class="card-sub">Average predicted conversion probability</div>', unsafe_allow_html=True)
        if st.session_state.batch is not None and len(st.session_state.batch):
            trend = st.session_state.batch["purchase_probability"].reset_index(drop=True)
            trend.index = range(1, len(trend) + 1)
            st.line_chart(trend, height=250, use_container_width=True)
        else:
            st.info("Upload leads to populate your live quality trend.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">ICP Distribution</div><div class="card-sub">Lead mix by score category</div>', unsafe_allow_html=True)
        if st.session_state.batch is not None and len(st.session_state.batch):
            dist = st.session_state.batch["lead_category"].value_counts().reindex(["Hot", "Warm", "Medium", "Cold"]).fillna(0)
            st.bar_chart(dist, height=250, use_container_width=True)
        else:
            st.info("Your category distribution will appear here.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1.55, .85])
    with c1:
        st.markdown('<div class="card"><div class="card-title">Recent High-Value Leads</div><div class="card-sub">Top predicted conversion opportunities</div>', unsafe_allow_html=True)
        if st.session_state.batch is not None and len(st.session_state.batch):
            top = st.session_state.batch.head(8).copy()
            display_cols = [c for c in ["age", "city", "course_enquiry", "lead_source", "purchase_probability", "lead_category"] if c in top.columns]
            st.dataframe(top[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No scored leads yet. Upload a dataset to get started.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="insight"><div class="eyebrow">QUICK AI INSIGHT</div><div class="card-title" style="margin-top:8px">Build your ICP from real behaviour.</div><p style="color:#94A3B8;font-size:13px;line-height:1.6">The strongest operational signals in this project come from observed engagement and recency. Use scores to prioritize action, then use the explanation panels to understand why.</p></div>', unsafe_allow_html=True)




# ============================================================
# BATCH ANALYSIS
# ============================================================


def render_batch():
    st.markdown("<div class='eyebrow'>BATCH ANALYSIS</div><h2 style='margin-top:5px'>Upload Customer Dataset</h2><p style='color:#94A3B8'>Analyze thousands of leads and identify your highest-value customer profiles.</p>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.35, .65])
    with c1:
        uploaded = st.file_uploader("Drop your CSV here", type=["csv"], help="Required columns must match the model feature names.")
    with c2:
        sample = pd.DataFrame([{f: 0 for f in FEATURES}])
        sample.loc[0, "age"] = 24
        st.download_button("↓ Download Sample Template", sample.to_csv(index=False).encode(), "icp_insight_template.csv", "text/csv", use_container_width=True)

    if uploaded is None:
        st.markdown('<div class="card"><div class="card-title">Dataset requirements</div><div class="card-sub">Your CSV should contain the same feature columns used by the trained model. The target column is not required.</div></div>', unsafe_allow_html=True)
        return

    try:
        batch = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in batch.columns]
        st.session_state.uploaded_batch_preview = batch
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Leads Detected", f"{len(batch):,}")
        c2.metric("Features Detected", f"{len(batch.columns):,}")
        c3.metric("Compatibility", "Ready" if not missing else "Needs Fix")
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            return
        st.success("Dataset validated successfully.")
        st.markdown('<div class="card"><div class="card-title">Dataset Preview</div><div class="card-sub">First rows of the uploaded dataset</div>', unsafe_allow_html=True)
        st.dataframe(batch.head(10), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("✦ Run ICP Analysis", type="primary", use_container_width=True):
            with st.spinner("Scoring leads…"):
                model = load_model()
                probs = model.predict_proba(batch[FEATURES])[:, 1]
                batch["purchase_probability"] = (probs * 100).round(1)
                batch["lead_score"] = (probs * 100).round(1)
                batch["lead_category"] = batch["lead_score"].apply(category)
                batch = batch.sort_values("purchase_probability", ascending=False).reset_index(drop=True)
                st.session_state.batch = batch
            st.session_state.page = "Batch Results"
            st.rerun()
    except Exception as exc:
        st.error("Something went wrong while validating the CSV.")
        st.code(str(exc))


# ============================================================
# BATCH RESULTS
# ============================================================


def render_batch_results():
    batch = st.session_state.batch
    if batch is None:
        st.info("Upload a dataset first.")
        return
    total = len(batch)
    hot = int((batch.lead_category == "Hot").sum())
    warm = int((batch.lead_category == "Warm").sum())
    avg = batch.lead_score.mean()
    render_metric_cards([
        ("Total Leads", f"{total:,}", "Processed"),
        ("Hot Leads", f"{hot:,}", "Score 80–100"),
        ("Warm Leads", f"{warm:,}", "Score 60–79"),
        ("Average ICP Score", f"{avg:.1f}", "Across dataset"),
    ])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">ICP Category Distribution</div>', unsafe_allow_html=True)
        st.bar_chart(batch.lead_category.value_counts().reindex(["Hot","Warm","Medium","Cold"]).fillna(0), height=250)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Lead Source Performance</div>', unsafe_allow_html=True)
        if "lead_source" in batch:
            st.bar_chart(batch.groupby("lead_source")["purchase_probability"].mean().sort_values(ascending=False), height=250)
        st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="card"><div class="card-title">Scored Leads</div><div class="card-sub">Use the native table search/filter controls to inspect the full dataset.</div>', unsafe_allow_html=True)
    search = st.text_input("Search leads", placeholder="Search city, course, source…")
    filtered = batch
    if search:
        mask = filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.download_button("↓ Download Scored Leads CSV", batch.to_csv(index=False).encode(), "scored_leads.csv", "text/csv", use_container_width=True)


# ============================================================
# ANALYTICS
# ============================================================


def render_analytics():
    st.markdown("<div class='eyebrow'>ANALYTICS</div><h2 style='margin-top:5px'>Understand your customer patterns</h2><p style='color:#94A3B8'>Turn scored leads into practical ICP signals for sales and marketing.</p>", unsafe_allow_html=True)
    batch = st.session_state.batch
    if batch is None or batch.empty:
        st.info("Run a batch analysis to unlock live analytics.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">Lead Source Performance</div>', unsafe_allow_html=True)
        st.bar_chart(batch.groupby("lead_source")["purchase_probability"].mean().sort_values(ascending=False))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Conversion Intent by City</div>', unsafe_allow_html=True)
        st.bar_chart(batch.groupby("city")["purchase_probability"].mean().sort_values(ascending=False))
        st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">Course Interest Distribution</div>', unsafe_allow_html=True)
        st.bar_chart(batch["course_enquiry"].value_counts())
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Engagement vs ICP Score</div>', unsafe_allow_html=True)
        if "website_visits" in batch:
            engagement = batch["website_visits"] + batch["course_page_views"] + batch["email_clicks"] + batch["whatsapp_responses"]
            st.scatter_chart(pd.DataFrame({"Engagement": engagement, "ICP Score": batch["lead_score"]}), x="Engagement", y="ICP Score", height=280)
        st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">Customer Activity Trend</div>', unsafe_allow_html=True)
        if "days_since_last_activity" in batch:
            activity = batch.sort_values("days_since_last_activity")["purchase_probability"].reset_index(drop=True)
            st.line_chart(activity)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        top_features = []
        for col in ["pricing_page_views", "website_visits", "course_page_views", "email_clicks", "demo_attended"]:
            if col in batch and pd.api.types.is_numeric_dtype(batch[col]):
                top_features.append((col, batch[col].corr(batch["lead_score"])))
        top_features = sorted(top_features, key=lambda x: abs(x[1]) if pd.notna(x[1]) else 0, reverse=True)
        insight = ", ".join([x[0].replace("_", " ") for x in top_features[:3]]) or "engagement signals"
        st.markdown(f'<div class="insight"><div class="eyebrow">AI-GENERATED SUMMARY</div><div class="card-title" style="margin-top:8px">Your strongest observed score relationships are around {insight}.</div><p style="color:#94A3B8;font-size:13px;line-height:1.7">This summary is descriptive rather than causal. Validate these patterns against business outcomes before turning them into targeting rules.</p></div>', unsafe_allow_html=True)


# ============================================================
# DATA GUIDE
# ============================================================


def render_data_guide():
    st.markdown("<div class='eyebrow'>DATA GUIDE</div><h2 style='margin-top:5px'>Prepare your customer dataset</h2><p style='color:#94A3B8'>Keep the feature names unchanged so the trained pipeline can score your leads reliably.</p>", unsafe_allow_html=True)
    groups = {
        "Customer Profile": FEATURES[:6],
        "Website Behaviour": FEATURES[6:9],
        "Communication Engagement": FEATURES[9:18],
        "Activity Information": FEATURES[18:],
    }
    descriptions = {
        "Customer Profile": "Who the customer is and what they are interested in.",
        "Website Behaviour": "Signals showing how deeply the customer researches the offering.",
        "Communication Engagement": "Responses to content and direct outreach.",
        "Activity Information": "Recent interaction and video-learning behaviour.",
    }
    for title, cols in groups.items():
        with st.expander(title, expanded=True):
            st.caption(descriptions[title])
            rows = []
            for col in cols:
                rows.append({"Column": col, "Description": col.replace("_", " ").title()})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info("Do not include the target column `purchased_within_30_days` when uploading data for prediction.")


# ============================================================
# ROUTER
# ============================================================

page = st.session_state.page
if page == "Dashboard":
    render_dashboard()
elif page == "Batch Analysis":
    render_batch()
elif page == "Batch Results":
    render_batch_results()
elif page == "Analytics":
    render_analytics()
elif page == "Data Guide":
    render_data_guide()


# ============================================================
# FOOTER
# ============================================================

st.markdown("<div style='text-align:center;color:#475569;font-size:11px;margin-top:40px'>ICP Insight AI · Customer Intelligence Platform · ML-powered purchase likelihood</div>", unsafe_allow_html=True)