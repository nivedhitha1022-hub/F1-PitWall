"""
app.py  —  PitWall Analytics Dashboard
───────────────────────────────────────
Run with:
    streamlit run app.py

Project structure
    app.py
    data_generator.py     ← data loading & cleaning
    model_utils.py        ← feature engineering, RF model, KMeans
    theme.py              ← F1 colour palette, CSS, layout helpers
    tab1_descriptive.py
    tab2_diagnostic.py
    tab3_predictive.py
    tab4_prescriptive.py
    data/
        PitWall_Analytics_Cleaned.xlsx
    .streamlit/
        config.toml
"""
from __future__ import annotations
import sys
from pathlib import Path

# ── Ensure project root is on sys.path (needed for Streamlit Cloud) ────────────
ROOT = Path(__file__).parent
for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

# ── Page config — MUST be the first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title   = "PitWall Analytics",
    page_icon    = "🏎",
    layout       = "wide",
    initial_sidebar_state = "collapsed",
)

from theme         import F1_CSS
from data_generator import load_data
import tab1_descriptive
import tab2_diagnostic
import tab3_predictive
import tab4_prescriptive

# ── Inject global CSS ──────────────────────────────────────────────────────────
st.markdown(F1_CSS, unsafe_allow_html=True)

# ── Load data (cached) ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load():
    return load_data()

with st.spinner("🏎  Loading race data…"):
    subs, sess, mrr = _load()

# ── Dashboard header ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0A0A0A 0%, #1A0005 50%, #0A0A0A 100%);
        border-bottom: 3px solid #E8002D;
        padding: 20px 28px 16px 28px;
        margin: -1rem -1rem 1.5rem -1rem;
    ">
        <div style="
            font-size: 9px; letter-spacing: 4px; color: #E8002D; font-weight: 700;
            text-transform: uppercase; margin-bottom: 6px;
            font-family: 'Titillium Web', Arial, sans-serif;
        ">
            F1 PERFORMANCE DATA PLATFORM  &nbsp;·&nbsp;  SUBSCRIBER RETENTION INTELLIGENCE
        </div>
        <div style="
            font-size: 28px; font-weight: 900; color: #F5F5F5; letter-spacing: 1.5px;
            font-family: 'Titillium Web', Arial Black, sans-serif;
        ">
            🏎&nbsp; PITWALL ANALYTICS
        </div>
        <div style="
            font-size: 12px; color: #666; margin-top: 5px;
            font-family: 'Titillium Web', Arial, sans-serif; letter-spacing: 0.5px;
        ">
            800 Subscribers &nbsp;·&nbsp; 29,240 Sessions &nbsp;·&nbsp;
            3 Plan Tiers &nbsp;·&nbsp; Seasons 2023 – 2024
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tab navigation ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋  Descriptive",
    "🔍  Diagnostic",
    "🔮  Predictive",
    "🎯  Prescriptive",
])

with tab1:
    tab1_descriptive.render(subs, sess, mrr)

with tab2:
    tab2_diagnostic.render(subs, sess, mrr)

with tab3:
    tab3_predictive.render(subs, sess, mrr)

with tab4:
    tab4_prescriptive.render(subs, sess, mrr)
