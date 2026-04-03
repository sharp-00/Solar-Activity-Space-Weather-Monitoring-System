"""utils/theme.py — Shared CSS theme and Plotly layout defaults."""
import streamlit as st

CSS = """
<style>
    html, body, [class*="css"] { font-family: 'Inter', 'Roboto', sans-serif !important; }
    .main .block-container { padding-top: 1rem; max-width: 98%; }
    h1 { color: #f59e0b; font-weight: 700; }
    h2 { font-weight: 600; font-size: 1.5rem; opacity: 0.9; }
    h3 { font-weight: 500; font-size: 1.25rem; opacity: 0.8; }
    .stInfo { background: rgba(245,158,11,0.1) !important; border-left: 4px solid #f59e0b !important; }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    .plotly { will-change: transform; }
</style>
"""

_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1),
)

def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)

def layout(height=400, **kwargs) -> dict:
    d = {**_BASE, "height": height}
    d.update(kwargs)
    return d

def axis(title: str) -> dict:
    return {"title": title}
