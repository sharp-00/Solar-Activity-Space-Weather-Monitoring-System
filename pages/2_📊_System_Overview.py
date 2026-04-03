"""pages/2_📊_System_Overview.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_main_data, load_kp_data, load_dst_data, load_f107_data, load_sunspots_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout, axis

st.set_page_config(page_title="System Overview", page_icon="📊", layout="wide")
apply_theme()
auto_refresh_check()
render_refresh_sidebar()

st.title("📊 System Overview: Time-Series Diagnostics")

df_all  = load_main_data()
kp_df   = load_kp_data()
dst_df  = load_dst_data()
f107_df = load_f107_data()
ssn_df  = load_sunspots_data()

# Enrich main df with individual datasets (fills columns missing from old CSVs)
_all = df_all.copy()
for src in [kp_df, dst_df, f107_df]:
    if not src.empty:
        for col in src.columns:
            if col not in _all.columns or _all[col].isna().all():
                _all[col] = src[col].reindex(_all.index)
# SSN: sunspot df uses 'sn' but main df uses 'ssn' — map both
# Also override if column exists but is entirely NaN
if not ssn_df.empty:
    sn_col = "sn" if "sn" in ssn_df.columns else ("ssn" if "ssn" in ssn_df.columns else None)
    if sn_col:
        needs_ssn = "ssn" not in _all.columns or _all["ssn"].isna().all()
        if needs_ssn:
            _all["ssn"] = ssn_df[sn_col].reindex(_all.index)
    if "f107" in ssn_df.columns:
        if "f107" not in _all.columns or _all["f107"].isna().all():
            _all["f107"] = ssn_df["f107"].reindex(_all.index)

df = render_date_filter(_all)

if df.empty:
    st.warning("No data. Use the sidebar to fetch data.")
    st.stop()

# Merge Kp and Dst into the filtered view if they have more complete data
def _get(col, preferred_df, fallback_df):
    """Return series for col from whichever df has it non-empty."""
    for d in [preferred_df, fallback_df]:
        if d is not None and not d.empty and col in d.columns and d[col].notna().any():
            return d[col].reindex(fallback_df.index)
    return None

st.info(
    "**Cause (top row)** vs **Effect (bottom row).**  \n"
    "27-day rolling means expose the 11-year cycle. "
    "When SSN peaks, Dst plummets past −100 nT (severe storms)."
)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Sunspot Number (SSN)")
    ssn = _get("ssn", df, df)
    if ssn is not None and ssn.notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=ssn.index, y=ssn,
                                   line=dict(color="#fbbf24", width=1), opacity=0.5, name="Daily"))
        rm = ssn.rolling(27, center=True, min_periods=5).mean()
        fig.add_trace(go.Scattergl(x=rm.index, y=rm,
                                   line=dict(color="#f59e0b", width=2), name="27d Mean"))
        fig.update_layout(**layout(350, xaxis=axis("Date"), yaxis=axis("SSN")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SSN data not found in dataset. Click **⚡ Fetch Latest Data** to rebuild.")

with c2:
    st.subheader("F10.7 Solar Flux")
    f107 = _get("f107", df, df)
    if f107 is not None and f107.notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=f107.index, y=f107,
                                   line=dict(color="#fb923c", width=1), opacity=0.5, name="Daily"))
        rm = f107.rolling(27, center=True, min_periods=5).mean()
        fig.add_trace(go.Scattergl(x=rm.index, y=rm,
                                   line=dict(color="#ea580c", width=2), name="27d Mean"))
        fig.update_layout(**layout(350, xaxis=axis("Date"), yaxis=axis("F10.7 (sfu)")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("F10.7 data not available.")

st.markdown("---")
c3, c4 = st.columns(2)

with c3:
    st.subheader("M & X-Class Flares")
    has_flares = any(c in df.columns for c in ["flare_M","flare_X"])
    if has_flares:
        fig = go.Figure()
        if "flare_M" in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df["flare_M"], name="M-Class", marker_color="#fbbf24"))
        if "flare_X" in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df["flare_X"], name="X-Class", marker_color="#ef4444"))
        fig.update_layout(**layout(350, barmode="stack", xaxis=axis("Date"), yaxis=axis("Count")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Flare data not available.")

with c4:
    st.subheader("Kp Index (Max Daily)")
    # Use dedicated kp_df first, filtered to same date range as df
    kp_source = kp_df if not kp_df.empty and "kp_daily_max" in kp_df.columns else df
    kp_filt   = kp_source.loc[df.index.min():df.index.max()] if not kp_source.empty else kp_source
    kp_col    = "kp_daily_max"

    if not kp_filt.empty and kp_col in kp_filt.columns and kp_filt[kp_col].notna().any():
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=kp_filt.index, y=kp_filt[kp_col],
                                   line=dict(color="#22c55e", width=1),
                                   fill="tozeroy", fillcolor="rgba(34,197,94,0.2)", name="Kp Max"))
        fig.add_hline(y=5, line_dash="dash", line_color="#fbbf24", annotation_text="G1")
        fig.add_hline(y=7, line_dash="dash", line_color="#ef4444", annotation_text="G3")
        fig.update_layout(**layout(350, yaxis_range=[0, 9.5],
                                   xaxis=axis("Date"), yaxis=axis("Kp (0–9)")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Kp data not found. Click **⚡ Fetch Latest Data** to rebuild.")

st.markdown("---")
st.subheader("Dst Index (Equatorial Ring Current)")
# Use dedicated dst_df first, filtered to date range
dst_source = dst_df if not dst_df.empty and "dst_daily_mean" in dst_df.columns else df
dst_filt   = dst_source.loc[df.index.min():df.index.max()] if not dst_source.empty else dst_source

if (not dst_filt.empty and
        all(c in dst_filt.columns for c in ["dst_daily_mean","dst_daily_min","dst_daily_max"])):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=dst_filt.index.tolist() + dst_filt.index[::-1].tolist(),
        y=dst_filt["dst_daily_max"].tolist() + dst_filt["dst_daily_min"][::-1].tolist(),
        fill="toself", fillcolor="rgba(34,197,94,0.15)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="Range",
    ))
    fig.add_trace(go.Scattergl(x=dst_filt.index, y=dst_filt["dst_daily_mean"],
                                line=dict(color="#22c55e", width=1.5), name="Mean"))
    fig.add_hline(y=-50,  line_dash="dash", line_color="#fbbf24", annotation_text="Moderate")
    fig.add_hline(y=-100, line_dash="dash", line_color="#f97316", annotation_text="Intense")
    fig.add_hline(y=-200, line_dash="dash", line_color="#dc2626", annotation_text="Severe")
    fig.update_layout(**layout(400, xaxis=axis("Date"), yaxis=axis("Dst (nT)")))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Dst data not available.")
