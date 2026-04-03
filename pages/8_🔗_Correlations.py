"""pages/8_🔗_Correlations.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.express as px
import streamlit as st

from utils.data_loader import load_main_data, load_kp_data, load_dst_data
from utils.refresh import auto_refresh_check, render_refresh_sidebar, render_date_filter
from utils.theme import apply_theme, layout

st.set_page_config(page_title="Correlations", page_icon="🔗", layout="wide")
apply_theme(); auto_refresh_check(); render_refresh_sidebar()

st.title("🔗 Correlation & Relationship Analytics")

with st.expander("ℹ️ What We Are Analyzing: Correlations", expanded=False):
    st.markdown("""
    - **What we are doing:** Computing the Pearson Correlation Coefficients ($r$) between the various measured indices to construct a correlation matrix.
    - **Goal:** Looking for collinearity. We want to statistically prove that as Sunspot Number goes up, the Solar Radio Flux ($F10.7$) reliably goes up with it, and that high flares correlate to subsequent geomagnetic disturbances.
    """)


st.info("💡 **Historical Trivia:** In 1946, Arthur Covington verified the strict correlation between the F10.7 cm radio flux and Sunspot Number using a repurposed military radar, proving that you could 'listen' to solar activity even on cloudy days.")

try:
    df_main = render_date_filter(load_main_data())
    kp_df   = load_kp_data()
    dst_df  = load_dst_data()
    if df_main.empty:
        st.warning("No data."); st.stop()

    df = df_main.copy()
    for src in [kp_df, dst_df]:
        if not src.empty:
            for col in src.columns:
                if col not in df.columns:
                    df[col] = src[col].reindex(df.index)

    st.info("Near **+1**: strong positive link.  Near **−1**: strong inverse link.")

    c_left, c_right = st.columns([1, 2])
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if df[c].notna().sum() > 100]

    with c_left:
        method = st.radio("Method", ["Pearson (Linear)","Spearman (Rank)"])
        default = [c for c in ["ssn","f107","flare_xray_total","kp_daily_max","dst_daily_min"]
                   if c in num_cols]   # must be in num_cols, not just df.columns
        feats = st.multiselect("Features", num_cols, default=default)
        if len(feats) > 1:
            corr = df[feats].corr(method="pearson" if "Pearson" in method else "spearman")
            fig  = px.imshow(corr, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(**layout(380))
            st.plotly_chart(fig, use_container_width=True)

    with c_right:
        if len(num_cols) < 2:
            st.info("Not enough numeric columns.")
        else:
            def_x = "f107" if "f107" in num_cols else num_cols[0]
            def_y = "kp_daily_max" if "kp_daily_max" in num_cols else num_cols[min(1,len(num_cols)-1)]
            x = st.selectbox("X axis", num_cols, index=num_cols.index(def_x))
            y = st.selectbox("Y axis", num_cols, index=num_cols.index(def_y))
            if x != y and "year" in df.columns:
                fig2 = px.scatter(df.reset_index(), x=x, y=y, color="year",
                                  hover_data=["date"], trendline="ols",
                                  trendline_color_override="#ef4444", opacity=0.5)
                fig2.update_layout(**layout(520))
                st.plotly_chart(fig2, use_container_width=True)
                try:
                    r2 = px.get_trendline_results(fig2).iloc[0]["px_fit_results"].rsquared
                    st.info(f"**OLS R²:** {r2:.4f}")
                except Exception:
                    pass
except Exception as e:
    st.error(f"Page error: {e}")
