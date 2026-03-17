"""
app.py
Solar Activity & Space Weather Monitoring System
Plotly Dash dashboard — run with: python3 app.py
"""

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px

from ingest import fetch_sunspots, fetch_kp_index
from pipeline import (
    clean_sunspots,
    clean_kp,
    resample_kp_daily,
    align_datasets,
    add_smoothed_columns,
    add_anomaly_scores,
)
from analysis import (
    cross_correlation,
    find_extreme_events,
    compute_monthly_stats,
)

# ---------------------------------------------------------------------------
# Design tokens — clean light theme
# ---------------------------------------------------------------------------

BG      = "#f7f8fa"
SURFACE = "#ffffff"
BORDER  = "#e8eaed"
TEXT    = "#1a1d23"
MUTED   = "#8a909e"

C_SN    = "#e8621a"   # warm orange  — sunspots
C_KP    = "#0f7adb"   # blue         — geomagnetic
C_R27   = "#f0a030"   # amber        — 27-day rolling
C_R365  = "#c74f10"   # deep orange  — 365-day rolling
C_BASE  = "#7c5cbf"   # violet       — SG baseline
C_ANOM  = "#d93854"   # red          — anomaly
C_STORM = "#1a6bb5"   # navy         — storm

CHART_BG   = SURFACE
CHART_GRID = "#f0f1f3"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=TEXT, family="'DM Sans', 'Helvetica Neue', sans-serif", size=12),
    xaxis=dict(gridcolor=CHART_GRID, zeroline=False, linecolor=BORDER,
               tickfont=dict(color=MUTED, size=11)),
    yaxis=dict(gridcolor=CHART_GRID, zeroline=False, linecolor=BORDER,
               tickfont=dict(color=MUTED, size=11)),
    margin=dict(l=52, r=40, t=40, b=44),
    hovermode="x unified",
)

# ---------------------------------------------------------------------------
# Data loading at startup
# ---------------------------------------------------------------------------

df_all     = None
df_monthly = None
load_error = None

try:
    sn_daily   = clean_sunspots(fetch_sunspots("daily"))
    kp_raw     = clean_kp(fetch_kp_index())
    kp_daily   = resample_kp_daily(kp_raw)
    df_all     = align_datasets(sn_daily, kp_daily)
    df_all     = add_smoothed_columns(df_all)
    df_all     = add_anomaly_scores(df_all)
    df_monthly = compute_monthly_stats(df_all)
except FileNotFoundError as exc:
    load_error = str(exc)
except Exception as exc:
    load_error = f"Unexpected error during data loading: {exc}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_date(df, start, end):
    if df is None or df.empty:
        return pd.DataFrame()
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df.index >= pd.Timestamp(start)
    if end:
        mask &= df.index <= pd.Timestamp(end)
    return df.loc[mask]


def _fig(**extra):
    layout = {**PLOTLY_LAYOUT, **extra}
    return go.Figure(layout=go.Layout(**layout))


def _card(label, value, accent=TEXT):
    return html.Div([
        html.Div(label, style={
            "fontSize": "10px", "fontWeight": "600", "letterSpacing": "0.07em",
            "textTransform": "uppercase", "color": MUTED, "marginBottom": "6px",
        }),
        html.Div(value, style={
            "fontSize": "26px", "fontWeight": "700", "color": accent,
            "letterSpacing": "-0.02em", "lineHeight": "1",
        }),
    ], style={
        "background": SURFACE, "border": f"1px solid {BORDER}",
        "borderRadius": "10px", "padding": "18px 22px",
        "flex": "1", "minWidth": "110px",
    })


def _compute_kpis(df):
    if df is None or df.empty:
        return [("Latest SN","—",TEXT),("Latest Kp","—",TEXT),
                ("Mean SN","—",TEXT),("Peak Kp","—",TEXT),("Storm Days","—",TEXT)]
    def _safe(series):
        s = series.dropna()
        return s.iloc[-1] if not s.empty else np.nan
    latest_sn  = _safe(df["sn"])       if "sn"             in df.columns else np.nan
    latest_kp  = _safe(df["Kp_max"])   if "Kp_max"         in df.columns else np.nan
    mean_sn    = df["sn"].mean()        if "sn"             in df.columns else np.nan
    peak_kp    = df["Kp_max"].max()     if "Kp_max"         in df.columns else np.nan
    storm_days = int((df["Kp_storm_hours"] > 0).sum()) if "Kp_storm_hours" in df.columns else 0
    f = lambda v, fmt: fmt % v if not np.isnan(v) else "—"
    return [
        ("Latest SN",     f(latest_sn, "%.0f"), C_SN),
        ("Latest Kp Max", f(latest_kp, "%.1f"), C_KP),
        ("Mean SN",       f(mean_sn,   "%.1f"), TEXT),
        ("Peak Kp",       f(peak_kp,   "%.1f"), TEXT),
        ("Storm Days",    str(storm_days),       TEXT),
    ]


def _section(title, children, note=None):
    inner = [
        html.Div(title, style={
            "fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.08em",
            "textTransform": "uppercase", "color": MUTED, "marginBottom": "14px",
        }),
    ] + (children if isinstance(children, list) else [children])
    if note:
        inner.append(html.Div(note, style={
            "fontSize": "12px", "color": MUTED, "marginTop": "10px",
            "borderLeft": f"2px solid {BORDER}", "paddingLeft": "10px",
            "lineHeight": "1.6",
        }))
    return html.Div(inner, style={
        "background": SURFACE, "border": f"1px solid {BORDER}",
        "borderRadius": "10px", "padding": "22px 24px", "marginBottom": "16px",
    })


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

GOOGLE_FONT = "https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap"

app = dash.Dash(
    __name__,
    title="Solar Monitor",
    suppress_callback_exceptions=True,
    external_stylesheets=[GOOGLE_FONT],
)

_TAB_BASE = dict(
    fontFamily="'DM Sans', sans-serif",
    fontSize="13px", fontWeight="500", color=MUTED,
    backgroundColor="transparent",
    border="none", borderBottom="2px solid transparent",
    padding="10px 20px", cursor="pointer",
)
_TAB_SEL = {**_TAB_BASE, "color": TEXT, "fontWeight": "700",
            "border": "none", "borderBottom": f"2px solid {C_SN}"}

# ---------------------------------------------------------------------------
# Error layout
# ---------------------------------------------------------------------------

if load_error:
    app.layout = html.Div([
        html.Div("Data missing", style={"color": C_ANOM, "fontWeight": "700",
                                        "fontSize": "18px", "marginBottom": "12px"}),
        html.Pre(load_error, style={
            "background": "#fff5f6", "border": f"1px solid #fcc",
            "borderRadius": "8px", "padding": "16px", "fontSize": "13px", "color": C_ANOM,
        }),
        html.Div("Run these commands first, then restart the app:",
                 style={"color": MUTED, "margin": "16px 0 8px", "fontSize": "13px"}),
        html.Code("python3 download_data.py", style={
            "background": "#f0f1f3", "borderRadius": "6px",
            "padding": "8px 14px", "fontSize": "14px", "fontWeight": "600", "color": TEXT,
        }),
    ], style={"fontFamily": "'DM Sans', sans-serif", "background": BG,
              "minHeight": "100vh", "padding": "64px 48px", "color": TEXT})

else:
    _min_date = df_all.index.min().date() if df_all is not None else None
    _max_date = df_all.index.max().date() if df_all is not None else None

    app.layout = html.Div(
        style={"fontFamily": "'DM Sans', sans-serif", "background": BG,
               "minHeight": "100vh", "color": TEXT},
        children=[

            # Top bar
            html.Div([
                html.Div([
                    html.Div("Solar Monitor", style={
                        "fontSize": "15px", "fontWeight": "700",
                        "color": TEXT, "letterSpacing": "-0.01em",
                    }),
                    html.Div("SILSO · NOAA SWPC", style={
                        "fontSize": "11px", "color": MUTED, "marginTop": "2px",
                    }),
                ]),
                html.Div([
                    html.Div("Date range", style={"fontSize": "11px", "color": MUTED,
                                                   "marginBottom": "4px"}),
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=_min_date, max_date_allowed=_max_date,
                        start_date=_min_date, end_date=_max_date,
                        display_format="YYYY-MM-DD",
                    ),
                ]),
            ], style={
                "display": "flex", "alignItems": "flex-end",
                "justifyContent": "space-between",
                "padding": "18px 32px", "background": SURFACE,
                "borderBottom": f"1px solid {BORDER}",
                "flexWrap": "wrap", "gap": "16px",
            }),

            # KPI row
            html.Div(id="kpi-row", style={
                "display": "flex", "gap": "10px",
                "padding": "20px 32px", "flexWrap": "wrap",
            }),

            # Tabs
            html.Div([
                dcc.Tabs(
                    id="main-tabs", value="tab-ts",
                    children=[
                        dcc.Tab(label="Time Series",    value="tab-ts",
                                style=_TAB_BASE, selected_style=_TAB_SEL),
                        dcc.Tab(label="Correlation",    value="tab-corr",
                                style=_TAB_BASE, selected_style=_TAB_SEL),
                        dcc.Tab(label="Extreme Events", value="tab-extreme",
                                style=_TAB_BASE, selected_style=_TAB_SEL),
                        dcc.Tab(label="Smoothing",      value="tab-smooth",
                                style=_TAB_BASE, selected_style=_TAB_SEL),
                    ],
                    colors={"border": BORDER, "primary": C_SN, "background": "transparent"},
                    style={"borderBottom": f"1px solid {BORDER}"},
                ),
            ], style={"padding": "0 32px", "background": SURFACE}),

            # Controls strip
            html.Div([
                html.Div([
                    html.Span("σ threshold", style={"fontSize": "11px", "color": MUTED,
                                                     "marginRight": "10px", "whiteSpace": "nowrap"}),
                    dcc.Slider(
                        id="sigma-slider", min=1.5, max=4.0, step=0.1, value=2.5,
                        marks={v: {"label": str(v), "style": {"color": MUTED, "fontSize": "10px"}}
                               for v in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]},
                        tooltip={"placement": "bottom"},
                    ),
                ], style={"display": "flex", "alignItems": "center",
                          "flex": "1", "minWidth": "260px", "maxWidth": "380px"}),
                html.Div([
                    html.Span("Max lag (days)", style={"fontSize": "11px", "color": MUTED,
                                                        "marginRight": "10px", "whiteSpace": "nowrap"}),
                    dcc.Slider(
                        id="lag-slider", min=5, max=60, step=1, value=30,
                        marks={v: {"label": str(v), "style": {"color": MUTED, "fontSize": "10px"}}
                               for v in [5, 15, 30, 45, 60]},
                        tooltip={"placement": "bottom"},
                    ),
                ], style={"display": "flex", "alignItems": "center",
                          "flex": "1", "minWidth": "260px", "maxWidth": "380px"}),
            ], style={
                "display": "flex", "gap": "32px", "padding": "14px 32px",
                "background": BG, "borderBottom": f"1px solid {BORDER}",
                "flexWrap": "wrap", "alignItems": "center",
            }),

            html.Div(id="tab-content", style={"padding": "24px 32px"}),
        ],
    )

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(Output("kpi-row", "children"),
          Input("date-range", "start_date"),
          Input("date-range", "end_date"))
def update_kpis(start, end):
    if df_all is None:
        return []
    df = _filter_date(df_all, start, end)
    return [_card(lbl, val, acc) for lbl, val, acc in _compute_kpis(df)]


@callback(Output("tab-content", "children"),
          Input("main-tabs", "value"),
          Input("date-range", "start_date"),
          Input("date-range", "end_date"),
          Input("sigma-slider", "value"),
          Input("lag-slider", "value"))
def render_tab(tab, start, end, sigma, max_lag):
    if df_all is None:
        return html.Div("No data loaded.", style={"color": C_ANOM})
    df = _filter_date(df_all, start, end)
    if   tab == "tab-ts":      return _tab_timeseries(df)
    elif tab == "tab-corr":    return _tab_correlation(df, max_lag)
    elif tab == "tab-extreme": return _tab_extreme(df, sigma)
    elif tab == "tab-smooth":  return _tab_smoothing(df)
    return html.Div()


# ---------------------------------------------------------------------------
# Tab 1 — Time Series
# ---------------------------------------------------------------------------

def _tab_timeseries(df):
    fig = _fig(
        height=380,
        yaxis=dict(title="Sunspot Number", gridcolor=CHART_GRID, zeroline=False,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis2=dict(title="Kp (max)", overlaying="y", side="right", range=[0, 9],
                    gridcolor=CHART_GRID, zeroline=False, showgrid=False,
                    tickfont=dict(color=MUTED, size=11)),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
    )
    if "sn" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sn"], name="Sunspot Number",
                                 line=dict(color=C_SN, width=1), opacity=0.4))
    if "sn_roll27d" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sn_roll27d"], name="27-day mean",
                                 line=dict(color=C_R27, width=2)))
    if "Kp_max" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Kp_max"], name="Kp max",
                                 line=dict(color=C_KP, width=1), opacity=0.55, yaxis="y2"))
        storms = df[df["Kp_max"] >= 5]
        if not storms.empty:
            fig.add_trace(go.Scatter(
                x=storms.index, y=storms["Kp_max"], name="Storm (Kp≥5)",
                mode="markers", marker=dict(color=C_STORM, size=5, symbol="diamond"),
                yaxis="y2",
            ))
    if not df.empty:
        fig.add_hline(y=5, line_dash="dot", line_color=C_STORM, line_width=1,
                      opacity=0.5, yref="y2",
                      annotation_text="Kp=5",
                      annotation_font=dict(color=MUTED, size=10))

    heatmap_content = html.Div("Insufficient data for heatmap.",
                                style={"color": MUTED, "fontSize": "13px"})
    if "sn" in df.columns and not df.empty:
        monthly = df["sn"].resample("ME").mean().dropna()
        if not monthly.empty:
            hdf = pd.DataFrame({"year": monthly.index.year,
                                 "month": monthly.index.month, "sn": monthly.values})
            pivot = hdf.pivot_table(index="year", columns="month", values="sn", aggfunc="mean")
            mnms  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            fig2  = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[mnms[m - 1] for m in pivot.columns],
                y=pivot.index.astype(str),
                colorscale=[[0, "#f7f8fa"], [0.5, "#f4a464"], [1, "#c84a0c"]],
                colorbar=dict(title="SN", tickfont=dict(color=MUTED, size=10), len=0.8),
            ))
            fig2.update_layout(
                **PLOTLY_LAYOUT, height=320,
                xaxis=dict(gridcolor=CHART_GRID, zeroline=False,
                           tickfont=dict(color=MUTED, size=11)),
                yaxis=dict(gridcolor=CHART_GRID, zeroline=False,
                           tickfont=dict(color=MUTED, size=11)),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            heatmap_content = dcc.Graph(figure=fig2, config={"displayModeBar": False})

    return html.Div([
        _section("Solar activity over time",
                 dcc.Graph(figure=fig, config={"displayModeBar": False})),
        _section("Monthly mean sunspot number", heatmap_content),
    ])


# ---------------------------------------------------------------------------
# Tab 2 — Correlation & Lag
# ---------------------------------------------------------------------------

def _tab_correlation(df, max_lag):
    return html.Div([
        _section("Settings", html.Div([
            html.Span("Compare SN against  ",
                      style={"color": MUTED, "fontSize": "13px"}),
            dcc.Dropdown(
                id="kp-metric-drop",
                options=[
                    {"label": "Kp Mean",        "value": "Kp_mean"},
                    {"label": "Kp Max",         "value": "Kp_max"},
                    {"label": "Kp Storm Hours", "value": "Kp_storm_hours"},
                ],
                value="Kp_max", clearable=False,
                style={"width": "200px", "display": "inline-block",
                       "fontSize": "13px", "verticalAlign": "middle"},
            ),
        ])),
        html.Div(id="corr-content"),
        _section("Lagged scatter", [
            html.Div([
                html.Span("Shift SN forward by ",
                          style={"color": MUTED, "fontSize": "13px"}),
                dcc.Slider(
                    id="lag-apply-slider", min=0, max=max_lag, step=1, value=0,
                    marks={0: "0d", max_lag // 2: f"{max_lag // 2}d", max_lag: f"{max_lag}d"},
                    tooltip={"placement": "bottom"},
                ),
            ], style={"maxWidth": "420px", "marginBottom": "20px"}),
            html.Div(id="lagged-scatter"),
        ],
        note="Positive lag = SN leads Kp — consistent with CME travel time of 1–3 days."),
    ])


@callback(Output("corr-content", "children"),
          Input("kp-metric-drop", "value"),
          Input("date-range", "start_date"),
          Input("date-range", "end_date"),
          Input("lag-slider", "value"))
def update_corr(kp_col, start, end, max_lag):
    if df_all is None:
        return html.Div()
    df = _filter_date(df_all, start, end)
    if "sn" not in df.columns or kp_col not in df.columns:
        return html.Div(f"Column '{kp_col}' not available in this date range.",
                        style={"color": MUTED, "fontSize": "13px"})
    cc = cross_correlation(df["sn"], df[kp_col], max_lag=max_lag).dropna(subset=["pearson_r"])
    if cc.empty:
        return html.Div("Insufficient overlapping data.", style={"color": MUTED})

    peak_row = cc.loc[cc["pearson_r"].abs().idxmax()]
    peak_lag = int(peak_row["lag_days"])
    peak_r   = peak_row["pearson_r"]
    colors   = [C_SN if r >= 0 else C_ANOM for r in cc["pearson_r"]]

    fig = _fig(
        height=320,
        xaxis=dict(title="Lag (days)", gridcolor=CHART_GRID, zeroline=False,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(title="Pearson r", gridcolor=CHART_GRID, zeroline=True,
                   zerolinecolor=BORDER, range=[-1, 1],
                   tickfont=dict(color=MUTED, size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.add_trace(go.Bar(x=cc["lag_days"], y=cc["pearson_r"],
                         marker_color=colors, showlegend=False))
    fig.add_vline(x=peak_lag, line_dash="dot", line_color=MUTED, line_width=1,
                  annotation_text=f"lag={peak_lag}d · r={peak_r:.3f}",
                  annotation_font=dict(color=MUTED, size=10))

    return html.Div([
        html.Div([_card("Peak r", f"{peak_r:.3f}"),
                  _card("Peak lag", f"{peak_lag} d")],
                 style={"display": "flex", "gap": "10px", "marginBottom": "14px"}),
        _section(f"Cross-correlation: SN vs {kp_col}",
                 dcc.Graph(figure=fig, config={"displayModeBar": False})),
    ])


@callback(Output("lagged-scatter", "children"),
          Input("lag-apply-slider", "value"),
          Input("kp-metric-drop", "value"),
          Input("date-range", "start_date"),
          Input("date-range", "end_date"))
def update_lagged_scatter(lag, kp_col, start, end):
    if df_all is None:
        return html.Div()
    df = _filter_date(df_all, start, end)
    if "sn" not in df.columns or kp_col not in df.columns:
        return html.Div()
    scatter_df = pd.DataFrame(
        {"sn_lagged": df["sn"].shift(lag), kp_col: df[kp_col]}
    ).dropna()
    if scatter_df.empty:
        return html.Div("No data.", style={"color": MUTED})
    fig = px.scatter(scatter_df, x="sn_lagged", y=kp_col,
                     trendline="lowess",
                     labels={"sn_lagged": f"SN (lag +{lag}d)", kp_col: kp_col},
                     color_discrete_sequence=[C_SN],
                     trendline_color_override=C_BASE)
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_traces(marker=dict(size=4, opacity=0.3), selector=dict(mode="markers"))
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Tab 3 — Extreme Events
# ---------------------------------------------------------------------------

def _tab_extreme(df, sigma):
    sn_ev = find_extreme_events(df, "sn",     sigma) if "sn"     in df.columns else pd.DataFrame()
    kp_ev = find_extreme_events(df, "Kp_max", sigma) if "Kp_max" in df.columns else pd.DataFrame()

    TH = {"padding": "8px 12px", "fontSize": "10px", "fontWeight": "700",
          "letterSpacing": "0.06em", "textTransform": "uppercase",
          "color": MUTED, "borderBottom": f"1px solid {BORDER}", "textAlign": "left"}
    TD = {"padding": "8px 12px", "fontSize": "13px",
          "borderBottom": f"1px solid {BORDER}", "color": TEXT}

    def _tbl(ev, col, color):
        if ev.empty:
            return html.Div(f"No events at σ > {sigma}.",
                            style={"color": MUTED, "fontSize": "13px"})
        rows = [html.Tr([html.Th("Date", style=TH),
                         html.Th(col, style=TH),
                         html.Th("z", style=TH)])]
        for r in ev.head(12).itertuples():
            date_str = str(r.Index.date()) if hasattr(r.Index, "date") else str(r.Index)
            val      = getattr(r, col, None)
            val_str  = f"{val:.1f}" if val is not None and not pd.isna(val) else "—"
            rows.append(html.Tr([
                html.Td(date_str, style=TD),
                html.Td(val_str,  style=TD),
                html.Td(f"{r.z_score:.2f}",
                        style={**TD, "color": color, "fontWeight": "600"}),
            ]))
        return html.Table(rows, style={"borderCollapse": "collapse", "width": "100%"})

    fig_tl = _fig(
        height=360,
        yaxis=dict(title="Sunspot Number", gridcolor=CHART_GRID, zeroline=False,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis2=dict(title="Kp max", overlaying="y", side="right", range=[0, 9],
                    gridcolor=CHART_GRID, zeroline=False, showgrid=False,
                    tickfont=dict(color=MUTED, size=11)),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
    )
    if "sn" in df.columns:
        fig_tl.add_trace(go.Scatter(x=df.index, y=df["sn"], name="SN",
                                    line=dict(color=C_SN, width=1), opacity=0.3))
    if "Kp_max" in df.columns:
        fig_tl.add_trace(go.Scatter(x=df.index, y=df["Kp_max"], name="Kp max",
                                    line=dict(color=C_KP, width=1), opacity=0.3,
                                    yaxis="y2"))
    if not sn_ev.empty:
        fig_tl.add_trace(go.Scatter(x=sn_ev.index, y=sn_ev["sn"],
                                    name="Extreme SN", mode="markers",
                                    marker=dict(color=C_ANOM, size=7)))
    if not kp_ev.empty:
        fig_tl.add_trace(go.Scatter(x=kp_ev.index, y=kp_ev["Kp_max"],
                                    name="Extreme Kp", mode="markers",
                                    marker=dict(color=C_STORM, size=7, symbol="diamond"),
                                    yaxis="y2"))

    fig_z = _fig(
        height=260,
        yaxis=dict(title="Z-score", gridcolor=CHART_GRID, zeroline=True,
                   zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=11)),
        legend=dict(orientation="h", y=-0.22, font=dict(size=11, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
    )
    if "sn_zscore" in df.columns:
        fig_z.add_trace(go.Scatter(x=df.index, y=df["sn_zscore"], name="SN z-score",
                                   line=dict(color=C_SN, width=1)))
    if "Kp_max_zscore" in df.columns:
        fig_z.add_trace(go.Scatter(x=df.index, y=df["Kp_max_zscore"], name="Kp z-score",
                                   line=dict(color=C_KP, width=1)))
    fig_z.add_hline(y=sigma,  line_dash="dot", line_color=C_ANOM, line_width=1,
                    annotation_text=f"+{sigma}σ",
                    annotation_font=dict(color=MUTED, size=10))
    fig_z.add_hline(y=-sigma, line_dash="dot", line_color=C_ANOM, line_width=1,
                    annotation_text=f"−{sigma}σ",
                    annotation_font=dict(color=MUTED, size=10))

    return html.Div([
        html.Div([
            html.Div([
                html.Div(f"Extreme SN events  (σ > {sigma})",
                         style={"fontSize": "11px", "fontWeight": "700",
                                "letterSpacing": "0.07em", "textTransform": "uppercase",
                                "color": MUTED, "marginBottom": "14px"}),
                _tbl(sn_ev, "sn", C_SN),
            ], style={"flex": 1, "background": SURFACE, "border": f"1px solid {BORDER}",
                      "borderRadius": "10px", "padding": "20px 22px"}),
            html.Div([
                html.Div(f"Extreme Kp events  (σ > {sigma})",
                         style={"fontSize": "11px", "fontWeight": "700",
                                "letterSpacing": "0.07em", "textTransform": "uppercase",
                                "color": MUTED, "marginBottom": "14px"}),
                _tbl(kp_ev, "Kp_max", C_KP),
            ], style={"flex": 1, "background": SURFACE, "border": f"1px solid {BORDER}",
                      "borderRadius": "10px", "padding": "20px 22px"}),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "16px", "flexWrap": "wrap"}),
        _section("Annotated timeline",
                 dcc.Graph(figure=fig_tl, config={"displayModeBar": False})),
        _section("Rolling z-score",
                 dcc.Graph(figure=fig_z, config={"displayModeBar": False})),
    ])


# ---------------------------------------------------------------------------
# Tab 4 — Smoothing
# ---------------------------------------------------------------------------

def _tab_smoothing(df):
    fig = _fig(
        height=380,
        yaxis=dict(title="Sunspot Number", gridcolor=CHART_GRID, zeroline=False,
                   tickfont=dict(color=MUTED, size=11)),
        legend=dict(orientation="h", y=-0.16, font=dict(size=11, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
    )
    for col, name, color, opacity, width in [
        ("sn",                "Raw daily",         C_SN,   0.22, 1.0),
        ("sn_roll27d",        "27-day rolling",    C_R27,  1.0,  2.0),
        ("sn_roll365d",       "365-day rolling",   C_R365, 1.0,  2.0),
        ("sn_cycle_baseline", "Savitzky-Golay",    C_BASE, 1.0,  2.5),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name,
                                     line=dict(color=color, width=width), opacity=opacity))

    fig_r = _fig(
        height=240,
        yaxis=dict(title="Residual", gridcolor=CHART_GRID, zeroline=True,
                   zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    if "sn" in df.columns and "sn_roll365d" in df.columns:
        resid = (df["sn"] - df["sn_roll365d"]).dropna()
        if not resid.empty:
            rm = resid.resample("ME").mean().dropna()
            fig_r.add_trace(go.Bar(
                x=rm.index, y=rm.values, showlegend=False,
                marker_color=[C_SN if v >= 0 else C_ANOM for v in rm.values],
            ))

    TH = {"padding": "8px 14px", "fontSize": "10px", "fontWeight": "700",
          "letterSpacing": "0.06em", "textTransform": "uppercase",
          "color": MUTED, "borderBottom": f"1px solid {BORDER}", "textAlign": "left"}
    TD_s = {"padding": "10px 14px", "fontSize": "12px", "fontWeight": "600",
             "color": C_SN, "verticalAlign": "top",
             "borderBottom": f"1px solid {BORDER}", "whiteSpace": "nowrap"}
    TD_d = {"padding": "10px 14px", "fontSize": "12px", "color": MUTED,
             "lineHeight": "1.6", "borderBottom": f"1px solid {BORDER}"}

    limits = [
        ("SILSO SN v2 (daily)",
         "Composite from ~80 observatories. v2 rescaling in 2015 creates a step discontinuity "
         "vs pre-2015 data. Gaps (originally -1) may indicate cloud cover rather than true zero activity."),
        ("SILSO SN v2 (monthly)",
         "Within-month variability is smoothed out; Carrington rotation (~27 d) is partly aliased."),
        ("GFZ Kp (3-hourly, 1932-present)",
         "Semi-logarithmic index derived from 13 sub-auroral stations; coarsely quantised "
         "in thirds (0, 0+, 1-, 1, ...). Station network and calibration have changed over the decades — "
         "trend comparisons across the full archive should account for this."),
        ("NOAA Kp (3-hourly, recent)",
         "Real-time feed covering the last ~7 days. Merged on top of the GFZ archive; "
         "NOAA values take priority where timestamps overlap."),
        ("Coverage",
         "Systematic sunspot photography began ~1848; daily SN available from 1818. "
         "Kp begins 1932. The joint SN+Kp analysis window is 1932-present (~90 years)."),
        ("Index construction",
         "Both are activity proxies, not direct physical measurements. "
         "Kp cannot distinguish sudden storm commencements from isolated substorm activity."),
    ]

    return html.Div([
        _section("Smoothing comparison",
                 dcc.Graph(figure=fig, config={"displayModeBar": False})),
        _section("Residuals: raw SN minus 365-day rolling mean",
                 dcc.Graph(figure=fig_r, config={"displayModeBar": False})),
        _section("Strategy notes", [
            html.P([html.Strong("27-day rolling — ", style={"color": C_R27}),
                    "Tracks the Carrington rotation. Suppresses day-to-day noise while "
                    "preserving medium-term variability; responds quickly to activity shifts."],
                   style={"fontSize": "13px", "color": MUTED, "lineHeight": "1.7",
                          "margin": "0 0 10px"}),
            html.P([html.Strong("365-day rolling — ", style={"color": C_R365}),
                    "Annual smoothing captures cycle rise/decline phases. "
                    "Best for year-on-year comparisons."],
                   style={"fontSize": "13px", "color": MUTED, "lineHeight": "1.7",
                          "margin": "0 0 10px"}),
            html.P([html.Strong("Savitzky-Golay (4-yr window, order 3) — ", style={"color": C_BASE}),
                    "Polynomial fit preserving peak shape. Avoids the phase-shift artefact "
                    "of centred rolling means at data boundaries."],
                   style={"fontSize": "13px", "color": MUTED, "lineHeight": "1.7",
                          "margin": "0 0 10px"}),
            html.P("Iterative refinement: daily resolution was chosen after finding that the "
                   "1–4 day CME travel-time lag was completely invisible at monthly granularity. "
                   "The Kp dataset was expanded from a 7-day NOAA feed to the full GFZ archive "
                   "(1932-present) to give the cross-correlation analysis a statistically meaningful "
                   "sample size.",
                   style={"fontSize": "12px", "color": MUTED, "fontStyle": "italic",
                          "borderLeft": f"2px solid {BORDER}", "paddingLeft": "10px",
                          "lineHeight": "1.6", "margin": 0}),
        ]),
        _section("Data limitations",
                 html.Table(
                     [html.Tr([html.Th("Source", style=TH),
                                html.Th("Limitation", style=TH)])] +
                     [html.Tr([html.Td(s, style=TD_s), html.Td(d, style=TD_d)])
                      for s, d in limits],
                     style={"borderCollapse": "collapse", "width": "100%"},
                 )),
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, port=8050)
