import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit.components.v1 as components

from riskboot.config import DEFAULT_MONTHS, DEFAULT_SCENS, SEED_SIM, BLOCK_RANGE
from riskboot.simulate import simulate_portfolios

# ---------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------
st.set_page_config(page_title="Portfolio Risk Simulator", layout="wide")
st.title("📈 Portfolio Risk Simulator (Static vs Trend)")

# ---------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------
with st.sidebar:
    st.header("Inputs")
    years = st.slider("Projection horizon (years)", 5, 40, 20, 1)
    months = years * 12

    scens = st.slider("Number of simulations", 500, 10000, 4000, 500)
    lookback = st.slider("Trend lookback (months)", 3, 12, 6, 1)
    block_low, block_high = st.slider("Bootstrap mean block length (months)", 3, 24, (6, 12))

    st.subheader("Weights (must sum to 100%)")
    w_eq = st.number_input("Global Equity %", 0, 100, 60, 5)
    w_bd = st.number_input("Government Bonds %", 0, 100, 30, 5)
    w_csh = st.number_input("Cash %", 0, 100, 10, 5)

    total = w_eq + w_bd + w_csh
    if total != 100:
        st.warning(f"Weights sum to {total}%. They will be normalised to 100%.")
    w_eq, w_bd, w_csh = [w / max(total, 1) for w in (w_eq, w_bd, w_csh)]
    weights = {"stocks": w_eq, "bonds": w_bd, "rf_1m": w_csh}

    seed = st.number_input("Random seed", 0, 10_000_000, SEED_SIM, 1)
    run = st.button("Run simulation", type="primary")

# ---------------------------------------------------
# Cached simulation run
# ---------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _run_sim(weights, months, scens, seed, lookback, block_low, block_high):
    return simulate_portfolios(
        weights=weights,
        months=months,
        n_scenarios=scens,
        seed=seed,
        lookback=lookback,
        block_range=(block_low, block_high)
    )

# ---------------------------------------------------
# When "Run" is clicked
# ---------------------------------------------------
if run:
    with st.spinner("Simulating scenarios..."):
        out = _run_sim(weights, months, scens, seed, lookback, block_low, block_high)
    st.success("Done!")

    # ---- Metrics tables
    def to_df(metrics: dict, label: str) -> pd.DataFrame:
        return pd.DataFrame(metrics).assign(Type=label)

    df_static = to_df(out["static"]["metrics"], "Static")
    df_trend = to_df(out["trend"]["metrics"], "Trend")
    df_all = pd.concat([df_static, df_trend], ignore_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Static: median MaxDD", f"{np.median(df_static['MaxDD']) * 100:.1f}%")
        st.metric("Trend: median MaxDD", f"{np.median(df_trend['MaxDD']) * 100:.1f}%")
    with c2:
        st.metric("Static: mean AnnVol", f"{np.mean(df_static['AnnVol']) * 100:.1f}%")
        st.metric("Trend: mean AnnVol", f"{np.mean(df_trend['AnnVol']) * 100:.1f}%")
    with c3:
        st.metric("Static: median AnnRet", f"{np.median(df_static['AnnReturn']) * 100:.1f}%")
        st.metric("Trend: median AnnRet", f"{np.median(df_trend['AnnReturn']) * 100:.1f}%")

    st.markdown("---")

    st.write("DEBUG df_all head:")
    st.dataframe(df_all.head())
    st.write("Unique Types:", df_all["Type"].unique())
    st.write("NaNs in AnnReturn:", df_all["AnnReturn"].isna().sum())
    st.write("NaNs in MaxDD:", df_all["MaxDD"].isna().sum())
    st.write("Infs in AnnReturn:", np.isinf(df_all["AnnReturn"]).sum())
    st.write("Infs in MaxDD:", np.isinf(df_all["MaxDD"]).sum())
    # # ---- Histograms (explicit data arrays)
    # ---- Histograms (direct array plotting)
    colA, colB = st.columns(2)

    # Filter clean data
    df_static_clean = df_all[df_all["Type"] == "Static"].dropna(subset=["AnnReturn", "MaxDD"])
    df_trend_clean = df_all[df_all["Type"] == "Trend"].dropna(subset=["AnnReturn", "MaxDD"])

    with colA:
        fig_r = go.Figure()
        fig_r.add_trace(go.Histogram(
            x=df_static_clean["AnnReturn"] * 100,
            name="Static",
            nbinsx=25,
            opacity=0.6,
            marker=dict(color="blue")
        ))
        fig_r.add_trace(go.Histogram(
            x=df_trend_clean["AnnReturn"] * 100,
            name="Trend",
            nbinsx=25,
            opacity=0.6,
            marker=dict(color="orange")
        ))
        fig_r.update_layout(
            title="Distribution of Annualised Returns",
            xaxis_title="Annualised Return (%)",
            yaxis_title="Frequency",
            barmode="overlay",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with colB:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Histogram(
            x=df_static_clean["MaxDD"] * 100,
            name="Static",
            nbinsx=25,
            opacity=0.6,
            marker=dict(color="blue")
        ))
        fig_dd.add_trace(go.Histogram(
            x=df_trend_clean["MaxDD"] * 100,
            name="Trend",
            nbinsx=25,
            opacity=0.6,
            marker=dict(color="orange")
        ))
        fig_dd.update_layout(
            title="Distribution of Maximum Drawdowns",
            xaxis_title="Maximum Drawdown (%)",
            yaxis_title="Frequency",
            barmode="overlay",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_dd, use_container_width=True)


    # ---- Combined fan chart (same y-scale)
    def combined_fan_figure(bands_static, bands_trend):
        x = np.arange(len(bands_static["q50"]))
        fig = go.Figure()

        # --- Static (blue)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([bands_static["q95"], bands_static["q05"][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Static 90% range'
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([bands_static["q75"], bands_static["q25"][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Static 50% range'
        ))
        fig.add_trace(go.Scatter(
            x=x, y=bands_static["q50"],
            line=dict(color='blue', width=2),
            name='Static median'
        ))

        # --- Trend (orange)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([bands_trend["q95"], bands_trend["q05"][::-1]]),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Trend 90% range'
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([bands_trend["q75"], bands_trend["q25"][::-1]]),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Trend 50% range'
        ))
        fig.add_trace(go.Scatter(
            x=x, y=bands_trend["q50"],
            line=dict(color='orange', width=2),
            name='Trend median'
        ))

        # --- Shared Y range
        ymax = max(np.nanmax(bands_static["q95"]), np.nanmax(bands_trend["q95"])) * 1.1
        ymin = min(np.nanmin(bands_static["q05"]), np.nanmin(bands_trend["q05"])) * 0.9

        fig.update_layout(
            title="Static vs Trend Portfolio Wealth Fans",
            xaxis_title="Months",
            yaxis_title="Wealth Index (start=1.0)",
            yaxis=dict(range=[ymin, ymax]),
            template="plotly_white",
            height=700,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        return fig

    # ---- Render combined fan chart
    fig_combined = combined_fan_figure(out["static"]["bands"], out["trend"]["bands"])
    html_combined = pio.to_html(fig_combined, full_html=False, include_plotlyjs="cdn")
    components.html(html_combined, height=750, scrolling=True)

    st.markdown("----")
    st.caption(
        "Method: joint stationary bootstrap (synchronous blocks) over monthly returns for equities, bonds, and cash. "
        "Trend overlay applied per asset (lookback positive → invested; otherwise cash)."
    )

else:
    st.info("Set your inputs in the sidebar and click **Run simulation**.")
