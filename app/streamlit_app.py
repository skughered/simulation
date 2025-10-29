import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit.components.v1 as components
from pathlib import Path

from riskboot.config import DEFAULT_MONTHS, DEFAULT_SCENS, SEED_SIM, BLOCK_RANGE, TREND_WEIGHTS_FILENAME, ALL_ASSETS_FILENAME
from riskboot.simulate import simulate_portfolios
from riskboot.data import load_trend_weights, parse_meta_csv

# Override DATA_DIR for deployment
DATA_DIR = Path(__file__).parent.parent / "riskboot" / "data"

# ---------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------
st.set_page_config(page_title="Portfolio Risk Simulator", layout="wide")
st.title("📈 Portfolio Risk Simulator")

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

    # Load meta data for public assets
    _, meta_df = parse_meta_csv(DATA_DIR, ALL_ASSETS_FILENAME)
    public_assets = meta_df[meta_df['public']].index.tolist()
    public_names = meta_df[meta_df['public']]['name'].tolist()

    # Create dynamic weight inputs for public assets
    weights = {}
    for ticker, name in zip(public_assets, public_names):
        weights[ticker] = st.number_input(f"{name} %", 0.0, 100.0, 0.0, 1.0, key=f'w_{ticker}')

    total = sum(weights.values())
    if total != 100.0 and total > 0:
        st.warning(f"Weights sum to {total:.1f}%. They will be normalised to 100%.")
        weights = {k: v / total * 100 for k, v in weights.items()}
    elif total == 0:
        st.warning("All weights are zero. Please set at least one weight.")

    seed = st.number_input("Random seed", 0, 10_000_000, SEED_SIM, 1)
    vol_increase_pct = st.slider("Volatility Increase", 0.0, 20.0, 0.0, 0.5)
    vol_increase = vol_increase_pct / 100 if vol_increase_pct > 0 else None

    # Load trend portfolio options
    trend_weights_df = load_trend_weights(DATA_DIR, TREND_WEIGHTS_FILENAME)
    trend_options = ["None"] + [col for col in trend_weights_df.columns if not col.startswith('BM')]
    trend_portfolio = st.selectbox("Trend Portfolio", trend_options, index=0)

    # Load benchmark portfolio options
    benchmark_options = ["None"] + [col for col in trend_weights_df.columns if col.startswith('BM')]
    if benchmark_options:
        benchmark_portfolio = st.selectbox("Benchmark Portfolio", benchmark_options, index=0)
    else:
        benchmark_portfolio = None

    run = st.button("Run simulation", type="primary")

# ---------------------------------------------------
# Cached simulation run
# ---------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _run_sim(weights, months, scens, seed, lookback, block_low, block_high, vol_increase, trend_portfolio, benchmark_portfolio):
    return simulate_portfolios(
        weights=weights,
        months=months,
        n_scenarios=scens,
        seed=seed,
        lookback=lookback,
        block_range=(block_low, block_high),
        vol_increase=vol_increase,
        trend_portfolio=trend_portfolio,
        benchmark_portfolio=benchmark_portfolio
    )

# ---------------------------------------------------
# When "Run" is clicked
# ---------------------------------------------------
if run:
    with st.spinner("Simulating scenarios..."):
        out = _run_sim(weights, months, scens, seed, lookback, block_low, block_high, vol_increase, trend_portfolio, benchmark_portfolio)
    st.success("Done!")

    # ---- Metrics tables
    def to_df(metrics: dict, label: str) -> pd.DataFrame:
        return pd.DataFrame(metrics).assign(Type=label)

    df_static = to_df(out["static"]["metrics"], "Static")
    df_all = pd.concat([df_static], ignore_index=True)
    if "trend" in out:
        df_trend = to_df(out["trend"]["metrics"], "Trend")
        df_all = pd.concat([df_all, df_trend], ignore_index=True)
    if "benchmark" in out:
        df_benchmark = to_df(out["benchmark"]["metrics"], "Benchmark")
        df_all = pd.concat([df_all, df_benchmark], ignore_index=True)

    st.markdown("---")

    # ---- Metrics display
    st.subheader("Static Portfolio Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("5th MaxDD", f"{np.percentile(df_static['MaxDD'], 5) * 100:.1f}%")
        st.metric("Median MaxDD", f"{np.median(df_static['MaxDD']) * 100:.1f}%")
        st.metric("95th MaxDD", f"{np.percentile(df_static['MaxDD'], 95) * 100:.1f}%")
    with c2:
        st.metric("5th AnnVol", f"{np.percentile(df_static['AnnVol'], 5) * 100:.1f}%")
        st.metric("Mean AnnVol", f"{np.mean(df_static['AnnVol']) * 100:.1f}%")
        st.metric("95th AnnVol", f"{np.percentile(df_static['AnnVol'], 95) * 100:.1f}%")
    with c3:
        st.metric("5th AnnRet", f"{np.percentile(df_static['AnnReturn'], 5) * 100:.1f}%")
        st.metric("Median AnnRet", f"{np.median(df_static['AnnReturn']) * 100:.1f}%")
        st.metric("95th AnnRet", f"{np.percentile(df_static['AnnReturn'], 95) * 100:.1f}%")

    if "trend" in out:
        st.subheader("Trend Portfolio Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("5th MaxDD", f"{np.percentile(df_trend['MaxDD'], 5) * 100:.1f}%")
            st.metric("Median MaxDD", f"{np.median(df_trend['MaxDD']) * 100:.1f}%")
            st.metric("95th MaxDD", f"{np.percentile(df_trend['MaxDD'], 95) * 100:.1f}%")
        with c2:
            st.metric("5th AnnVol", f"{np.percentile(df_trend['AnnVol'], 5) * 100:.1f}%")
            st.metric("Mean AnnVol", f"{np.mean(df_trend['AnnVol']) * 100:.1f}%")
            st.metric("95th AnnVol", f"{np.percentile(df_trend['AnnVol'], 95) * 100:.1f}%")
        with c3:
            st.metric("5th AnnRet", f"{np.percentile(df_trend['AnnReturn'], 5) * 100:.1f}%")
            st.metric("Median AnnRet", f"{np.median(df_trend['AnnReturn']) * 100:.1f}%")
            st.metric("95th AnnRet", f"{np.percentile(df_trend['AnnReturn'], 95) * 100:.1f}%")

    if "benchmark" in out:
        st.subheader("Benchmark Portfolio Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("5th MaxDD", f"{np.percentile(df_benchmark['MaxDD'], 5) * 100:.1f}%")
            st.metric("Median MaxDD", f"{np.median(df_benchmark['MaxDD']) * 100:.1f}%")
            st.metric("95th MaxDD", f"{np.percentile(df_benchmark['MaxDD'], 95) * 100:.1f}%")
        with c2:
            st.metric("5th AnnVol", f"{np.percentile(df_benchmark['AnnVol'], 5) * 100:.1f}%")
            st.metric("Mean AnnVol", f"{np.mean(df_benchmark['AnnVol']) * 100:.1f}%")
            st.metric("95th AnnVol", f"{np.percentile(df_benchmark['AnnVol'], 95) * 100:.1f}%")
        with c3:
            st.metric("5th AnnRet", f"{np.percentile(df_benchmark['AnnReturn'], 5) * 100:.1f}%")
            st.metric("Median AnnRet", f"{np.median(df_benchmark['AnnReturn']) * 100:.1f}%")
            st.metric("95th AnnRet", f"{np.percentile(df_benchmark['AnnReturn'], 95) * 100:.1f}%")

    st.markdown("---")

    # ---- Histograms (direct array plotting)
    colA, colB, colC = st.columns(3)

    # Filter clean data
    df_static_clean = df_all[df_all["Type"] == "Static"].dropna(subset=["AnnReturn", "MaxDD", "AnnVol"])
    df_trend_clean = None
    if "trend" in out:
        df_trend_clean = df_all[df_all["Type"] == "Trend"].dropna(subset=["AnnReturn", "MaxDD", "AnnVol"])
        df_trend_clean = df_trend_clean[np.isfinite(df_trend_clean["AnnReturn"]) & np.isfinite(df_trend_clean["MaxDD"]) & np.isfinite(df_trend_clean["AnnVol"])]
    df_benchmark_clean = None
    if "benchmark" in out:
        df_benchmark_clean = df_all[df_all["Type"] == "Benchmark"].dropna(subset=["AnnReturn", "MaxDD", "AnnVol"])
        df_benchmark_clean = df_benchmark_clean[np.isfinite(df_benchmark_clean["AnnReturn"]) & np.isfinite(df_benchmark_clean["MaxDD"]) & np.isfinite(df_benchmark_clean["AnnVol"])]

    # Remove inf values as well
    df_static_clean = df_static_clean[np.isfinite(df_static_clean["AnnReturn"]) & np.isfinite(df_static_clean["MaxDD"]) & np.isfinite(df_static_clean["AnnVol"])]

    with colA:
        fig_r = go.Figure()
        fig_r.add_trace(go.Histogram(
            x=(df_static_clean["AnnReturn"] * 100).tolist(),
            name="Static",
            nbinsx=50,
            marker=dict(color="blue"),
            histnorm='probability density'
        ))
        if df_trend_clean is not None:
            fig_r.add_trace(go.Histogram(
                x=(df_trend_clean["AnnReturn"] * 100).tolist(),
                name="Trend",
                nbinsx=50,
                marker=dict(color="orange"),
                histnorm='probability density'
            ))
        if df_benchmark_clean is not None:
            fig_r.add_trace(go.Histogram(
                x=(df_benchmark_clean["AnnReturn"] * 100).tolist(),
                name="Benchmark",
                nbinsx=50,
                marker=dict(color="green"),
                histnorm='probability density'
            ))
        fig_r.update_layout(
            title="Distribution of Annualised Returns",
            xaxis_title="Annualised Return (%)",
            yaxis_title="Probability Density",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with colB:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Histogram(
            x=(df_static_clean["MaxDD"] * 100).tolist(),
            name="Static",
            nbinsx=50,
            marker=dict(color="blue"),
            histnorm='probability density'
        ))
        if df_trend_clean is not None:
            fig_dd.add_trace(go.Histogram(
                x=(df_trend_clean["MaxDD"] * 100).tolist(),
                name="Trend",
                nbinsx=50,
                marker=dict(color="orange"),
                histnorm='probability density'
            ))
        if df_benchmark_clean is not None:
            fig_dd.add_trace(go.Histogram(
                x=(df_benchmark_clean["MaxDD"] * 100).tolist(),
                name="Benchmark",
                nbinsx=50,
                marker=dict(color="green"),
                histnorm='probability density'
            ))
        fig_dd.update_layout(
            title="Distribution of Maximum Drawdowns",
            xaxis_title="Maximum Drawdown (%)",
            yaxis_title="Probability Density",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with colC:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Histogram(
            x=(df_static_clean["AnnVol"] * 100).tolist(),
            name="Static",
            nbinsx=50,
            marker=dict(color="blue"),
            histnorm='probability density'
        ))
        if df_trend_clean is not None:
            fig_vol.add_trace(go.Histogram(
                x=(df_trend_clean["AnnVol"] * 100).tolist(),
                name="Trend",
                nbinsx=50,
                marker=dict(color="orange"),
                histnorm='probability density'
            ))
        if df_benchmark_clean is not None:
            fig_vol.add_trace(go.Histogram(
                x=(df_benchmark_clean["AnnVol"] * 100).tolist(),
                name="Benchmark",
                nbinsx=50,
                marker=dict(color="green"),
                histnorm='probability density'
            ))
        fig_vol.update_layout(
            title="Distribution of Annualised Volatilities",
            xaxis_title="Annualised Volatility (%)",
            yaxis_title="Probability Density",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig_vol, use_container_width=True)


    # ---- Combined fan chart (same y-scale)
    def combined_fan_figure(bands_static, bands_trend=None, bands_benchmark=None):
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
        if bands_trend is not None:
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

        # --- Benchmark (green)
        if bands_benchmark is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([bands_benchmark["q95"], bands_benchmark["q05"][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 128, 0, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Benchmark 90% range'
            ))
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([bands_benchmark["q75"], bands_benchmark["q25"][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 128, 0, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Benchmark 50% range'
            ))
            fig.add_trace(go.Scatter(
                x=x, y=bands_benchmark["q50"],
                line=dict(color='green', width=2),
                name='Benchmark median'
            ))

        # --- Shared Y range
        all_bands = [bands_static]
        if bands_trend is not None:
            all_bands.append(bands_trend)
        if bands_benchmark is not None:
            all_bands.append(bands_benchmark)
        ymax = max(np.nanmax(b["q95"]) for b in all_bands) * 1.1
        ymin = min(np.nanmin(b["q05"]) for b in all_bands) * 0.9

        # Dynamic title
        title_parts = ["Static"]
        if bands_trend is not None:
            title_parts.append("Trend")
        if bands_benchmark is not None:
            title_parts.append("Benchmark")
        title = " vs ".join(title_parts) + " Portfolio Wealth Fans"

        fig.update_layout(
            title=title,
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
    trend_bands = out.get("trend", {}).get("bands")
    benchmark_bands = out.get("benchmark", {}).get("bands")
    fig_combined = combined_fan_figure(out["static"]["bands"], trend_bands, benchmark_bands)
    html_combined = pio.to_html(fig_combined, full_html=False, include_plotlyjs="cdn")
    components.html(html_combined, height=750, scrolling=True)

    st.markdown("----")
    st.caption(
        "Method: joint stationary bootstrap (synchronous blocks) over monthly returns for equities, bonds, and cash. "
        "Trend overlay applied per asset (lookback positive → invested; otherwise cash)."
    )

else:
    st.info("Set your inputs in the sidebar and click **Run simulation**.")
