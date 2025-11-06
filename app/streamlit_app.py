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
st.set_page_config(page_title="Simulator", layout="wide")
st.title("📈 Simulator")

# ---------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------
with st.sidebar:
    st.header("Inputs")
    years = st.slider("Projection horizon (years)", 5, 50, 20, 1)
    months = years * 12

    scens = st.slider("Number of simulations", 500, 10000, 4000, 500)
    lookback = st.slider("Trend lookback (months)", 3, 12, 6, 1)
    block_low, block_high = st.slider("Bootstrap mean block length (months)", 3, 24, (6, 12))

    st.subheader("Weights (must sum to 100%)")
    st.caption("(AnnRets%, -MaxDD%, AnnVol%)")

    # Load meta data for public assets
    df_hist, meta_df = parse_meta_csv(DATA_DIR, ALL_ASSETS_FILENAME)
    public_assets = meta_df[meta_df['public']].index.tolist()
    public_names = meta_df[meta_df['public']]['name'].tolist()

    # Calculate or load historical stats for public assets
    stats_file = DATA_DIR / "asset_stats.csv"
    if not stats_file.exists():
        stats = []
        for ticker in public_assets:
            if ticker in df_hist.columns:
                returns = df_hist[ticker].dropna()
                if len(returns) > 0:
                    cumulative = (1 + returns).cumprod()
                    ann_ret = (cumulative.iloc[-1] ** (12 / len(returns)) - 1) * 100
                    running_max = cumulative.expanding().max()
                    drawdown = cumulative / running_max - 1
                    max_dd = drawdown.min() * 100
                    ann_vol = returns.std() * (12 ** 0.5) * 100
                    stats.append([ticker, ann_ret, max_dd, ann_vol])
        stats_df = pd.DataFrame(stats, columns=['ticker', 'ann_ret', 'max_dd', 'ann_vol'])
        stats_df.to_csv(stats_file, index=False)
    else:
        stats_df = pd.read_csv(stats_file)

    stats_dict = stats_df.set_index('ticker').to_dict('index')

    # Create dynamic weight inputs for public assets
    weights = {}
    for ticker, name in zip(public_assets, public_names):
        s = stats_dict.get(ticker, {'ann_ret': 0, 'max_dd': 0, 'ann_vol': 0})
        ann_ret = s['ann_ret']
        max_dd = s['max_dd']
        ann_vol = s['ann_vol']
        label = f"{name} % - ({ann_ret:.1f}%, {max_dd:.1f}%, {ann_vol:.1f}%)"
        weights[ticker] = st.number_input(label, 0.0, 100.0, 0.0, 1.0, key=f'w_{ticker}')

    total = sum(weights.values())
    if total != 100.0 and total > 0:
        st.warning(f"Weights sum to {total:.1f}%. They will be normalised to 100%.")
        weights = {k: v / total * 100 for k, v in weights.items()}
    elif total == 0:
        st.warning("All weights are zero. Please set at least one weight.")

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

    # Cashflow / sequencing risk option
    st.markdown("---")
    cashflow_enabled = st.checkbox("Enable cash flow (sequencing) analysis", value=False)
    if cashflow_enabled:
        st.subheader("Cashflow inputs")
        start_value = st.number_input("Starting portfolio value (£)", min_value=0.0, value=1000000.0, step=1000.0)
        inflation_pct = st.number_input("Inflation (annual %)", min_value=0.0, max_value=100.0, value=2.0, step=0.1) / 100.0
        st.caption("Enter desired withdrawal per year (nominal). Leave zeros for none.")
        annual_withdrawals = []
        # Use st.number_input with unique keys per year
        for y in range(1, years + 1):
            default = 0.0
            if y == 1:
                default = round(start_value * 0.04)
            val = st.number_input(f"Year {y} withdrawal (£)", min_value=0.0, value=float(default), step=100.0, key=f"withdraw_{y}")
            annual_withdrawals.append(val)
    else:
        start_value = None
        inflation_pct = 0.0
        annual_withdrawals = None

    run = st.button("Run simulation", type="primary")

def fmt_pct(val):
    """
    Safely convert val to a float and format as percent string.
    Returns 'n/a' when not available.
    """
    try:
        # handle numpy scalars/size-1 arrays
        v = float(np.asarray(val).item())
    except Exception:
        try:
            v = float(val)
        except Exception:
            v = np.nan
    if np.isnan(v):
        return "n/a"
    return f"{v * 100:.1f}%"

# --- Helper: sanitize percentile bands for plotting (handle NaNs/empty)
def _sanitize_bands(bands: dict | None):
    if not bands or not isinstance(bands, dict):
        return None
    q50 = np.asarray(bands.get("q50", []), dtype=float)
    if q50.size == 0 or not np.isfinite(q50).any():
        return None
    cleaned = {}
    for k, v in bands.items():
        arr = np.asarray(v, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            cleaned[k] = None
            continue
        if np.isnan(arr).any():
            idx = np.arange(arr.size)
            good = np.isfinite(arr)
            if good.sum() == 0:
                arr = np.zeros_like(arr)
            else:
                arr[np.isnan(arr)] = np.interp(idx[np.isnan(arr)], idx[good], arr[good])
        cleaned[k] = arr
    return cleaned

# --- Combined fan figure (module-level, safe) ---
def combined_fan_figure(bands_static, bands_trend=None, bands_benchmark=None, months=None, is_wealth_index=True, starting_value=1.0):
    b_static = _sanitize_bands(bands_static) if bands_static is not None else None
    b_trend = _sanitize_bands(bands_trend) if bands_trend is not None else None
    b_bench = _sanitize_bands(bands_benchmark) if bands_benchmark is not None else None

    first = b_static or b_trend or b_bench
    if not first:
        if months is None:
            x = np.arange(12) / 12.0
        else:
            x = np.arange(months) / 12.0
    else:
        x = np.arange(len(first["q50"])) / 12.0

    fig = go.Figure()

    def _add_band(band, fillcol, mediancol, label):
        if not band:
            return
        q05 = band.get("q05")
        q50 = band.get("q50")
        q95 = band.get("q95")
        if q05 is None or q50 is None or q95 is None:
            return
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([q95, q05[::-1]]),
            fill='toself',
            fillcolor=fillcol,
            line=dict(color=mediancol, width=1),
            hoverinfo='skip',
            name=f'{label} 90% range'
        ))
        fig.add_trace(go.Scatter(x=x, y=q50, line=dict(color=mediancol, width=3), mode='lines+markers', marker=dict(size=2), name=f'{label} median'))

    _add_band(b_static, 'rgba(173,216,230,0.8)', 'blue', 'Static')
    _add_band(b_trend, 'rgba(255,165,0,0.8)', 'orange', 'Trend')
    _add_band(b_bench, 'rgba(0,128,0,0.8)', 'green', 'Benchmark')

    # Add starting balance line
    fig.add_trace(go.Scatter(
        x=[x[0], x[-1]], y=[starting_value, starting_value],
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='Starting Balance'
    ))

    all_vals = []
    for band in (b_static, b_trend, b_bench):
        if band:
            for key in ("q05", "q95"):
                arr = band.get(key)
                if arr is not None:
                    all_vals.append(arr)
    if all_vals:
        stacked = np.concatenate([a.ravel() for a in all_vals])
        finite = stacked[np.isfinite(stacked)]
        if finite.size > 0:
            ymax = finite.max() * 1.1
            ymin = finite.min() * 0.9
            if not is_wealth_index:
                ymin = 0
        else:
            ymax, ymin = 1.1, 0.9
    else:
        ymax, ymin = 1.1, 0.9

    title_parts = []
    if b_static:
        title_parts.append('Static')
    if b_trend:
        title_parts.append('Trend')
    if b_bench:
        title_parts.append('Benchmark')
    if is_wealth_index:
        title = ' vs '.join(title_parts) + ' Portfolio Wealth Fans' if title_parts else 'Portfolio Wealth Fans'
        yaxis_title = 'Wealth Index (start=1.0)'
    else:
        title = ' vs '.join(title_parts) + ' Portfolio Wealth After Withdrawals Fans' if title_parts else 'Portfolio Wealth After Withdrawals Fans'
        yaxis_title = 'Wealth (£)'

    fig.update_layout(
        title=title,
        xaxis_title='Years',
        yaxis_title=yaxis_title,
        yaxis=dict(range=[ymin, ymax]),
        template='plotly_white',
        height=700,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
    )
    return fig

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
        out = _run_sim(weights, months, scens, SEED_SIM, lookback, block_low, block_high, None, trend_portfolio, benchmark_portfolio)
    st.success("Done!")

    # ---- Metrics tables
    def to_df(metrics: dict, label: str) -> pd.DataFrame:
        return pd.DataFrame(metrics).assign(Type=label)

    df_all = pd.DataFrame()
    if "static" in out:
        df_static = to_df(out["static"]["metrics"], "Static")
        df_all = pd.concat([df_all, df_static], ignore_index=True)
    if "trend" in out:
        df_trend = to_df(out["trend"]["metrics"], "Trend")
        df_all = pd.concat([df_all, df_trend], ignore_index=True)
    if "benchmark" in out:
        df_benchmark = to_df(out["benchmark"]["metrics"], "Benchmark")
        df_all = pd.concat([df_all, df_benchmark], ignore_index=True)

    st.markdown("---")

    # ---- Metrics display
    if "static" in out:
        st.subheader("Static Portfolio Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("5th AnnRet", f"{np.percentile(df_static['AnnReturn'], 5) * 100:.1f}%")
            st.metric("Median AnnRet", f"{np.median(df_static['AnnReturn']) * 100:.1f}%")
            st.metric("95th AnnRet", f"{np.percentile(df_static['AnnReturn'], 95) * 100:.1f}%")

        with c2:
            st.metric("5th MaxDD", f"{np.percentile(df_static['MaxDD'], 5) * 100:.1f}%")
            st.metric("Median MaxDD", f"{np.median(df_static['MaxDD']) * 100:.1f}%")
            st.metric("95th MaxDD", f"{np.percentile(df_static['MaxDD'], 95) * 100:.1f}%")
            st.metric("Windowed MaxDD 5th", fmt_pct(out['static']['metrics'].get('WindowedMaxDD5th', np.nan)))
            st.metric("Windowed MaxDD 25th", fmt_pct(out['static']['metrics'].get('WindowedMaxDD25th', np.nan)))

        with c3:
            st.metric("5th AnnVol", f"{np.percentile(df_static['AnnVol'], 5) * 100:.1f}%")
            st.metric("Median AnnVol", f"{np.median(df_static['AnnVol']) * 100:.1f}%")
            st.metric("95th AnnVol", f"{np.percentile(df_static['AnnVol'], 95) * 100:.1f}%")

    if "trend" in out:
        st.subheader("Trend Portfolio Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:

            st.metric("5th AnnRet", f"{np.percentile(df_trend['AnnReturn'], 5) * 100:.1f}%")
            st.metric("Median AnnRet", f"{np.median(df_trend['AnnReturn']) * 100:.1f}%")
            st.metric("95th AnnRet", f"{np.percentile(df_trend['AnnReturn'], 95) * 100:.1f}%")
        with c2:
            st.metric("5th MaxDD", f"{np.percentile(df_trend['MaxDD'], 5) * 100:.1f}%")
            st.metric("Median MaxDD", f"{np.median(df_trend['MaxDD']) * 100:.1f}%")
            st.metric("95th MaxDD", f"{np.percentile(df_trend['MaxDD'], 95) * 100:.1f}%")
            st.metric("Windowed MaxDD 5th", fmt_pct(out['trend']['metrics'].get('WindowedMaxDD5th', np.nan)))
            st.metric("Windowed MaxDD 25th", fmt_pct(out['trend']['metrics'].get('WindowedMaxDD25th', np.nan)))

        with c3:
            st.metric("5th AnnVol", f"{np.percentile(df_trend['AnnVol'], 5) * 100:.1f}%")
            st.metric("Median AnnVol", f"{np.median(df_trend['AnnVol']) * 100:.1f}%")
            st.metric("95th AnnVol", f"{np.percentile(df_trend['AnnVol'], 95) * 100:.1f}%")

    if "benchmark" in out:
        st.subheader("Benchmark Portfolio Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("5th AnnRet", f"{np.percentile(df_benchmark['AnnReturn'], 5) * 100:.1f}%")
            st.metric("Median AnnRet", f"{np.median(df_benchmark['AnnReturn']) * 100:.1f}%")
            st.metric("95th AnnRet", f"{np.percentile(df_benchmark['AnnReturn'], 95) * 100:.1f}%")
        with c2:
            st.metric("5th MaxDD", f"{np.percentile(df_benchmark['MaxDD'], 5) * 100:.1f}%")
            st.metric("Median MaxDD", f"{np.median(df_benchmark['MaxDD']) * 100:.1f}%")
            st.metric("95th MaxDD", f"{np.percentile(df_benchmark['MaxDD'], 95) * 100:.1f}%")
            st.metric("Windowed MaxDD 5th", fmt_pct(out['benchmark']['metrics'].get('WindowedMaxDD5th', np.nan)))
            st.metric("Windowed MaxDD 25th", fmt_pct(out['benchmark']['metrics'].get('WindowedMaxDD25th', np.nan)))

        with c3:
            st.metric("5th AnnVol", f"{np.percentile(df_benchmark['AnnVol'], 5) * 100:.1f}%")
            st.metric("Median AnnVol", f"{np.median(df_benchmark['AnnVol']) * 100:.1f}%")
            st.metric("95th AnnVol", f"{np.percentile(df_benchmark['AnnVol'], 95) * 100:.1f}%")


    st.markdown("---")

    # ---- Histograms (direct array plotting)
    colA, colB, colC = st.columns(3)

    # Filter clean data
    df_static_clean = None
    if "static" in out:
        df_static_clean = df_all[df_all["Type"] == "Static"].dropna(subset=["AnnReturn", "MaxDD", "AnnVol"])
        df_static_clean = df_static_clean[np.isfinite(df_static_clean["AnnReturn"]) & np.isfinite(df_static_clean["MaxDD"]) & np.isfinite(df_static_clean["AnnVol"])]

    df_trend_clean = None
    if "trend" in out:
        df_trend_clean = df_all[df_all["Type"] == "Trend"].dropna(subset=["AnnReturn", "MaxDD", "AnnVol"])
        df_trend_clean = df_trend_clean[np.isfinite(df_trend_clean["AnnReturn"]) & np.isfinite(df_trend_clean["MaxDD"]) & np.isfinite(df_trend_clean["AnnVol"])]

    df_benchmark_clean = None
    if "benchmark" in out:
        df_benchmark_clean = df_all[df_all["Type"] == "Benchmark"].dropna(subset=["AnnReturn", "MaxDD", "AnnVol"])
        df_benchmark_clean = df_benchmark_clean[np.isfinite(df_benchmark_clean["AnnReturn"]) & np.isfinite(df_benchmark_clean["MaxDD"]) & np.isfinite(df_benchmark_clean["AnnVol"])]

    with colA:
        fig_r = go.Figure()
        if df_static_clean is not None:
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
        if df_static_clean is not None:
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
        if df_static_clean is not None:
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


    # ---- Render combined fan chart
    static_bands = out.get("static", {}).get("bands") if "static" in out else None
    trend_bands = out.get("trend", {}).get("bands")
    benchmark_bands = out.get("benchmark", {}).get("bands")
    fig_combined = combined_fan_figure(static_bands, trend_bands, benchmark_bands)
    # st.plotly_chart(fig_combined, use_container_width=True)
    html_string = fig_combined.to_html(full_html=False)
    components.html(html_string, height=700)

    st.markdown("----")

    # If cashflow enabled, compute and display sequencing results per portfolio
    if cashflow_enabled and start_value is not None and annual_withdrawals is not None:
        st.header("Cashflow / Sequencing Analysis")
        from riskboot.simulate import apply_withdrawals

        res_static = None
        res_trend = None
        res_benchmark = None

        for key, label in [("static", "Static"), ("trend", "Trend"), ("benchmark", "Benchmark")]:
            if key in out:
                returns = out[key]["returns"]  # (S, M)
                res = apply_withdrawals(returns, float(start_value), annual_withdrawals, float(inflation_pct))
                if key == "static":
                    res_static = res
                elif key == "trend":
                    res_trend = res
                elif key == "benchmark":
                    res_benchmark = res
                st.subheader(f"{label} - Cashflow results")
                st.metric("Survival rate", f"{res['survival_rate']*100:.1f}%")
                final_balances = res['wealth'][:, -1]
                st.write(f"Median final balance: £{np.median(final_balances):,.0f}")
                ruin_months = res['ruin_month'][~np.isnan(res['ruin_month'])]
                if len(ruin_months) > 0:
                    st.write(f"Median time to ruin (years): {np.median(ruin_months)/12:.1f}")
                else:
                    st.write("No ruin observed in any simulated path.")

        # Combined wealth fan chart
        bands_static = res_static['bands'] if res_static else None
        bands_trend = res_trend['bands'] if res_trend else None
        bands_benchmark = res_benchmark['bands'] if res_benchmark else None
        fig_wealth = combined_fan_figure(bands_static, bands_trend, bands_benchmark, months=months, is_wealth_index=False, starting_value=start_value)
        html_string_wealth = fig_wealth.to_html(full_html=False)
        components.html(html_string_wealth, height=700)

        # Combined survival curves
        st.markdown("### Survival curves (fraction not ruined over time)")
        fig_surv = go.Figure()
        for key, label in [("static", "Static"), ("trend", "Trend"), ("benchmark", "Benchmark")]:
            if key in out:
                returns = out[key]['returns']
                res = apply_withdrawals(returns, float(start_value), annual_withdrawals, float(inflation_pct))
                surv_by_month = np.mean(res['wealth'] > 0.0, axis=0)
                fig_surv.add_trace(go.Scatter(x=np.arange(len(surv_by_month))/12.0, y=surv_by_month, name=label, mode='lines+markers', line=dict(width=3), marker=dict(size=3)))
        fig_surv.update_layout(xaxis_title='Years', yaxis_title='Fraction surviving', height=350, yaxis=dict(range=[0, 1]))
        # st.plotly_chart(fig_surv, use_container_width=True)
        html_string_surv = fig_surv.to_html(full_html=False)
        components.html(html_string_surv, height=350)
