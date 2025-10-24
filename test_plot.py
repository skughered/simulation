import numpy as np
import plotly.graph_objects as go
from riskboot.simulate import simulate_portfolios

weights = {"stocks": 0.6, "bonds": 0.3, "rf_1m": 0.1}
out = simulate_portfolios(weights=weights, months=60, n_scenarios=1000)

bands = out["static"]["bands"]

x = np.arange(len(bands["q50"]))
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=bands["q50"], line=dict(color="blue"), name="median"))
fig.add_trace(go.Scatter(x=x, y=bands["q05"], line=dict(color="lightblue"), name="q05"))
fig.add_trace(go.Scatter(x=x, y=bands["q95"], line=dict(color="lightblue"), name="q95"))
fig.update_layout(title="Fan sanity check", template="plotly_white")
fig.show()
import plotly.io as pio
pio.write_html(fig, file="fan_check.html", auto_open=True)