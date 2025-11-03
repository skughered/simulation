import streamlit as st

st.set_page_config(page_title="User Guide", layout="wide")
st.title("📖 User Guide")

st.markdown("""
This guide explains the portfolio risk simulator app.
""")

# FAQ section for easy addition of questions/answers
faq = {
    "Why Bootstrap Simulation?": """
    This app uses a **joint stationary bootstrap** method to generate portfolio risk scenarios, rather than traditional 
    Monte Carlo simulations (e.g., assuming normal distributions for returns). Here's why:

    - **Bootstrap preserves real-world data patterns**: Instead of inventing returns from assumed models (which can 
    underestimate extreme events like crashes), bootstrap resamples actual historical monthly returns. This keeps the 
    "fat tails" (rare big losses) and asymmetries seen in markets, making projections more realistic for
    clients' portfolios.

    - **Synchronous blocks, not single months**: We resample in blocks of 6-12 months (averaging 9 months) across all 
    assets at once. This maintains:
      - **Cross-asset correlations**: Stocks, bonds, and cash move together as they did historically (e.g., 
      equity-bond linkages during crises).
      - **Volatility clustering**: Periods of high/low volatility persist, reflecting real market regimes.
      - **Return clustering**: Good/bad months tend to bunch, avoiding unrealistic randomness.

    This approach avoids Monte Carlo's pitfalls, like assuming independent, identically distributed returns, which can 
    lead to overly optimistic risk estimates.
    """,
    # Add more Q&A here as needed, e.g.:
    # "How to interpret the fan chart?": "Explanation here...",
}

for question, answer in faq.items():
    with st.expander(question):
        st.markdown(answer)

st.markdown("---")
# st.caption("For more details, refer to the app's inputs and outputs.")
