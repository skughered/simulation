try:
    from riskboot.data import parse_meta_csv, load_trend_weights
    from riskboot.config import DATA_DIR, ALL_ASSETS_FILENAME, TREND_WEIGHTS_FILENAME, DEFAULT_MONTHS, DEFAULT_SCENS, SEED_SIM
    from riskboot.simulate import simulate_portfolios
    import pandas as pd

    # Load data
    data_df, meta_df = parse_meta_csv(DATA_DIR, ALL_ASSETS_FILENAME)
    print('Data loaded successfully.')
    print('Data shape:', data_df.shape)
    print('Public assets:', meta_df[meta_df['public']].index.tolist()[:5], '...')

    # Load weights
    weights_df = load_trend_weights(DATA_DIR, TREND_WEIGHTS_FILENAME)
    print('Weights loaded.')
    print('Weights shape:', weights_df.shape)
    print('Portfolios:', weights_df.columns.tolist())

    # Sample weights for simulation (e.g., equal weights on first 3 public assets)
    public_assets = meta_df[meta_df['public']].index.tolist()
    sample_weights = {asset: 1.0 / len(public_assets) for asset in public_assets}

    # Run simulation
    print('Running simulation...')
    results = simulate_portfolios(
        weights=sample_weights,
        months=DEFAULT_MONTHS,
        n_scenarios=100,  # Smaller for test
        seed=SEED_SIM,
        trend_portfolio='TWP2',
        benchmark_portfolio='BM1'
    )
    print('Simulation completed.')

    # Print some results
    print('Static metrics:', results['static']['metrics'])
    print('Trend metrics:', results['trend']['metrics'])
    if 'benchmark' in results:
        print('Benchmark metrics:', results['benchmark']['metrics'])

except Exception as e:
    print('Error:', str(e))
    import traceback
    traceback.print_exc()
