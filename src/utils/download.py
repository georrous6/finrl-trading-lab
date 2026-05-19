from __future__ import annotations

import itertools
import pandas as pd
import numpy as np
from pathlib import Path

from src.meta.preprocessor.preprocessors import FeatureEngineer
from src.meta.preprocessor.yahoodownloader import YahooDownloader


def download_data(
        ticker_list, 
        tech_indicator_list,
        interval,
        start_date, 
        end_date, 
        data_path, 
        use_vix=True):

    if Path(data_path).exists():
        print(f"File {data_path} already exists. Skipping download.")
        df = pd.read_csv(data_path)
        print("\n=== Loaded data ===")
        print(df.head())
    else:
        df_raw = YahooDownloader(
            start_date=start_date,
            end_date=end_date,
            ticker_list=ticker_list,
            interval=interval
        ).fetch_data()

        print("\n=== Raw data ===")
        print(df_raw.head())

        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=tech_indicator_list,
            use_vix=use_vix,
            use_turbulence=False,
            user_defined_feature=False,
        )

        processed = fe.preprocess_data(df_raw)

        list_ticker = processed["tic"].unique().tolist()
        list_date = list(
            pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
        )
        combination = list(itertools.product(list_date, list_ticker))

        df = pd.DataFrame(combination, columns=["date", "tic"]).merge(
            processed, on=["date", "tic"], how="left"
        )
        df = df[df["date"].isin(processed["date"])]
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df = df.fillna(0)
        df.index = df["date"].factorize()[0]

        print("\n=== Processed data ===")
        print(df.head())

        print(f'Length of processed data: {len(df)}')
        df.to_csv(data_path)
        print(f"Data saved to {data_path}")

    # Prepare arrays for environment

    # stable ordering
    df = df.sort_values(["date", "tic"]).copy()

    dates = sorted(df["date"].unique())
    tickers = sorted(df["tic"].unique())

    df = df.set_index(["date", "tic"]).sort_index()

    # --- PRICE ---
    price_array = (
        df["close"]
        .unstack("tic")
        .reindex(index=dates, columns=tickers)
        .fillna(0)
        .to_numpy()
    )

    # --- TECH FEATURES ---
    tech_cols = [c for c in tech_indicator_list]
    if use_vix:
        tech_cols = tech_cols + ["vix"]

    tech_list = []
    for c in tech_cols:
        mat = (
            df[c]
            .unstack("tic")
            .reindex(index=dates, columns=tickers)
            .fillna(0)
            .to_numpy()
        )
        tech_list.append(mat)

    tech_array = np.stack(tech_list, axis=-1)  # (T, N, F)

    print(f"Price array shape: {price_array.shape}")
    print(f"Technical array shape: {tech_array.shape}")

    return price_array, tech_array