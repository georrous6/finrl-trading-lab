from __future__ import annotations

import itertools
import pandas as pd

from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


def download_data(
        train_start_date,
        train_end_date,
        test_start_date,
        test_end_date,
        trade_start_date,
        trade_end_date,
        ticker_list,
        tech_indicator_list,
        train_output_path,
        test_output_path,
        trade_output_path,
        use_vix=True):

    def _download_and_process(start_date, end_date, output_path):
        df_raw = YahooDownloader(
            start_date=start_date,
            end_date=end_date,
            ticker_list=ticker_list,
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

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
            processed, on=["date", "tic"], how="left"
        )
        processed_full = processed_full[processed_full["date"].isin(processed["date"])]
        processed_full = processed_full.sort_values(["date", "tic"], ignore_index=True)
        processed_full = processed_full.fillna(0)
        processed_full.index = processed_full["date"].factorize()[0]

        print("\n=== Processed data ===")
        print(processed_full.head())

        print(f'Length of processed data: {len(processed_full)}')
        processed_full.to_csv(output_path)
        print(f"Data saved to {output_path}")
    
    _download_and_process(train_start_date, train_end_date, train_output_path)
    _download_and_process(test_start_date, test_end_date, test_output_path)
    _download_and_process(trade_start_date, trade_end_date, trade_output_path)
