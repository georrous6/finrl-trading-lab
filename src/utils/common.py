from __future__ import annotations

import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

from data.preprocessor import FeatureEngineer
from data.downloader import DataDownloader


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # Add common arguments
    parser.add_argument(
        "--asset-type",
        dest="asset_type",
        help="type of asset: crypto or stock",
        metavar="ASSET_TYPE",
        choices=["crypto", "stock"],
        default="stock",
    )
    parser.add_argument(
        "--use-fuzzy",
        dest="use_fuzzy",
        help="whether to use the fuzzy-enhanced environment",
        action="store_true",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        help="model name, a2c, ppo, ddpg, td3, sac, recurrent_ppo",
        metavar="MODEL",
        choices=["a2c", "ppo", "ddpg", "td3", "sac", "recurrent_ppo"],
        default="ppo",
    )
    parser.add_argument(
        "--policy-name",
        dest="policy_name",
        help="policy architecture, MlpPolicy, MlpLstmPolicy, "
        "TransformerPolicy, MlpTransformerPolicy",
        metavar="POLICY_NAME",
        choices=["MlpPolicy", "MlpLstmPolicy", 
                 "TransformerPolicy", "MlpTransformerPolicy"],
        default="TransformerPolicy",
    )
    parser.add_argument(
        "--norm",
        dest="norm",
        default="rolling_window",       # default when not provided
        choices=["rolling_window"],     # don't include None here
        required=False,
        nargs="?",                      # makes it optional with no value
        const=None,                     # --norm with no value -> None
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        help="verbosity level: 0 (no output), 1 (info), 2 (debug)",
        metavar="VERBOSE",
        type=int,
        choices=[0, 1, 2],
        default=1,
    )

    return parser


def make_directories(directories: list[str]):
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_data(
        ticker_list, 
        tech_indicator_list,
        interval,
        start_date, 
        end_date, 
        data_path, 
        use_vix=True,
        use_turbulence=True):

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_indicator_list,
        use_vix=use_vix,
        use_turbulence=use_turbulence,
    )

    if Path(data_path).exists():
        print(f"File {data_path} already exists. Skipping download.")
        df = pd.read_csv(data_path)
        print("\n=== Loaded data ===")
        print(df.head())
    else:
        df_raw = DataDownloader(
            start_date=start_date,
            end_date=end_date,
            ticker_list=ticker_list,
            interval=interval
        ).fetch_data()

        print(f"\n=== Raw data (shape: {df_raw.shape}) ===")
        print(df_raw.head())

        processed = fe.preprocess_data(df_raw)
        print(f"\n=== Processed data (shape: {processed.shape}) ===")
        print(processed.head())

        df = fe.finalize_data(processed)

        print(f"\n=== Finalized data (shape: {df.shape}) ===")
        print(df.head())

        print(f'Length of processed data: {len(df)}')
        df.to_csv(data_path)
        print(f"Data saved to {data_path}")

    price_array, tech_array, vix_array, turbulence_array = fe.build_feature_arrays(df)

    print(f"Price array shape: {price_array.shape}")
    print(f"Technical array shape: {tech_array.shape}")
    if vix_array is not None:
        print(f"VIX array shape: {vix_array.shape}")
    if turbulence_array is not None:
        print(f"Turbulence array shape: {turbulence_array.shape}")


    return {
        'price_array': price_array,
        'tech_array': tech_array,
        'vix_array': vix_array,
        'turbulence_array': turbulence_array,
    }
