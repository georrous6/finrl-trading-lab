TRANSFORMER_POLICY_PARAMS = {
    "features_extractor_kwargs": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "net_arch": dict(pi=[64, 64], vf=[64, 64]),
}