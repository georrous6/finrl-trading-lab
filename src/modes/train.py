from __future__ import annotations
from src.agents.reccurent_agent import ReccurentDRLAgent
from src.utils.download import download_data

def train(
    start_date,
    end_date,
    indicator_list,
    ticker_list,
    data_path,
    env,
    norm,
    algo,
    policy,
    env_kwargs,
    norm_kwargs,
    policy_kwargs,
    algo_kwargs,
    tensorboard_log,
    seq_len=10,
    verbose=1,
    total_timesteps=10_000,
    save_path=None
):

    price_array, tech_array = download_data(
        ticker_list=ticker_list,
        tech_indicator_list=indicator_list,
        start_date=start_date,
        end_date=end_date,
        data_path=data_path,
        use_vix=True,
    )

    env = env(
        price_array=price_array,
        tech_array=tech_array,
        **env_kwargs,
    )

    agent = ReccurentDRLAgent(
        algo=algo,
        policy=policy,
        env=env,
        seq_len=seq_len,
        norm=norm,
        norm_kwargs=norm_kwargs,
        policy_kwargs=policy_kwargs,
        algo_kwargs=algo_kwargs,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )

    agent.train(total_timesteps=total_timesteps)
    
    print("Training is finished!")

    if save_path is not None:
        agent.save(save_path)
        print(f"Trained model is saved in {save_path}")