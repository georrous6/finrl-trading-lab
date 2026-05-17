from __future__ import annotations
from finrl.agents.stablebaselines3.models import DRLAgent


def train(
    env,
    model_name,
    cwd,
    agent_params,
    total_timesteps=1e6
):

    agent = DRLAgent(env=env)
    model = agent.get_model(model_name, model_kwargs=agent_params)
    trained_model = agent.train_model(
        model=model, tb_log_name=model_name, total_timesteps=total_timesteps
    )
    print("Training is finished!")
    trained_model.save(cwd)
    print("Trained model is saved in " + str(cwd))
