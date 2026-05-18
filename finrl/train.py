from __future__ import annotations
from finrl.agents.models import DRLAgent


def train(
    env,
    model_name,
    policy,
    cwd,
    agent_params,
    total_timesteps=7
):

    agent = DRLAgent(env=env,
                     model_name=model_name,
                     policy=policy,
                     model_kwargs=agent_params,
                     verbose=1,
                     )

    agent.train(total_timesteps=total_timesteps)
    
    print("Training is finished!")
    agent.save(cwd)
    print("Trained model is saved in " + str(cwd))
