sbatch --partition=rome --job-name=ppo-mlp train.sh \
--total-timesteps=1000000 \
--model-name=ppo \
--policy-name=MlpPolicy

sbatch --partition=rome --job-name=ppo-transformer train.sh \
--total-timesteps=1000000 \
--model-name=ppo \
--policy-name=TransformerPolicy

sbatch --partition=rome --job-name=ppo-mlp-transformer train.sh \
--total-timesteps=1000000 \
--model-name=ppo \
--policy-name=MlpTransformerPolicy

sbatch --partition=rome --job-name=ddpg-mlp train.sh \
--total-timesteps=1000000 \
--model-name=ddpg \
--policy-name=MlpPolicy