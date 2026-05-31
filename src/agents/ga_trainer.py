from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np


@dataclass
class GAConfig:
    population_size: int = 8
    generations: int = 5
    elite_fraction: float = 0.25
    mutation_rate: float = 0.3
    train_timesteps: int = 10_000
    eval_episodes: int = 1
    seed: Optional[int] = 42
    workers: int = 1


@dataclass
class GABuildSpec:
    model_name: str
    policy_name: str
    env_cls: type
    env_params: dict
    data: dict
    norm: Optional[str]
    seq_len: int
    verbose: int


_SEARCH_SPACE: dict[str, dict[str, dict[str, Any]]] = {
    "ppo": {
        "learning_rate": {"type": "log_float", "low": 1e-6, "high": 1e-3},
        "n_steps": {"type": "choice", "values": [128, 256, 512, 1024, 2048]},
        "batch_size": {"type": "choice", "values": [32, 64, 128, 256]},
        "gamma": {"type": "float", "low": 0.90, "high": 0.999},
        "gae_lambda": {"type": "float", "low": 0.80, "high": 0.99},
        "ent_coef": {"type": "log_float", "low": 1e-4, "high": 1e-2},
        "clip_range": {"type": "float", "low": 0.10, "high": 0.30},
    },
    "a2c": {
        "learning_rate": {"type": "log_float", "low": 1e-6, "high": 1e-3},
        "n_steps": {"type": "choice", "values": [32, 64, 128, 256]},
        "gamma": {"type": "float", "low": 0.90, "high": 0.999},
        "gae_lambda": {"type": "float", "low": 0.80, "high": 0.99},
        "ent_coef": {"type": "log_float", "low": 1e-4, "high": 1e-2},
        "vf_coef": {"type": "float", "low": 0.1, "high": 1.0},
        "max_grad_norm": {"type": "float", "low": 0.3, "high": 1.0},
    },
    "sac": {
        "learning_rate": {"type": "log_float", "low": 1e-6, "high": 1e-3},
        "batch_size": {"type": "choice", "values": [64, 128, 256, 512]},
        "buffer_size": {"type": "choice", "values": [50_000, 100_000, 200_000]},
        "gamma": {"type": "float", "low": 0.90, "high": 0.999},
        "tau": {"type": "float", "low": 0.005, "high": 0.02},
    },
    "td3": {
        "learning_rate": {"type": "log_float", "low": 1e-6, "high": 1e-3},
        "batch_size": {"type": "choice", "values": [64, 128, 256, 512]},
        "buffer_size": {"type": "choice", "values": [50_000, 100_000, 200_000]},
        "gamma": {"type": "float", "low": 0.90, "high": 0.999},
        "tau": {"type": "float", "low": 0.005, "high": 0.02},
    },
    "ddpg": {
        "learning_rate": {"type": "log_float", "low": 1e-6, "high": 1e-3},
        "batch_size": {"type": "choice", "values": [64, 128, 256, 512]},
        "buffer_size": {"type": "choice", "values": [50_000, 100_000, 200_000]},
        "gamma": {"type": "float", "low": 0.90, "high": 0.999},
        "tau": {"type": "float", "low": 0.005, "high": 0.02},
    },
    "recurrent_ppo": {
        "learning_rate": {"type": "log_float", "low": 1e-6, "high": 1e-3},
        "n_steps": {"type": "choice", "values": [128, 256, 512, 1024]},
        "batch_size": {"type": "choice", "values": [32, 64, 128]},
        "gamma": {"type": "float", "low": 0.90, "high": 0.999},
        "gae_lambda": {"type": "float", "low": 0.80, "high": 0.99},
        "ent_coef": {"type": "log_float", "low": 1e-4, "high": 1e-2},
        "clip_range": {"type": "float", "low": 0.10, "high": 0.30},
        "n_epochs": {"type": "choice", "values": [5, 10, 15]},
    },
}


def _evaluate_model(model, env, episodes: int = 1) -> float:
    scores = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        last_value = None
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            total_reward += float(rewards[0])
            if "portfolio_value" in infos[0]:
                last_value = float(infos[0]["portfolio_value"])

            done = bool(dones[0])

        scores.append(last_value if last_value is not None else total_reward)
    return float(np.mean(scores))


def _sample_param(rng: np.random.Generator, spec: dict[str, Any]) -> Any:
    if spec["type"] == "choice":
        return rng.choice(spec["values"])
    if spec["type"] == "float":
        return float(rng.uniform(spec["low"], spec["high"]))
    if spec["type"] == "log_float":
        low, high = np.log(spec["low"]), np.log(spec["high"])
        return float(np.exp(rng.uniform(low, high)))
    raise ValueError(f"Unknown search space type: {spec['type']}")


def _get_search_space(model_name: str) -> dict[str, dict[str, Any]]:
    if model_name not in _SEARCH_SPACE:
        raise ValueError(f"No GA search space for model '{model_name}'")
    return _SEARCH_SPACE[model_name]


def _sanitize_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    if model_name in {"ppo", "recurrent_ppo"}:
        n_steps = params.get("n_steps")
        batch_size = params.get("batch_size")
        if n_steps is not None and batch_size is not None and batch_size > n_steps:
            params["batch_size"] = int(n_steps)
    return params


def _random_candidate(model_name: str, rng: np.random.Generator) -> dict[str, Any]:
    space = _get_search_space(model_name)
    params = {key: _sample_param(rng, spec) for key, spec in space.items()}
    return _sanitize_params(model_name, params)


def _mutate_candidate(
    model_name: str,
    candidate: dict[str, Any],
    rng: np.random.Generator,
    mutation_rate: float,
) -> dict[str, Any]:
    space = _get_search_space(model_name)
    mutated = dict(candidate)
    for key, spec in space.items():
        if rng.random() < mutation_rate:
            mutated[key] = _sample_param(rng, spec)
    return _sanitize_params(model_name, mutated)


def _crossover(
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    child = {}
    for key in parent_a.keys():
        child[key] = parent_a[key] if rng.random() < 0.5 else parent_b[key]
    return child


def _build_env_and_policy(spec: GABuildSpec):
    from policies.factory import make_policy
    from envs.assembly import make_env

    env = spec.env_cls(**spec.data, **spec.env_params)
    policy, policy_kwargs, requires_sequence = make_policy(
        model_name=spec.model_name,
        policy_name=spec.policy_name,
        observation_space=env.observation_space,
    )
    env = make_env(
        env=env,
        norm=spec.norm,
        seq_len=spec.seq_len,
        requires_sequence=requires_sequence,
    )
    return env, policy, policy_kwargs


def _build_model(spec: GABuildSpec, params: dict[str, Any]):
    from agents.drl_agent import _MODEL_MAP

    env, policy, policy_kwargs = _build_env_and_policy(spec)
    model_cls = _MODEL_MAP[spec.model_name]["cls"]
    model_kwargs = dict(_MODEL_MAP[spec.model_name]["kwargs"])
    model_kwargs.update(params)
    model_kwargs["policy_kwargs"] = policy_kwargs

    model = model_cls(
        policy=policy,
        env=env,
        verbose=spec.verbose,
        tensorboard_log=None,
        **model_kwargs,
    )
    return model, env


def _evaluate_candidate(
    params: dict[str, Any],
    spec: GABuildSpec,
    train_timesteps: int,
    eval_episodes: int,
) -> float:
    model, train_env = _build_model(spec, params)
    model.learn(total_timesteps=train_timesteps)

    eval_env, _, _ = _build_env_and_policy(spec)
    score = _evaluate_model(model, eval_env, episodes=eval_episodes)

    train_env.close()
    eval_env.close()
    return score


def run_ga(
    model,
    env,
    config: GAConfig,
    build_spec: Optional[GABuildSpec] = None,
) -> Tuple[float, Dict[str, Any]]:
    if build_spec is None:
        raise ValueError("build_spec is required for hyperparameter GA")

    rng = np.random.default_rng(config.seed)

    default_params = _random_candidate(build_spec.model_name, rng)
    population = [default_params]
    for _ in range(config.population_size - 1):
        population.append(_random_candidate(build_spec.model_name, rng))

    best_score = -float("inf")
    best_params = default_params

    if config.workers < 1:
        raise ValueError("workers must be >= 1")

    for _ in range(config.generations):
        if config.workers == 1:
            scores = [
                _evaluate_candidate(p, build_spec, config.train_timesteps, config.eval_episodes)
                for p in population
            ]
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=config.workers, mp_context=ctx) as executor:
                futures = [
                    executor.submit(
                        _evaluate_candidate,
                        p,
                        build_spec,
                        config.train_timesteps,
                        config.eval_episodes,
                    )
                    for p in population
                ]
                scores = [future.result() for future in futures]

        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        best_score, best_params = ranked[0]

        elite_count = max(1, int(config.elite_fraction * config.population_size))
        elites = [params for _, params in ranked[:elite_count]]

        next_population = list(elites)
        while len(next_population) < config.population_size:
            parent_a, parent_b = rng.choice(elites, size=2, replace=True)
            child = _crossover(parent_a, parent_b, rng=rng)
            child = _mutate_candidate(
                build_spec.model_name,
                child,
                rng=rng,
                mutation_rate=config.mutation_rate,
            )
            next_population.append(child)

        population = next_population

    return best_score, best_params


def build_model_for_params(
    build_spec: GABuildSpec,
    params: dict[str, Any],
):
    return _build_model(build_spec, params)
