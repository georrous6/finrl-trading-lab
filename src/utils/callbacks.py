from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


_INTERVALS_MAP = {
    "1d": 252,
    "1h": 252 * 6.5,
    "5m": 252 * 6.5 * 12,
    "1m": 252 * 6.5 * 60,
}


class FinancialMetricsCallback(BaseCallback):
    """
    Tracks per-episode financial metrics and logs them to TensorBoard.

    Metrics logged at episode end:
        `financial/sharpe_ratio`: annualized Sharpe over episode
        `financial/total_return`: episode total return
        `financial/max_drawdown`: episode max drawdown
        `financial/final_value`: final portfolio value

    Metrics logged at rollout end:
        `financial/mean_episode_return`: mean total return across episodes
    """

    def __init__(self, interval: str = "1d", verbose: int = 0):
        super().__init__(verbose)

        interval = interval.lower()
        if interval not in _INTERVALS_MAP:
            raise ValueError(f"Unknown interval '{interval}'. "
                             f"Choose from: {list(_INTERVALS_MAP)}")
        self.steps_per_year = _INTERVALS_MAP[interval]

        # Per-episode tracking (one deque per env)
        self._episode_values: list[list[float]] = []

        # Across episodes within rollout
        self._completed_episode_returns: list[float] = []

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self._episode_values = [[] for _ in range(n_envs)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, (info, done) in enumerate(zip(infos, dones)):
            if "portfolio_value" in info:
                self._episode_values[i].append(info["portfolio_value"])

            if done and len(self._episode_values[i]) >= 2:
                self._log_episode(i)
                self._episode_values[i] = []

        return True

    def _log_episode(self, env_idx: int):
        values = np.array(self._episode_values[env_idx], dtype=np.float64)
        step_returns = values[1:] / (values[:-1] + 1e-8) - 1.0

        sharpe = self._sharpe(step_returns)
        total_return = values[-1] / (values[0] + 1e-8) - 1.0
        max_dd = self._max_drawdown(values)

        self.logger.record("financial/sharpe_ratio", sharpe)
        self.logger.record("financial/total_return", total_return)
        self.logger.record("financial/max_drawdown", max_dd)
        self.logger.record("financial/final_value", values[-1])

        self._completed_episode_returns.append(total_return)

        if self.verbose > 0:
            print(
                f"[Ep {env_idx}] "
                f"Return={total_return:.2%} "
                f"Sharpe={sharpe:.2f} "
                f"MaxDD={max_dd:.2%}"
            )

    def _on_rollout_end(self):
        if self._completed_episode_returns:
            mean_return = np.mean(self._completed_episode_returns)
            self.logger.record("financial/mean_episode_return", mean_return)
            self._completed_episode_returns = []

    def _sharpe(self, step_returns: np.ndarray) -> float:
        if len(step_returns) < 2:
            return 0.0
        mean = np.mean(step_returns)
        std = np.std(step_returns) + 1e-8
        return float(mean / std * np.sqrt(self.steps_per_year))

    def _max_drawdown(self, values: np.ndarray) -> float:
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / (peak + 1e-8)
        return float(drawdown.min())