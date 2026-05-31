from __future__ import annotations

import gymnasium as gym
import numpy as np
import skfuzzy as fuzz
from gymnasium import spaces
from skfuzzy import control as ctrl
from typing import Optional


class StockTradingFuzzyEnv(gym.Env):
    """
    Fuzzy-Enhanced Stock Trading Environment for OpenAI Gymnasium.
    Integrates Fuzzy Action Smoothing and Fuzzy Reward Shaping,
    discarding all hard threshold cutoffs.
    """

    def __init__(
        self,
        price_array: np.ndarray,
        tech_array: np.ndarray,
        vix_array: Optional[np.ndarray] = None,
        turbulence_array: Optional[np.ndarray] = None,
        max_stocks: np.ndarray | int = 1000,
        initial_capital: float = 1e6,
        transaction_cost: float = 1e-3,
        volatility_penalty: float = 1.0,  # Acts as a scaling factor for fuzzy rewards
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()

        # Arrays and Shapes Validation
        assert len(price_array.shape) == 2, "price_array must be (T, N)"
        assert len(tech_array.shape) == 3, "tech_array must be (T, N, F)"
        assert price_array.shape[0] == tech_array.shape[0], "Time dimension mismatch"
        assert price_array.shape[1] == tech_array.shape[1], "Asset dimension mismatch"

        self.price = price_array
        self.tech = tech_array
        self.eps = eps

        self.T, self.N = price_array.shape
        _, _, self.F = tech_array.shape

        # Handle External Volatility and Noise Trackers
        self.vix = (
            vix_array if vix_array is not None else np.zeros(self.T, dtype=np.float32)
        )
        self.turbulence = (
            turbulence_array
            if turbulence_array is not None
            else np.zeros(self.T, dtype=np.float32)
        )

        if isinstance(max_stocks, int):
            self.max_stocks = np.full(self.N, max_stocks, dtype=np.int32)
        else:
            assert len(max_stocks) == self.N, "max_stocks must have length N"
            self.max_stocks = np.array(max_stocks, dtype=np.int32)

        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.vol_penalty_scale = volatility_penalty

        self.max_step = self.T - 1

        # Portfolio State Spaces
        self.cash = initial_capital
        self.shares = np.zeros(self.N, dtype=np.int32)
        self.time = 0

        # Action and Observation Spaces Configuration
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32
        )

        portfolio_dim = 1 + self.N
        market_dim = self.N * (1 + self.F) + 2
        self.observation_space = spaces.Dict(
            {
                "portfolio": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(portfolio_dim,), dtype=np.float32
                ),
                "market": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(market_dim,), dtype=np.float32
                ),
            }
        )

        # Initialize the Internal Fuzzy Inference Engines
        self._build_fuzzy_systems()

    def _build_fuzzy_systems(self):
        """Builds both the action smoother and reward shaper fuzzy rules."""
        # ----------------------------------------------------
        # SYSTEM 1: Action Smoother (Turbulence Mitigation)
        # ----------------------------------------------------
        raw_act = ctrl.Antecedent(np.linspace(-1.0, 1.0, 21), "raw_act")
        turb = ctrl.Antecedent(
            np.linspace(0.0, float(np.max(self.turbulence) + 1.0), 21), "turb"
        )
        safe_act = ctrl.Consequent(np.linspace(-1.0, 1.0, 21), "safe_act")

        # Membership Functions
        raw_act["sell"] = fuzz.trimf(raw_act.universe, [-1.0, -1.0, 0.0])
        raw_act["hold"] = fuzz.trimf(raw_act.universe, [-0.2, 0.0, 0.2])
        raw_act["buy"] = fuzz.trimf(raw_act.universe, [0.0, 1.0, 1.0])

        turb["low"] = fuzz.trimf(
            turb.universe, [0.0, 0.0, np.max(self.turbulence) * 0.3]
        )
        turb["medium"] = fuzz.trimf(
            turb.universe,
            [
                np.max(self.turbulence) * 0.1,
                np.max(self.turbulence) * 0.5,
                np.max(self.turbulence) * 0.8,
            ],
        )
        turb["high"] = fuzz.smf(
            turb.universe, np.max(self.turbulence) * 0.5, np.max(self.turbulence)
        )

        safe_act["sell"] = fuzz.trimf(safe_act.universe, [-1.0, -1.0, 0.0])
        safe_act["hold"] = fuzz.trimf(safe_act.universe, [-0.2, 0.0, 0.2])
        safe_act["buy"] = fuzz.trimf(safe_act.universe, [0.0, 1.0, 1.0])

        # Rules
        act_r1 = ctrl.Rule(turb["high"], safe_act["sell"])  # Severe crisis forcing defensive positions
        act_r2 = ctrl.Rule(turb["medium"] & raw_act["buy"], safe_act["hold"])  # Dampen buying in unstable regimes
        act_r3 = ctrl.Rule(turb["medium"] & raw_act["sell"], safe_act["sell"])
        act_r4 = ctrl.Rule(turb["low"], safe_act["buy"] if raw_act["buy"] else safe_act["sell"] if raw_act["sell"] else safe_act["hold"])

        self.action_sim = ctrl.ControlSystemSimulation(
            ctrl.ControlSystem([act_r1, act_r2, act_r3, act_r4])
        )

        # ----------------------------------------------------
        # SYSTEM 2: Reward Shaper (Risk-Adjusted Deductions)
        # ----------------------------------------------------
        exposure = ctrl.Antecedent(np.linspace(0.0, 1.0, 11), "exposure")
        vix_max = float(np.max(self.vix) + 1.0)
        if vix_max < 35.0:
            vix_max = 35.0
        vix = ctrl.Antecedent(np.linspace(0.0, vix_max, 21), "vix")
        penalty = ctrl.Consequent(np.linspace(0.0, 1.0, 21), "penalty")

        exposure["low"] = fuzz.trimf(exposure.universe, [0.0, 0.0, 0.4])
        exposure["high"] = fuzz.trimf(exposure.universe, [0.3, 1.0, 1.0])

        vix["calm"] = fuzz.trimf(vix.universe, [0.0, 0.0, 20.0])
        vix["stressed"] = fuzz.trimf(vix.universe, [15.0, 30.0, 45.0])
        vix["panic"] = fuzz.smf(vix.universe, 35.0, vix_max)

        penalty["none"] = fuzz.trimf(penalty.universe, [0.0, 0.0, 0.2])
        penalty["medium"] = fuzz.trimf(penalty.universe, [0.1, 0.5, 0.9])
        penalty["severe"] = fuzz.trimf(penalty.universe, [0.7, 1.0, 1.0])

        rew_r1 = ctrl.Rule(vix["panic"] & exposure["high"], penalty["severe"])
        rew_r2 = ctrl.Rule(vix["stressed"] & exposure["high"], penalty["medium"])
        rew_r3 = ctrl.Rule(vix["calm"] | exposure["low"], penalty["none"])

        self.reward_sim = ctrl.ControlSystemSimulation(
            ctrl.ControlSystem([rew_r1, rew_r2, rew_r3])
        )

    def _apply_fuzzy_action_smoothing(self, raw_actions: np.ndarray) -> np.ndarray:
        """Processes raw agent arrays step-by-step through the fuzzifier engine."""
        smoothed_actions = []
        current_turb = self.turbulence[self.time]

        for act in raw_actions:
            try:
                self.action_sim.input["raw_act"] = act
                self.action_sim.input["turb"] = current_turb
                self.action_sim.compute()
                smoothed_actions.append(self.action_sim.output["safe_act"])
            except ValueError:
                # Fallback to clip if calculations fall outside edge membership boundaries
                smoothed_actions.append(act)

        return np.array(smoothed_actions, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # 1. Map Raw Input Matrix through the Fuzzy Framework Core
        action = self._apply_fuzzy_action_smoothing(action)

        price = self.price[self.time]
        current_asset = self._portfolio_value(price)

        valid_mask = price > 0
        sell_mask = (action < 0) & valid_mask
        buy_mask = (action > 0) & valid_mask
        sell_idx = np.where(sell_mask)[0]
        buy_idx = np.where(buy_mask)[0]

        # Execute Transaction Operations
        for i in sell_idx:
            self._sell_asset(i, action)

        for i in buy_idx:
            self._buy_asset(i, action)

        # Shift Forward Chronologically
        self.time += 1
        next_price = self.price[self.time]
        next_asset = self._portfolio_value(next_price)

        # 2. Extract Reward Structuring using Fuzzy Logics
        reward = self._get_fuzzy_reward(current_asset, next_asset)
        terminated = self.time == self.max_step

        info = {
            "portfolio_value": next_asset,
            "cash": self.cash,
            "shares": self.shares.copy(),
        }

        return self._get_obs(), reward, terminated, False, info

    def _get_fuzzy_reward(self, prev_asset: float, next_asset: float) -> float:
        """Applies dynamic penalty scoring based on exposure and stress."""
        log_return = np.log((next_asset + self.eps) / (prev_asset + self.eps))

        total_equity = np.sum(self.shares * self.price[self.time])
        exposure_ratio = total_equity / (prev_asset + self.eps)
        exposure_ratio = np.clip(exposure_ratio, 0.0, 1.0)

        try:
            self.reward_sim.input["exposure"] = exposure_ratio
            self.reward_sim.input["vix"] = self.vix[self.time]
            self.reward_sim.compute()
            fuzzy_penalty = self.reward_sim.output["penalty"]
        except ValueError:
            fuzzy_penalty = 0.0

        # Scale output metrics cleanly according to hyperparameter configs
        return log_return - (fuzzy_penalty * self.vol_penalty_scale)

    def _sell_asset(self, index, action):
        price = self.price[self.time]
        sell_shares = min(
            self.shares[index], int(-action[index] * self.shares[index])
        )
        self.shares[index] -= sell_shares
        self.cash += sell_shares * price[index] * (1 - self.tc)

    def _buy_asset(self, index, action):
        price = self.price[self.time]
        total_asset = self._portfolio_value(price)

        desired_shares = int(action[index] * total_asset / price[index])
        max_shares = self.max_stocks[index] - self.shares[index]
        buy_shares = min(
            desired_shares,
            self.cash // (price[index] * (1 + self.tc)),
            max_shares,
        )

        self.shares[index] += buy_shares
        self.cash -= buy_shares * price[index] * (1 + self.tc)

    def _get_obs(self):
        price = self.price[self.time]
        tech = self.tech[self.time]

        portfolio = np.concatenate([[self.cash], self.shares.copy()]).astype(
            np.float32
        )
        market = np.concatenate(
            [
                price.copy(),
                tech.flatten().copy(),
                [self.vix[self.time]],
                [self.turbulence[self.time]],
            ]
        ).astype(np.float32)

        return {"portfolio": portfolio, "market": market}

    def _portfolio_value(self, price):
        return self.cash + np.sum(self.shares * price)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_capital
        self.shares = np.zeros(self.N, dtype=np.int32)
        self.time = 0
        return self._get_obs(), {}

    def close(self):
        pass