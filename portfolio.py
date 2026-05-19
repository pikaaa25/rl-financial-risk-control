import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# ── Asset Configuration ─────────────────────────────────────────────────────
ASSETS = ['Equity', 'Bond', 'Commodity', 'REIT', 'Cash']
N_ASSETS = 5
LOOKBACK = 20   # rolling window for vol/momentum
MAX_STEPS = 252 # 1 trading year

# ── Asset Parameters (annualized, then scaled to daily) ──────────────────────
ASSET_MU    = np.array([0.10, 0.04, 0.06, 0.08, 0.02]) / 252
ASSET_SIGMA = np.array([0.18, 0.06, 0.22, 0.14, 0.002]) / np.sqrt(252)
CORR_MATRIX = np.array([
    [1.0,  -0.3,  0.2,  0.5,  0.0],
    [-0.3,  1.0, -0.1, -0.2,  0.1],
    [0.2,  -0.1,  1.0,  0.3,  0.0],
    [0.5,  -0.2,  0.3,  1.0,  0.0],
    [0.0,   0.1,  0.0,  0.0,  1.0],
])
COV_MATRIX  = np.outer(ASSET_SIGMA, ASSET_SIGMA) * CORR_MATRIX


@dataclass
class EnvConfig:
    n_assets: int = N_ASSETS
    lookback: int = LOOKBACK
    max_steps: int = MAX_STEPS
    transaction_cost: float = 0.001   # 0.1% per trade
    lambda_vol: float = 2.0           # penalty weight for volatility
    lambda_drawdown: float = 3.0       # penalty weight for max drawdown
    lambda_cvar: float = 1.5          # penalty weight for CVaR
    entropy_bonus: float = 0.05       # diversification incentive
    cvar_alpha: float = 0.05          # tail percentile (5%)

    # HMM Regime parameters
    regime_transition: np.ndarray = None  # set in __post_init__
    regime_vol_mult: Tuple = (1.0, 2.5)  # (normal, crisis) vol multiplier
    regime_mu_mult:  Tuple = (1.0, -1.5) # (normal, crisis) return mult

    def __post_init__(self):
        if self.regime_transition is None:
            # HMM transition matrix: [[stay normal, go crisis],[exit crisis, stay crisis]]
            self.regime_transition = np.array([[0.97, 0.03],
                                               [0.15, 0.85]])


class PortfolioEnv:
    """
    Multi-asset portfolio environment with:
      - GARCH(1,1)-like volatility clustering
      - HMM regime switching (normal / crisis)
      - Risk-sensitive composite reward
      - Transaction cost penalties
    State: [returns(5), vol(5), momentum(5), drawdown, var_5, port_vol, regime, step_norm]
    Action: weight vector in simplex ∈ ℝ⁵ (softmax)
    """

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self.state_dim = cfg.n_assets * 3 + 5  # 20-dim state
        self.action_dim = cfg.n_assets
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state, return first observation."""
        self.step_count = 0
        self.regime = 0          # 0=normal, 1=crisis
        self.weights = np.ones(self.cfg.n_assets) / self.cfg.n_assets
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.return_history = []
        self.garch_vol = ASSET_SIGMA.copy()

        # Warm up price history for lookback calculations
        self.price_history = np.zeros((self.cfg.lookback, self.cfg.n_assets))
        for i in range(self.cfg.lookback):
            returns = self._sample_returns()
            self.price_history[i] = returns

        return self._get_state()

    def _update_regime(self):
        """HMM regime transition."""
        p = self.cfg.regime_transition[self.regime]
        self.regime = np.random.choice(2, p=p)

    def _update_garch_vol(self, realized_returns: np.ndarray):
        """GARCH(1,1): σ²(t) = ω + α·r²(t-1) + β·σ²(t-1)"""
        omega = ASSET_SIGMA ** 2 * (1 - 0.1 - 0.85)
        alpha, beta = 0.10, 0.85
        self.garch_vol = np.sqrt(
            np.clip(omega + alpha * realized_returns**2 + beta * self.garch_vol**2,
                    1e-8, None)
        )

    def _sample_returns(self) -> np.ndarray:
        """Sample correlated asset returns with regime-adjusted parameters."""
        vol_mult = self.cfg.regime_vol_mult[self.regime]
        mu_mult  = self.cfg.regime_mu_mult[self.regime]

        # Cholesky decomposition for correlated sampling
        cov = np.outer(self.garch_vol * vol_mult, self.garch_vol * vol_mult) * CORR_MATRIX
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(self.cfg.n_assets))
        z = np.random.randn(self.cfg.n_assets)

        return ASSET_MU * mu_mult + L @ z

    def _get_state(self) -> np.ndarray:
        """Construct 20-dim state from price history and portfolio metrics."""
        hist = self.price_history
        # 1. Recent log returns (last day)
        returns = hist[-1]
        # 2. Rolling volatility (20-day realized)
        vol = np.std(hist, axis=0) * np.sqrt(252)
        # 3. Momentum z-score (mean / std of 20d returns)
        momentum = np.mean(hist, axis=0) / (np.std(hist, axis=0) + 1e-8)

        # 4. Risk metrics
        portfolio_returns = hist @ self.weights
        drawdown = self._compute_drawdown(portfolio_returns)
        var_5 = self._compute_var(portfolio_returns, self.cfg.cvar_alpha)
        port_vol = np.std(portfolio_returns) * np.sqrt(252)
        step_norm = self.step_count / self.cfg.max_steps

        state = np.concatenate([
            returns,           # 5 dims
            np.clip(vol, 0, 1),    # 5 dims
            np.clip(momentum, -3, 3), # 5 dims
            [drawdown, var_5, port_vol, self.regime, step_norm]  # 5 dims
        ])
        return state.astype(np.float32)

    def _compute_drawdown(self, returns: np.ndarray) -> float:
        """Maximum drawdown over lookback window."""
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / (peak + 1e-8)
        return float(np.max(dd))

    def _compute_var(self, returns: np.ndarray, alpha: float) -> float:
        """CVaR (Expected Shortfall) at alpha% tail."""
        if len(returns) < 5:
            return 0.0
        var = np.percentile(returns, alpha * 100)
        cvar = np.mean(returns[returns <= var])
        return float(-cvar)  # positive = bad

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one trading step.
        Args:
            action: raw logits or weight vector (will be softmaxed)
        Returns:
            (next_state, reward, done, info)
        """
        # ── Action processing ─────────────────────────────────────────
        new_weights = self._softmax(action)
        turnover = np.sum(np.abs(new_weights - self.weights))
        tx_cost = self.cfg.transaction_cost * turnover
        self.weights = new_weights

        # ── Market simulation ─────────────────────────────────────────
        self._update_regime()
        returns = self._sample_returns()
        self._update_garch_vol(returns)
        self.price_history = np.roll(self.price_history, -1, axis=0)
        self.price_history[-1] = returns

        # ── Portfolio return ──────────────────────────────────────────
        portfolio_return = np.dot(self.weights, returns) - tx_cost
        self.portfolio_value *= (1 + portfolio_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.return_history.append(portfolio_return)

        # ── Risk-Sensitive Reward ─────────────────────────────────────
        reward = self._compute_reward(portfolio_return, tx_cost)

        self.step_count += 1
        done = self.step_count >= self.cfg.max_steps
        next_state = self._get_state()

        info = {
            'portfolio_return': portfolio_return,
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'regime': self.regime,
            'turnover': turnover,
        }
        return next_state, reward, done, info

    def _compute_reward(self, portfolio_return: float, tx_cost: float) -> float:
        """
        Risk-sensitive composite reward:
          R = log_return
            − λ₁ · rolling_volatility
            − λ₂ · current_drawdown
            − λ₃ · CVaR_penalty
            + entropy_bonus (diversification)
        """
        log_return = np.log1p(portfolio_return)

        if len(self.return_history) >= 10:
            recent = np.array(self.return_history[-20:])
            rolling_vol = np.std(recent) * np.sqrt(252)
            drawdown = (1 - self.portfolio_value / self.peak_value)
            cvar = self._compute_var(recent, self.cfg.cvar_alpha)
        else:
            rolling_vol = drawdown = cvar = 0.0

        # Diversification entropy: H = -Σ wᵢ·log(wᵢ)
        entropy = -np.sum(self.weights * np.log(self.weights + 1e-8))

        reward = (
            log_return
            - self.cfg.lambda_vol      * rolling_vol
            - self.cfg.lambda_drawdown * drawdown
            - self.cfg.lambda_cvar     * cvar
            + self.cfg.entropy_bonus   * entropy
        )
        return float(reward)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(np.clip(x, -20, 20))
        return e / (np.sum(e) + 1e-8)