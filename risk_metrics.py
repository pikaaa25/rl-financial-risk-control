import numpy as np


def sharpe_ratio(returns: np.ndarray, rf: float = 0.02 / 252) -> float:
    """
    Annualized Sharpe Ratio: (E[r] − rf) / σ · √252
    Measures risk-adjusted return per unit of volatility.
    """
    excess = returns - rf
    if np.std(excess) < 1e-8:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252))


def sortino_ratio(returns: np.ndarray, rf: float = 0.02 / 252) -> float:
    """
    Sortino Ratio: (E[r] − rf) / σ_downside · √252
    Like Sharpe but only penalizes downside volatility.
    """
    excess = returns - rf
    downside = returns[returns < rf]
    if len(downside) < 2:
        return 0.0
    sigma_d = np.std(downside)
    if sigma_d < 1e-8:
        return 0.0
    return float(np.mean(excess) / sigma_d * np.sqrt(252))


def max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum Drawdown: max peak-to-trough decline in portfolio value.
    MDD = max_t { (peak_t − value_t) / peak_t }
    """
    cum_returns = np.cumprod(1 + np.array(returns))
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / (peak + 1e-8)
    return float(np.max(drawdown))


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    VaR at α%: the α-percentile of the return distribution.
    P(r ≤ VaR) = α  →  VaR = quantile(returns, α)
    """
    return float(np.percentile(returns, alpha * 100))


def conditional_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    CVaR / Expected Shortfall at α%:
    CVaR = E[r | r ≤ VaR_α] — average loss in worst α% of scenarios.
    More coherent risk measure than VaR (sub-additive).
    """
    var = value_at_risk(returns, alpha)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(np.mean(tail))


def calmar_ratio(returns: np.ndarray) -> float:
    """Calmar Ratio: Annual Return / Max Drawdown. Popular in hedge funds."""
    annual = np.mean(returns) * 252
    mdd = max_drawdown(returns)
    return annual / (mdd + 1e-8)


def portfolio_volatility(returns: np.ndarray) -> float:
    """Annualized volatility of portfolio returns."""
    return float(np.std(returns) * np.sqrt(252))


def herfindahl_index(weights: np.ndarray) -> float:
    """
    HHI concentration index: Σ wᵢ².
    HHI = 1 → fully concentrated, HHI = 1/n → perfectly diversified.
    """
    return float(np.sum(weights ** 2))


def compute_episode_metrics(returns, weight_history, cfg) -> dict:
    """Compute all metrics for one episode."""
    ret = np.array(returns)
    avg_weights = np.mean(weight_history, axis=0)
    return {
        'sharpe':       sharpe_ratio(ret),
        'sortino':      sortino_ratio(ret),
        'max_drawdown': max_drawdown(ret),
        'cvar_5':       -conditional_var(ret, 0.05),
        'calmar':       calmar_ratio(ret),
        'annual_vol':   portfolio_volatility(ret),
        'annual_ret':   np.mean(ret) * 252,
        'hhi':          herfindahl_index(avg_weights),
        'total_return': float(np.prod(1 + ret) - 1),
    }