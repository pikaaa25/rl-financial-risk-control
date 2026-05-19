import numpy as np
from portfolio_env import PortfolioEnv, EnvConfig
from q_learning_agent import DQNAgent
from reinforce_agent import REINFORCEAgent
from a2c_agent import A2CAgent
from risk_metrics import compute_episode_metrics


def train(algo='a2c', n_episodes=2000, seed=42):
    np.random.seed(seed)
    cfg = EnvConfig(lambda_vol=2.0, lambda_drawdown=3.0, lambda_cvar=1.5)
    env = PortfolioEnv(cfg)

    if algo == 'dqn':
        agent = DQNAgent(env.state_dim)
    elif algo == 'reinforce':
        agent = REINFORCEAgent(env.state_dim, env.action_dim)
    else:  # a2c (default)
        agent = A2CAgent(env.state_dim, env.action_dim)

    history = []
    best_sharpe = -np.inf

    for ep in range(n_episodes):
        state = env.reset()
        if hasattr(agent, 'net'): agent.net.reset_lstm()

        ep_rewards, ep_returns, ep_weights = [], [], []
        done = False
        step = 0

        while not done:
            # ── Select action ────────────────────────────────────────
            if algo == 'dqn':
                weights, a_idx = agent.select_action(state)
            elif algo == 'reinforce':
                weights, alpha = agent.select_action(state)
            else:  # a2c
                weights, alpha, value = agent.select_action(state)

            # ── Environment step ─────────────────────────────────────
            next_state, reward, done, info = env.step(weights)

            # ── Store experience ─────────────────────────────────────
            if algo == 'dqn':
                agent.store(state, a_idx, reward, next_state, done)
                agent.learn()
            elif algo == 'reinforce':
                agent.store(state, weights, reward, alpha)
            else:  # a2c: n-step learning
                agent.store(state, weights, reward, alpha, value)
                if (step + 1) % agent.n_steps == 0:
                    agent.learn(next_state=next_state, done=done)

            ep_rewards.append(reward)
            ep_returns.append(info['portfolio_return'])
            ep_weights.append(info['weights'])
            state = next_state
            step += 1

        # ── End of episode updates ────────────────────────────────
        if algo == 'reinforce':
            agent.learn()
        elif algo == 'a2c' and agent.buffer:
            agent.learn(done=True)

        # ── Compute episode metrics ───────────────────────────────
        metrics = compute_episode_metrics(ep_returns, ep_weights, cfg)
        metrics['total_reward'] = np.sum(ep_rewards)
        metrics['episode'] = ep
        history.append(metrics)

        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']

        if (ep + 1) % 100 == 0:
            recent = history[-100:]
            avg_sharpe = np.mean([m['sharpe'] for m in recent])
            avg_dd = np.mean([m['max_drawdown'] for m in recent])
            print(f"Ep {ep+1:4d} | Sharpe: {avg_sharpe:.3f} | "
                  f"MaxDD: {avg_dd:.3f} | Best Sharpe: {best_sharpe:.3f}")

    return history


if __name__ == '__main__':
    for algo in ['dqn', 'reinforce', 'a2c']:
        print(f"\n{'='*50}")
        print(f"Training: {algo.upper()}")
        history = train(algo=algo, n_episodes=2000)
        final = history[-100:]
        print(f"Final Sharpe: {np.mean([m['sharpe'] for m in final]):.3f}")
        print(f"Final MaxDD:  {np.mean([m['max_drawdown'] for m in final]):.3f}")