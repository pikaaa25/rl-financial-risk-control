import numpy as np


class PolicyNetwork:
    """
    Stochastic policy π_θ(a|s) outputting Dirichlet concentration params.
    Architecture: Linear(state_dim → 64) → Tanh → Linear(64 → n_assets)
                  → Softplus (ensure positive) → Dirichlet(α)
    """

    def __init__(self, state_dim, n_assets, lr=3e-4):
        self.lr = lr
        self.n_assets = n_assets
        scale = np.sqrt(2 / state_dim)
        self.W1 = np.random.randn(state_dim, 64) * scale
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, n_assets) * np.sqrt(2/64)
        self.b2 = np.ones(n_assets)   # init bias to 1 → uniform Dirichlet

    def forward(self, s):
        """Returns Dirichlet concentration params α > 0."""
        h = np.tanh(s @ self.W1 + self.b1)
        raw = h @ self.W2 + self.b2
        # Softplus: log(1 + e^x) ensures α > 0
        alpha = np.log1p(np.exp(np.clip(raw, -20, 20))) + 0.1
        return alpha, h

    def sample_action(self, s):
        """
        Sample portfolio weights from Dirichlet distribution.
        Dirichlet(α) is the conjugate prior for categorical distributions;
        ensures weights sum to 1 and are non-negative.
        """
        alpha, _ = self.forward(s)
        weights = np.random.dirichlet(alpha)
        return weights, alpha

    def log_prob(self, weights, alpha):
        """
        Log-probability of sampled weights under Dirichlet(α).
        log p(w|α) = log Γ(Σαᵢ) − Σ log Γ(αᵢ) + Σ (αᵢ−1) log wᵢ
        """
        from scipy.special import gammaln  # or implement lgamma via Stirling
        log_p = (gammaln(np.sum(alpha))
                 - np.sum(gammaln(alpha))
                 + np.sum((alpha - 1) * np.log(weights + 1e-8)))
        return log_p

    def update(self, trajectories):
        """
        REINFORCE gradient: ∇θ J = 𝔼[∇θ log π(a|s) · G_t]
        with baseline (mean return) to reduce variance.
        """
        all_returns = [g for _, _, g, _ in trajectories]
        baseline = np.mean(all_returns)  # variance reduction baseline

        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)

        for state, weights, G, alpha in trajectories:
            advantage = G - baseline  # G_t − b(s)

            # ∂log p/∂α: gradient of Dirichlet log-prob w.r.t. concentration
            from scipy.special import digamma
            d_log_p_d_alpha = (digamma(np.sum(alpha))
                               - digamma(alpha)
                               + np.log(weights + 1e-8))

            # Chain rule: ∂log p/∂raw via softplus
            sig = 1 / (1 + np.exp(-np.clip(alpha, -20, 20)))
            d_log_p_d_raw = d_log_p_d_alpha * sig

            # Backprop through network
            h = np.tanh(state @ self.W1 + self.b1)
            grad_W2 = np.outer(h, d_log_p_d_raw) * advantage
            grad_b2 = d_log_p_d_raw * advantage
            d_h = d_log_p_d_raw @ self.W2.T * advantage
            d_tanh = d_h * (1 - h**2)
            grad_W1 = np.outer(state, d_tanh)
            grad_b1 = d_tanh

            dW2 += grad_W2; db2 += grad_b2
            dW1 += grad_W1; db1 += grad_b1

        n = len(trajectories)
        for p, g in [(self.W1,dW1/n),(self.b1,db1/n),(self.W2,dW2/n),(self.b2,db2/n)]:
            p += self.lr * np.clip(g, -1, 1)


class REINFORCEAgent:
    """Monte Carlo policy gradient for portfolio optimization."""

    def __init__(self, state_dim, n_assets, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, n_assets)
        self.gamma = gamma
        self.trajectory = []

    def select_action(self, state):
        weights, alpha = self.policy.sample_action(state)
        return weights, alpha

    def store(self, state, weights, reward, alpha):
        self.trajectory.append((state, weights, reward, alpha))

    def learn(self):
        """Compute discounted returns G_t and update policy."""
        T = len(self.trajectory)
        G = 0
        # Compute discounted returns backwards
        discounted = []
        for _, _, r, _ in reversed(self.trajectory):
            G = r + self.gamma * G
            discounted.insert(0, G)

        # Normalize returns for numerical stability
        discounted = np.array(discounted)
        discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

        grad_data = [(s, w, G, a) for (s,w,_,a), G in zip(self.trajectory, discounted)]
        self.policy.update(grad_data)
        self.trajectory = [] 