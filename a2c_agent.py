import numpy as np


class LSTMCell:
    """
    Single LSTM cell implemented in numpy.
    Equations:
      fₜ = σ(Wf·[h,x] + bf)   ← forget gate
      iₜ = σ(Wi·[h,x] + bi)   ← input gate
      g̃ₜ = tanh(Wg·[h,x]+bg) ← cell gate
      oₜ = σ(Wo·[h,x] + bo)   ← output gate
      cₜ = fₜ⊙cₜ₋₁ + iₜ⊙g̃ₜ  ← cell state
      hₜ = oₜ⊙tanh(cₜ)        ← hidden state
    """

    def __init__(self, input_dim, hidden_dim):
        self.h_dim = hidden_dim
        d = input_dim + hidden_dim
        scale = np.sqrt(1 / hidden_dim)
        # Concatenated weight matrices [Wf, Wi, Wg, Wo] → shape (d, 4·h)
        self.W = np.random.randn(d, 4 * hidden_dim) * scale
        self.b = np.zeros(4 * hidden_dim)
        self.b[hidden_dim:2*hidden_dim] = 1.0  # forget gate bias = 1 (remember by default)
        self.h = np.zeros(hidden_dim)
        self.c = np.zeros(hidden_dim)

    def step(self, x):
        hx = np.concatenate([self.h, x])
        gates = hx @ self.W + self.b
        h = self.h_dim
        f = 1 / (1 + np.exp(-np.clip(gates[:h],   -15, 15)))
        i = 1 / (1 + np.exp(-np.clip(gates[h:2*h],  -15, 15)))
        g = np.tanh(np.clip(gates[2*h:3*h], -15, 15))
        o = 1 / (1 + np.exp(-np.clip(gates[3*h:],   -15, 15)))
        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)
        return self.h.copy()

    def reset(self):
        self.h = np.zeros(self.h_dim)
        self.c = np.zeros(self.h_dim)


class ActorCriticNetwork:
    """
    Shared LSTM encoder → Actor head + Critic head.
    Actor:  outputs Dirichlet α for portfolio weights
    Critic: outputs scalar V(s)
    """

    def __init__(self, state_dim, n_assets, hidden=64, lr_actor=1e-3, lr_critic=5e-3):
        self.n_assets = n_assets
        self.lr_a = lr_actor
        self.lr_c = lr_critic
        self.lstm = LSTMCell(state_dim, hidden)

        # Actor head: hidden → n_assets (Dirichlet α)
        s = np.sqrt(2/hidden)
        self.Wa = np.random.randn(hidden, n_assets) * s
        self.ba = np.ones(n_assets)

        # Critic head: hidden → 1 (V(s))
        self.Wc = np.random.randn(hidden, 1) * s
        self.bc = np.zeros(1)

    def forward(self, state):
        h = self.lstm.step(state)
        alpha = np.log1p(np.exp(h @ self.Wa + self.ba)) + 0.5
        value = (h @ self.Wc + self.bc)[0]
        return alpha, value, h

    def reset_lstm(self):
        self.lstm.reset()


class A2CAgent:
    """
    Advantage Actor-Critic agent.
    Uses n-step returns for reduced variance.
    """

    def __init__(self, state_dim, n_assets, gamma=0.99, entropy_coef=0.01, n_steps=16):
        self.net = ActorCriticNetwork(state_dim, n_assets)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        self.buffer = []   # (state, weights, reward, alpha, value)

    def select_action(self, state):
        alpha, value, _ = self.net.forward(state)
        weights = np.random.dirichlet(alpha)
        return weights, alpha, value

    def store(self, s, w, r, alpha, v):
        self.buffer.append((s, w, r, alpha, v))

    def learn(self, next_state=None, done=False):
        """
        Compute advantages and update both actor and critic.
        A(s,a) = G_t − V(s)   where G_t = Σ γᵏ rₜ₊ₖ + γⁿ V(sₙ)
        """
        if not self.buffer:
            return

        # Bootstrap value
        if done or next_state is None:
            G = 0.0
        else:
            _, G, _ = self.net.forward(next_state)

        # Compute n-step returns backwards
        returns = []
        for _, _, r, _, _ in reversed(self.buffer):
            G = r + self.gamma * G
            returns.insert(0, G)

        actor_grad_total = {'Wa': np.zeros_like(self.net.Wa),
                             'ba': np.zeros_like(self.net.ba)}
        critic_loss_total = 0.0

        for (s, w, _, alpha, v), G in zip(self.buffer, returns):
            advantage = G - v  # Advantage: A(s,a) = G - V(s)

            # ── Critic update (minimize TD MSE) ──────────────────────
            critic_error = G - v
            # In full implementation: backprop δ through Wc
            delta_Wc = advantage  # simplified gradient
            self.net.Wc += self.net.lr_c * delta_Wc * 0.01
            critic_loss_total += critic_error ** 2

            # ── Actor update (policy gradient + entropy) ─────────────
            from scipy.special import digamma
            d_log_p = (digamma(np.sum(alpha)) - digamma(alpha)
                       + np.log(w + 1e-8))
            # Entropy gradient for Dirichlet: encourages exploration
            entropy_grad = digamma(np.sum(alpha)) - digamma(alpha)
            total_grad = d_log_p * advantage + self.entropy_coef * entropy_grad

            # Simplified: update actor weights proportional to gradient
            sig = 1 / (1 + np.exp(-alpha))
            actor_grad_total['ba'] += total_grad * sig

        n = len(self.buffer)
        self.net.ba += self.net.lr_a * np.clip(actor_grad_total['ba'] / n, -0.5, 0.5)
        self.buffer.clear()
        self.net.reset_lstm()

        return critic_loss_total / n