"""Categorical HMM algorithms adapted from Rafael Ode Brino's Markov study.

A[i, j] = P(state[t + 1] = j | state[t] = i).
B[i, k] = P(observation[t] = k | state[t] = i).
Inference preserves exact zero probabilities. Optional initialization smoothing
is applied to counts, rather than adding mass during likelihood evaluation.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.special import logsumexp


class DiscreteHMM:
    """Single-sequence categorical HMM with explicit probability parameters.

    Parameters can be assigned through ``pi``, ``A`` and ``B``. They must be
    finite, nonnegative, and normalized. The generic core has no IDS-specific
    transition ceiling or validation-recall checkpoint selection.
    """

    def __init__(self, n_states: int, n_obs: int):
        for value in (n_states, n_obs):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("State and observation counts must be positive integers.")
        self.N, self.M = int(n_states), int(n_obs)
        self.pi = np.full(self.N, 1.0 / self.N)
        self.A = np.full((self.N, self.N), 1.0 / self.N)
        self.B = np.full((self.N, self.M), 1.0 / self.M)

    def _observations(self, values) -> np.ndarray:
        obs = np.asarray(values)
        if obs.ndim != 1 or obs.size == 0 or obs.dtype.kind not in "iu":
            raise ValueError("Observations must be a nonempty 1D integer sequence.")
        if np.any(obs < 0) or np.any(obs >= self.M):
            raise ValueError(f"Observation symbols must lie in [0, {self.M}).")
        return obs.astype(np.int64, copy=False)

    def _log_parameters(self):
        logs = []
        for name, shape in [("pi", (self.N,)), ("A", (self.N, self.N)), ("B", (self.N, self.M))]:
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != shape or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name} must have shape {shape} and finite nonnegative probabilities.")
            if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-12, rtol=1e-10):
                raise ValueError(f"{name} probabilities must sum to one.")
            with np.errstate(divide="ignore"):
                logs.append(np.log(values))
        return tuple(logs)

    def fit_supervised(self, states_seq, obs_seq, pseudocount: float = 1e-6):
        """Initialize from one labeled sequence, with optional count smoothing.

        ``pi`` is estimated from the first state, not state occupancy. For
        several independent sequences, their initial states should be pooled
        explicitly; that multi-sequence estimator is outside this small API.
        """
        obs = self._observations(obs_seq)
        states = np.asarray(states_seq)
        if (states.shape != obs.shape or states.dtype.kind not in "iu"
                or np.any(states < 0) or np.any(states >= self.N)):
            raise ValueError("States must be integer indices matching the observation sequence.")
        if not np.isfinite(pseudocount) or pseudocount < 0:
            raise ValueError("pseudocount must be finite and nonnegative.")
        initial = np.full(self.N, float(pseudocount))
        initial[states[0]] += 1
        transitions = np.full((self.N, self.N), float(pseudocount))
        emissions = np.full((self.N, self.M), float(pseudocount))
        np.add.at(transitions, (states[:-1], states[1:]), 1)
        np.add.at(emissions, (states, obs), 1)
        self.pi = initial / initial.sum()
        self.A = self._normalize_counts(transitions, np.full_like(transitions, 1 / self.N))
        self.B = self._normalize_counts(emissions, np.full_like(emissions, 1 / self.M))
        return self

    @staticmethod
    def _normalize_counts(counts, fallback):
        totals = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, totals, out=np.asarray(fallback, dtype=float).copy(), where=totals > 0)

    def forward_log(self, obs_seq):
        """Return log alpha and log P(observations), including -inf if impossible."""
        obs = self._observations(obs_seq)
        log_pi, log_A, log_B = self._log_parameters()
        alpha = np.empty((len(obs), self.N))
        alpha[0] = log_pi + log_B[:, obs[0]]
        for t in range(1, len(obs)):
            alpha[t] = logsumexp(alpha[t - 1, :, None] + log_A, axis=0) + log_B[:, obs[t]]
        return alpha, float(logsumexp(alpha[-1]))

    def backward_log(self, obs_seq):
        """Return log beta, where beta[t, i] is the future-observation likelihood."""
        obs = self._observations(obs_seq)
        _, log_A, log_B = self._log_parameters()
        beta = np.zeros((len(obs), self.N))
        for t in range(len(obs) - 2, -1, -1):
            beta[t] = logsumexp(log_A + log_B[:, obs[t + 1]] + beta[t + 1], axis=1)
        return beta

    def compute_gamma(self, log_alpha, log_beta):
        """Return log P(state[t] | the complete observation sequence)."""
        alpha, beta = np.asarray(log_alpha), np.asarray(log_beta)
        if alpha.ndim != 2 or alpha.shape != beta.shape or alpha.shape[1] != self.N or not len(alpha):
            raise ValueError("Forward/backward arrays must have matching nonempty (T, N) shapes.")
        joint = alpha + beta
        norm = logsumexp(joint, axis=1, keepdims=True)
        if not np.isfinite(norm).all():
            raise ValueError("State posteriors are undefined for a zero-probability sequence.")
        return joint - norm

    def compute_xi(self, obs_seq, log_alpha, log_beta):
        """Return log posterior transition probabilities, shape (T - 1, N, N)."""
        obs = self._observations(obs_seq)
        _, log_A, log_B = self._log_parameters()
        alpha, beta = np.asarray(log_alpha), np.asarray(log_beta)
        if alpha.shape != (len(obs), self.N) or beta.shape != alpha.shape:
            raise ValueError("Forward/backward arrays must match the observation sequence.")
        if not np.isfinite(logsumexp(alpha[-1])):
            raise ValueError("Transition posteriors are undefined for a zero-probability sequence.")
        xi = np.empty((len(obs) - 1, self.N, self.N))
        for t in range(len(obs) - 1):
            joint = (alpha[t, :, None] + log_A + log_B[:, obs[t + 1]][None, :]
                     + beta[t + 1][None, :])
            xi[t] = joint - logsumexp(joint)
        return xi

    def filter_probabilities(self, obs_seq):
        """Return P(state[t] | observations through t), without future observations."""
        alpha, _ = self.forward_log(obs_seq)
        norm = logsumexp(alpha, axis=1, keepdims=True)
        if not np.isfinite(norm).all():
            raise ValueError("Filtered probabilities are undefined for an impossible prefix.")
        return np.exp(alpha - norm)

    def baum_welch(self, obs_seq, n_iter: int = 20, tol: float = 1e-8):
        """Run unconstrained EM and return ``(history, final_log_likelihood)``.

        History includes the initial likelihood and each post-update value.
        The returned score therefore corresponds to the returned parameters.
        Unvisited rows retain their previous distribution. No smoothing is
        introduced during EM; strictly positive initialization is recommended.
        """
        obs = self._observations(obs_seq)
        if isinstance(n_iter, bool) or not isinstance(n_iter, Integral) or n_iter < 0:
            raise ValueError("n_iter must be a nonnegative integer.")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative.")
        alpha, score = self.forward_log(obs)
        if not np.isfinite(score):
            raise ValueError("EM requires a sequence with positive probability under the initial model.")
        history = [score]
        for _ in range(n_iter):
            beta = self.backward_log(obs)
            gamma = np.exp(self.compute_gamma(alpha, beta))
            xi = np.exp(self.compute_xi(obs, alpha, beta))
            self.pi = gamma[0] / gamma[0].sum()
            self.A = self._normalize_counts(xi.sum(axis=0), self.A)
            emission_counts = np.column_stack([gamma[obs == symbol].sum(axis=0) for symbol in range(self.M)])
            self.B = self._normalize_counts(emission_counts, self.B)
            alpha, score = self.forward_log(obs)
            history.append(score)
            if history[-1] - history[-2] <= tol:
                break
        return history, history[-1]

    def viterbi(self, obs_seq):
        """Return the most likely complete state path (offline decoding)."""
        obs = self._observations(obs_seq)
        log_pi, log_A, log_B = self._log_parameters()
        delta = np.empty((len(obs), self.N))
        previous = np.zeros((len(obs), self.N), dtype=int)
        delta[0] = log_pi + log_B[:, obs[0]]
        for t in range(1, len(obs)):
            scores = delta[t - 1, :, None] + log_A
            previous[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_B[:, obs[t]]
        if not np.isfinite(np.max(delta[-1])):
            raise ValueError("No valid state path exists for this observation sequence.")
        path = np.zeros(len(obs), dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(len(obs) - 2, -1, -1):
            path[t] = previous[t + 1, path[t + 1]]
        return path.tolist()
