"""
Implementation of a discrete HMM

"""

import numpy as np
from scipy.special import logsumexp

EPSILON = 1e-6   # smoothing to avoid log(0)
LOG_ZERO = -np.inf

class DiscreteHMM:
    """
    Discrete Hidden Markov Model.

    Parameters:

    n_states -> number of hidden states
    n_obs    -> size of observation alphabet

    """
    def __init__(self, n_states: int, n_obs: int):
        self.N = n_states
        self.M = n_obs

        self.pi = np.full(self.N, 1.0 / self.N)
        self.A  = np.full((self.N, self.N), 1.0 / self.N)
        self.B  = np.full((self.N, self.M), 1.0 / self.M)

    # ------------------------------------------------------------------
    # 1.  Supervised Initialization
    # ------------------------------------------------------------------

    def fit_supervised(self, states_seq, obs_seq):
        """
        Initializes A, B, pi from labeled sequences.
        """
        states = np.array(states_seq, dtype=int)
        obs    = np.array(obs_seq,    dtype=int)
        T = len(states)

        # pi estimation
        pi = np.zeros(self.N)
        for s in states:
            pi[s] += 1.0
        pi += EPSILON
        self.pi = pi / pi.sum()

        # A estimation (transition counts based on chronological sequence)
        A = np.zeros((self.N, self.N))
        for t in range(T - 1):
            A[states[t], states[t + 1]] += 1.0
        A += EPSILON
        self.A = A / A.sum(axis=1, keepdims=True)

        # B estimation (emission symbol distributions per state)
        B = np.zeros((self.N, self.M))
        for t in range(T):
            B[states[t], obs[t]] += 1.0
        B += EPSILON
        self.B = B / B.sum(axis=1, keepdims=True)

        print("[HMM] Supervised initialization completed.")
        print(f"  π  = {self.pi.round(3)}")
        print(f"  A  =\n{self.A.round(3)}")
        print(f"  B  =\n{self.B.round(3)}")

    # ------------------------------------------------------------------
    # 2.  Log-space Forward  (alpha-pass)
    # ------------------------------------------------------------------

    def forward_log(self, obs_seq):
        obs = np.array(obs_seq, dtype=int)
        T = len(obs)

        log_A  = np.log(self.A  + EPSILON)
        log_B  = np.log(self.B  + EPSILON)
        log_pi = np.log(self.pi + EPSILON)

        log_alpha = np.full((T, self.N), LOG_ZERO)
        log_alpha[0] = log_pi + log_B[:, obs[0]]

        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1, :, np.newaxis] + log_A, axis=0) + log_B[:, obs[t]]

        log_likelihood = logsumexp(log_alpha[T - 1])
        return log_alpha, log_likelihood

    # ------------------------------------------------------------------
    # 3.  Log-space Backward  (beta-pass)
    # ------------------------------------------------------------------

    def backward_log(self, obs_seq):
        obs = np.array(obs_seq, dtype=int)
        T = len(obs)

        log_A = np.log(self.A + EPSILON)
        log_B = np.log(self.B + EPSILON)

        log_beta = np.full((T, self.N), LOG_ZERO)
        log_beta[T - 1] = 0.0

        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_A + log_B[:, obs[t + 1]] + log_beta[t + 1], axis=1
            )

        return log_beta

    # ------------------------------------------------------------------
    # 4.  Computation of ξ
    # ------------------------------------------------------------------

    def compute_xi(self, obs_seq, log_alpha, log_beta):
        obs = np.array(obs_seq, dtype=int)
        T = len(obs)

        log_A = np.log(self.A + EPSILON)
        log_B = np.log(self.B + EPSILON)
        log_xi = np.full((T - 1, self.N, self.N), LOG_ZERO)

        for t in range(T - 1):
            log_xi_t = (
                log_alpha[t, :, np.newaxis]
                + log_A
                + log_B[:, obs[t + 1]][np.newaxis, :]
                + log_beta[t + 1, :][np.newaxis, :]
            )
            norm = logsumexp(log_xi_t.flatten())
            log_xi[t] = log_xi_t - norm

        return log_xi

    # ------------------------------------------------------------------
    # 5.  Computation of γ
    # ------------------------------------------------------------------

    def compute_gamma(self, log_alpha, log_beta):
        log_gamma = log_alpha + log_beta
        norms = logsumexp(log_gamma, axis=1, keepdims=True)
        log_gamma -= norms
        return log_gamma

    # ------------------------------------------------------------------
    # 6.  Baum-Welch (EM)
    # ------------------------------------------------------------------

    def baum_welch(self, obs_seq, n_iter=20, val_obs=None, val_states=None):
        obs = np.array(obs_seq, dtype=int)
        log_likelihoods = []

        print(f"\n[HMM] Starting Baum-Welch ({n_iter} iterations)...")

        best_recall = -1.0
        best_A = None
        best_B = None
        best_pi = None
        best_ll = None
        patience_counter = 0

        for it in range(n_iter):
            self.A  = (self.A  + EPSILON)
            self.A  /= self.A.sum(axis=1, keepdims=True)
            self.B  = (self.B  + EPSILON)
            self.B  /= self.B.sum(axis=1, keepdims=True)
            self.pi = (self.pi + EPSILON)
            self.pi /= self.pi.sum()

            log_alpha, log_ll = self.forward_log(obs)
            log_beta           = self.backward_log(obs)
            log_xi             = self.compute_xi(obs, log_alpha, log_beta)
            log_gamma          = self.compute_gamma(log_alpha, log_beta)

            log_likelihoods.append(log_ll)
            print(f"  Iter {it+1:3d} / {n_iter}  —  log-likelihood = {log_ll:.4f}")

            self.pi = np.exp(log_gamma[0])
            self.pi = np.clip(self.pi, EPSILON, None)
            self.pi /= self.pi.sum()

            log_sum_xi_ij = logsumexp(log_xi, axis=0)          # (N, N)
            log_sum_xi_i  = logsumexp(log_sum_xi_ij, axis=1, keepdims=True)  # (N, 1)
            log_A_new = log_sum_xi_ij - log_sum_xi_i           # (N, N)
            self.A = np.exp(log_A_new)
            self.A = np.clip(self.A, EPSILON, None)

            # --- Topological restriction: limit lingering in Suspicious state ---
            if self.A[1, 1] > 0.4:
                excess = self.A[1, 1] - 0.4
                self.A[1, 1] = 0.4
                other_sum = self.A[1, 0] + self.A[1, 2]
                if other_sum > EPSILON:
                    self.A[1, 0] += excess * (self.A[1, 0] / other_sum)
                    self.A[1, 2] += excess * (self.A[1, 2] / other_sum)
                else:
                    self.A[1, 0] = 0.3
                    self.A[1, 2] = 0.3

            self.A /= self.A.sum(axis=1, keepdims=True)

            T = len(obs)
            new_B = np.full((self.N, self.M), np.log(EPSILON))
            for k in range(self.M):
                mask = (obs == k)
                if mask.any():
                    new_B[:, k] = logsumexp(log_gamma[mask], axis=0)

            log_sum_gamma = logsumexp(log_gamma, axis=0)
            log_B_new = new_B - log_sum_gamma[:, np.newaxis]
            self.B = np.exp(log_B_new)
            self.B = np.clip(self.B, EPSILON, None)
            self.B /= self.B.sum(axis=1, keepdims=True)

            if val_obs is not None and val_states is not None:
                pred_val = self.viterbi(val_obs)
                tp = sum(1 for r, p in zip(val_states, pred_val) if r == 2 and p == 2)
                fn = sum(1 for r, p in zip(val_states, pred_val) if r == 2 and p != 2)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                print(f"    Validation Attack Recall : {recall:.4f}")

                if recall > best_recall:
                    best_recall = recall
                    best_A = self.A.copy()
                    best_B = self.B.copy()
                    best_pi = self.pi.copy()
                    best_ll = log_ll
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 2:
                    print(f"    [Early Stopping] No improvement in validation recall for 2 iterations.")
                    print(f"    Restoring best matrices from highest validation recall (Recall = {best_recall:.4f}, LL = {best_ll:.4f})")
                    self.A = best_A
                    self.B = best_B
                    self.pi = best_pi
                    break

        print(f"[HMM] Baum-Welch finished. Initial LL={log_likelihoods[0]:.2f}, final sequence LL={log_likelihoods[-1]:.2f}")
        return log_likelihoods, (best_ll if best_ll is not None else log_likelihoods[-1])

    # ------------------------------------------------------------------
    # 7.  Log-space Viterbi
    # ------------------------------------------------------------------

    def viterbi(self, obs_seq):
        obs = np.array(obs_seq, dtype=int)
        T = len(obs)

        log_A  = np.log(self.A  + EPSILON)
        log_B  = np.log(self.B  + EPSILON)
        log_pi = np.log(self.pi + EPSILON)

        log_delta = np.full((T, self.N), LOG_ZERO)
        psi       = np.zeros((T, self.N), dtype=int)

        log_delta[0] = log_pi + log_B[:, obs[0]]

        for t in range(1, T):
            for j in range(self.N):
                scores = log_delta[t - 1] + log_A[:, j]
                psi[t, j]       = np.argmax(scores)
                log_delta[t, j] = np.max(scores) + log_B[j, obs[t]]

        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = np.argmax(log_delta[T - 1])
        for t in range(T - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]

        return best_path.tolist()
