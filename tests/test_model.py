from itertools import product

import numpy as np
import pytest

from hmm_ids import DiscreteHMM


@pytest.fixture
def model():
    m = DiscreteHMM(2, 3)
    m.pi = np.array([0.6, 0.4])
    m.A = np.array([[0.7, 0.3], [0.2, 0.8]])
    m.B = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    return m


def enumerate_paths(m, obs):
    paths = np.array(list(product(range(m.N), repeat=len(obs))))
    probabilities = []
    for path in paths:
        p = m.pi[path[0]] * m.B[path[0], obs[0]]
        for t in range(1, len(obs)):
            p *= m.A[path[t - 1], path[t]] * m.B[path[t], obs[t]]
        probabilities.append(p)
    return paths, np.array(probabilities)


def test_forward_backward_and_viterbi_match_exhaustive_oracle(model):
    obs = [0, 1, 2, 2]
    paths, weights = enumerate_paths(model, obs)
    alpha, score = model.forward_log(obs)
    beta = model.backward_log(obs)
    expected_probability = weights.sum()
    assert np.exp(score) == pytest.approx(expected_probability, rel=1e-12)
    assert np.sum(model.pi * model.B[:, obs[0]] * np.exp(beta[0])) == pytest.approx(expected_probability)
    assert model.viterbi(obs) == paths[weights.argmax()].tolist()


def test_smoothed_state_and_transition_posteriors_match_enumeration(model):
    obs = [0, 1, 2, 2]
    paths, weights = enumerate_paths(model, obs)
    weights /= weights.sum()
    alpha, _ = model.forward_log(obs)
    beta = model.backward_log(obs)
    gamma = np.exp(model.compute_gamma(alpha, beta))
    xi = np.exp(model.compute_xi(obs, alpha, beta))
    for t in range(len(obs)):
        for state in range(model.N):
            assert gamma[t, state] == pytest.approx(weights[paths[:, t] == state].sum())
    for t in range(len(obs) - 1):
        for left in range(model.N):
            for right in range(model.N):
                mask = (paths[:, t] == left) & (paths[:, t + 1] == right)
                assert xi[t, left, right] == pytest.approx(weights[mask].sum())
    np.testing.assert_allclose(gamma.sum(axis=1), 1)
    np.testing.assert_allclose(xi.sum(axis=(1, 2)), 1)


def test_filter_uses_only_each_observed_prefix(model):
    obs = [0, 1, 2, 2]
    filtered = model.filter_probabilities(obs)
    for t in range(1, len(obs) + 1):
        np.testing.assert_allclose(filtered[:t], model.filter_probabilities(obs[:t]))
        paths, weights = enumerate_paths(model, obs[:t])
        expected = [weights[paths[:, -1] == state].sum() / weights.sum() for state in range(model.N)]
        np.testing.assert_allclose(filtered[t - 1], expected)


def test_em_history_is_monotone_and_score_matches_final_parameters(model):
    observations = np.random.default_rng(42).integers(0, 3, 200)
    history, selected_score = model.baum_welch(observations, n_iter=15, tol=0)
    assert len(history) >= 2
    assert np.all(np.diff(history) >= -1e-9)
    assert selected_score == pytest.approx(model.forward_log(observations)[1])
    np.testing.assert_allclose(model.A.sum(axis=1), 1)
    np.testing.assert_allclose(model.B.sum(axis=1), 1)
    assert model.pi.sum() == pytest.approx(1)


@pytest.mark.parametrize("n_states", [1, 2, 3, 4])
def test_em_has_no_hard_coded_three_state_constraint(n_states):
    m = DiscreteHMM(n_states, 3)
    history, score = m.baum_welch([0, 1, 1, 0, 2, 2], n_iter=3)
    assert np.isfinite(score)
    assert np.all(np.diff(history) >= -1e-9)


def test_exact_zeros_do_not_create_impossible_transitions():
    m = DiscreteHMM(2, 2)
    m.pi = np.array([1.0, 0.0]); m.A = np.eye(2); m.B = np.eye(2)
    assert m.forward_log([0, 1])[1] == -np.inf
    assert m.viterbi([0, 0, 0]) == [0, 0, 0]
    with pytest.raises(ValueError, match="No valid"):
        m.viterbi([0, 1])
    with pytest.raises(ValueError, match="positive probability"):
        m.baum_welch([0, 1])
    alpha, _ = m.forward_log([0, 1])
    with pytest.raises(ValueError, match="zero-probability"):
        m.compute_gamma(alpha, m.backward_log([0, 1]))


def test_long_sequence_avoids_probability_underflow(model):
    _, score = model.forward_log(np.tile([0, 1, 2], 1000))
    assert np.isfinite(score) and score < -745


def test_single_observation_and_zero_iterations(model):
    initial = model.A.copy()
    history, score = model.baum_welch([2], n_iter=2)
    np.testing.assert_allclose(model.A, initial)
    assert score == pytest.approx(model.forward_log([2])[1])
    history, score = model.baum_welch([2], n_iter=0)
    assert history == [score]


def test_supervised_initialization_estimates_initial_state_not_occupancy():
    m = DiscreteHMM(2, 3).fit_supervised([0, 1, 1, 1], [0, 1, 2, 2], pseudocount=0)
    np.testing.assert_allclose(m.pi, [1, 0])
    np.testing.assert_allclose(m.A, [[0, 1], [0, 1]])
    np.testing.assert_allclose(m.B.sum(axis=1), 1)


@pytest.mark.parametrize("obs", [[], [-1], [3], [0.0], [[0]], [True]])
def test_bad_observations_are_rejected(model, obs):
    with pytest.raises(ValueError):
        model.forward_log(obs)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_bad_dimensions_are_rejected(size):
    with pytest.raises(ValueError):
        DiscreteHMM(size, 3)


@pytest.mark.parametrize("kind", ["negative", "unnormalized", "nan", "shape"])
def test_bad_parameters_are_rejected(model, kind):
    if kind == "negative": model.pi = np.array([-0.1, 1.1])
    elif kind == "unnormalized": model.A *= 2
    elif kind == "nan": model.B[0, 0] = np.nan
    else: model.B = np.ones((2, 2)) / 2
    with pytest.raises(ValueError): model.forward_log([0])
