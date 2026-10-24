# Numerical implementation and verification

The maintained `hmm_ids` package is a reusable categorical-HMM core adapted from the supplied study implementation. It retains explicit forward/backward, posterior, EM, and Viterbi algorithms rather than calling a packaged HMM estimator.

## Conventions

- `pi[i] = P(S_0=i)`; transition rows and emission rows sum to one.
- `A[i,j] = P(S_(t+1)=j | S_t=i)`.
- `B[i,k] = P(O_t=k | S_t=i)`.
- Observations are a nonempty integer sequence with symbols `0..M-1`.
- Forward/backward quantities are evaluated using logs and SciPy `logsumexp`.
- Exact zero probabilities map to negative infinity. An impossible observation sequence has likelihood zero; conditional posteriors and a Viterbi path are then undefined and rejected explicitly.

The implementation stores the forward/backward tables and transition posteriors. EM uses O(T N²) time and transition-posterior storage for one sequence. It is a clear reference implementation, not a low-latency or distributed system.

## Study code versus maintained core

| Behavior | Supplied IDS study (`study/hmm_model.py`) | Maintained generic core |
| --- | --- | --- |
| Model size | Experiment uses three states and 64 symbols | Tested with one through four states and arbitrary categorical alphabets |
| Initial-state initialization | Occupancy counts over the labeled sequence | First state of one labeled sequence, with optional count smoothing |
| Likelihood evaluation | Adds epsilon to probability arrays | Uses the normalized parameters, preserving exact zeros |
| Transition restriction | Caps the middle state's self-transition at 0.4 | No IDS-specific transition constraint |
| Model selection | Validation attack-recall checkpoint selection | Unconstrained training-likelihood EM |
| Score return | Recorded training score can precede the selected parameter update | `(history, final_score)`, with post-update scores matching returned parameters |
| Filtering | Primarily full-sequence decoding | Explicit prefix-only filtered probabilities as well as offline decoding |

These are documented implementation changes, not a claim that the report's metrics improved. Keep `study/` when examining the original empirical procedure.

## Independent checks

Small sequences are evaluated by enumerating every possible hidden-state path. This independent oracle verifies total likelihood, Viterbi's maximizing path, marginal state posteriors, and pairwise transition posteriors. Additional tests cover:

- prefix invariance of filtered probabilities;
- EM likelihood progression and agreement between the returned score and final parameters;
- normalized stochastic parameters and generic model sizes;
- exact-zero/impossible paths;
- a sequence long enough to underflow ordinary probability products;
- single-observation and zero-update behavior;
- invalid observation types/ranges and invalid parameter arrays;
- a complete generated-data CLI example from another working directory.

The old private test treated the Baum–Welch return tuple as a plain list. This edition documents the tuple API and tests its history and score separately. The generic EM monotonicity check applies to its unconstrained updates; it is not asserted for the study's capped-transition, validation-selected procedure.

## Example interpretation

The example samples categorical symbols from a known two-state model. Known-model filtering/smoothing/decoding are plotted together to illustrate conditioning on different information. A separately initialized model fits the symbols by unconstrained EM. The last plot shows its training objective, not a held-out predictive gain. Generated state IDs and fitted state IDs are not assumed semantically interchangeable.
