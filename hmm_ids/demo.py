"""Seeded inference/learning example; no external data or IDS performance claim."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import DiscreteHMM


def run_demo(output_dir: Path, seed: int = 42):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    known = DiscreteHMM(2, 3)
    known.pi = np.array([0.65, 0.35])
    known.A = np.array([[0.94, 0.06], [0.10, 0.90]])
    known.B = np.array([[0.75, 0.20, 0.05], [0.05, 0.20, 0.75]])
    states = np.zeros(400, dtype=int)
    observations = np.zeros(400, dtype=int)
    states[0] = rng.choice(known.N, p=known.pi)
    for t in range(len(states)):
        if t:
            states[t] = rng.choice(known.N, p=known.A[states[t - 1]])
        observations[t] = rng.choice(known.M, p=known.B[states[t]])
    alpha, score = known.forward_log(observations)
    filtered = known.filter_probabilities(observations)
    smoothed = np.exp(known.compute_gamma(alpha, known.backward_log(observations)))
    decoded = known.viterbi(observations)
    learned = DiscreteHMM(2, 3)
    learned.pi = rng.dirichlet(np.ones(2))
    learned.A = rng.dirichlet(np.ones(2), size=2)
    learned.B = rng.dirichlet(np.ones(3), size=2)
    history, final_score = learned.baum_welch(observations, n_iter=20)
    records = np.column_stack([observations, states, decoded, filtered[:, 1], smoothed[:, 1]])
    np.savetxt(output_dir / "inference.csv", records, delimiter=",", comments="",
               header="observation,generated_state,viterbi_known_model,filtered_state1,smoothed_state1",
               fmt=["%d", "%d", "%d", "%.12g", "%.12g"])
    summary = {
        "data_kind": "synthetic_categorical_hmm", "seed": seed, "observations": len(states),
        "observation_sha256": hashlib.sha256(observations.astype("<i8").tobytes()).hexdigest(),
        "known_model_log_likelihood": score,
        "em_history": history, "em_final_log_likelihood": final_score,
        "filtered_row_sum_max_error": float(np.max(np.abs(filtered.sum(axis=1) - 1))),
        "python": sys.version.split()[0],
        "packages": {name: version(name) for name in ["numpy", "scipy", "matplotlib"]},
        "interpretation": "Known-model inference and a separate unsupervised EM fit on generated symbols; no IDS benchmark.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
    shown = slice(0, 120)
    axes[0].step(np.arange(120), states[shown], where="mid", label="Generated state", color="#263b53")
    axes[0].plot(np.arange(120), np.asarray(decoded)[shown], ".", label="Viterbi, known model", color="#d47728", markersize=3)
    axes[0].set_ylabel("State"); axes[0].set_yticks([0, 1]); axes[0].legend(loc="upper right")
    axes[1].plot(filtered[shown, 1], label="Filtered: observations through t", color="#3276a5")
    axes[1].plot(smoothed[shown, 1], label="Smoothed: complete sequence", color="#d47728", alpha=0.8)
    axes[1].set_ylabel("P(state 1)"); axes[1].set_xlabel("First 120 sequence positions")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(range(len(history)), history, "o-", color="#3276a5", markersize=4)
    axes[2].set_xlabel("Unconstrained EM updates (separate fitted model)")
    axes[2].set_ylabel("Train log likelihood")
    for ax in axes: ax.grid(alpha=0.2)
    fig.suptitle("Synthetic categorical HMM: inference and learning", fontsize=14)
    fig.savefig(output_dir / "inference-demo.png", dpi=160)
    plt.close(fig)
    print(f"Generated-data example and provenance: {output_dir / 'summary.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_demo(args.output_dir, args.seed)


if __name__ == "__main__": main()
