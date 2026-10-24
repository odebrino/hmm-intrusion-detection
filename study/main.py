"""
Entry point for the HMM-IDS project (DDoS detection by Hidden Markov Model).

"""

import os
import argparse
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from data_loader   import load_and_prepare, N_CLUSTERS, FEATURES, KMEANS_SEED
from hmm_model     import DiscreteHMM
from evaluation    import evaluate, baseline_comparison
from visualization import (plot_timeline, plot_confusion_matrix,plot_bw_convergence, plot_transition_matrix,plot_emission_matrix)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

CSV_FILENAME  = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
N_STATES      = 3       # Normal (0), Suspicious (1), Attack (2)
N_BW_ITER     = 20      # Baum-Welch iterations
TRAIN_RATIO   = 0.70
RANDOM_STATE  = 42


# -----------------------------------------------------------------------
# Dataset Localization
# -----------------------------------------------------------------------

def find_csv(filename):
    candidates = [
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), '..', filename),
        filename,
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


# -----------------------------------------------------------------------
# Main Program
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Historical ordered-stream CICIDS2017 study.")
    parser.add_argument("--csv", default=CSV_FILENAME, help="Path to the original Friday DDoS flow CSV")
    parser.add_argument("--output-dir", default="outputs/study", help="Directory for newly generated figures")
    args = parser.parse_args()
    print("=" * 70)
    print("  HMM-IDS v3  —  DDoS Attack Detection via Temporal Hidden Markov Model")
    print("  ENSTA — constructed ordered-sequence study")
    print("=" * 70)

    # --- 1. Locate the dataset ---
    csv_path = find_csv(args.csv)
    if csv_path is None:
        print(f"\n[ERROR] CSV file not found: {args.csv}")
        sys.exit(1)

    # --- 2. Loading and temporal chunking ---
    X_log, states_seq, df = load_and_prepare(csv_path)

    N_OBS  = N_CLUSTERS
    total  = len(states_seq)
    X_arr = np.array(X_log)
    states_arr = np.array(states_seq)
    print(f"\n[main] Total samples loaded: {total}")

    # --- 3. Split WITHOUT Shuffling (Preserve Markov Chains) ---
    # Because data_loader interleaved chunks of 100 samples, an ordered
    # slice preserves this constructed order; it is not original trace chronology.
    X_train, X_test, states_train, states_test = train_test_split(
        X_arr, states_arr,
        test_size=round(1 - TRAIN_RATIO, 10),
        shuffle=False
    )
    
    # Create validation set (chronological slice of the remainder)
    X_val, X_test, states_val, states_test = train_test_split(
        X_test, states_test, test_size=0.5, shuffle=False
    )
    print(f"[main] CONSTRUCTED ORDERED Split — Train: {len(X_train)} | Val: {len(X_val)} | Final Test: {len(X_test)}")

    # --- 4. Strict Non-Leaking Preprocessing ---
    print(f"[main] Fitting RobustScaler on the train set only to prevent leakage...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # --- K-Means fitted on TRAIN only ---
    print(f"[main] Fitting K-Means (k={N_OBS}) on the train set only...")
    kmeans = KMeans(n_clusters=N_OBS, random_state=KMEANS_SEED, n_init=10)
    obs_train = kmeans.fit_predict(X_train_scaled)
    obs_val   = kmeans.predict(X_val_scaled)
    obs_test  = kmeans.predict(X_test_scaled)

    # --- Verification of class representation ---
    state_names_map = {0: 'Normal', 1: 'Suspicious', 2: 'Attack'}
    print("\n[main] State distribution in the TEST set:")
    for s in range(N_STATES):
        cnt = np.sum(states_test == s)
        print(f"  └─ State {s} ({state_names_map[s]}): {cnt} samples ({100*cnt/len(states_test):.1f}%)")

    # --- Baseline comparison ---
    baselines = baseline_comparison(X_train_scaled, states_train, X_test_scaled, states_test)

    # --- 5. Supervised Initialization ---
    print("\n[main] Step 1: Supervised HMM initialization...")
    hmm = DiscreteHMM(n_states=N_STATES, n_obs=N_OBS)
    hmm.fit_supervised(states_train.tolist(), obs_train.tolist())

    # --- Adjust Transition Matrix A (Sweet Spot Normal state) ---
    print("[main] Adjusting Transition Matrix A: P(Normal|Normal) = 0.95 (Sweet Spot)")
    hmm.A[0, 0] = 0.95
    hmm.A[0, 1:] = 0.05 / (hmm.N - 1)

    A_initial = hmm.A.copy()

    # Log-likelihood before Baum-Welch
    _, ll_initial = hmm.forward_log(obs_train.tolist())
    print(f"[main] Initial Log-likelihood (train, before BW): {ll_initial:.4f}")

    # --- 6. Refinement via Baum-Welch ---
    print("\n[main] Step 2: Baum-Welch Algorithm (EM)...")
    log_likelihoods, ll_final = hmm.baum_welch(
        obs_train.tolist(), 
        n_iter=N_BW_ITER, 
        val_obs=obs_val.tolist(), 
        val_states=states_val.tolist()
    )

    # --- 7. Viterbi Decoding on test set ---
    print("\n[main] Step 3: Offline Viterbi decoding on the constructed test sequence...")
    states_pred = hmm.viterbi(obs_test.tolist())

    n_susp = sum(1 for s in states_pred if s == 1)
    if n_susp > 0:
        print(f"[WARNING] Viterbi predicted {n_susp} Suspicious latent states ({100*n_susp/len(states_pred):.1f}%).")

    # --- 8. Evaluation ---
    conf_matrix, accuracy, attack_recall, attack_precision, attack_f1 = evaluate(
        states_test.tolist(), states_pred
    )

    # --- 9. Graph Generation ---
    print("\n[main] Generating graphs...")
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    plot_timeline(
        states_test.tolist(), states_pred, n_show=500,
        save_path=os.path.join(out_dir, 'timeline_states.png')
    )
    plot_confusion_matrix(
        conf_matrix,
        save_path=os.path.join(out_dir, 'confusion_matrix.png')
    )
    plot_bw_convergence(
        log_likelihoods,
        ll_before_bw=ll_initial,
        save_path=os.path.join(out_dir, 'bw_convergence.png')
    )
    plot_transition_matrix(
        A_initial, hmm.A,
        save_path=os.path.join(out_dir, 'transition_matrix.png')
    )
    plot_emission_matrix(
        hmm.B,
        save_path=os.path.join(out_dir, 'emission_matrix.png')
    )

    # --- 10. Final Summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (Constructed Ordered-Sequence Study)")
    print("=" * 70)
    print(f"  Split Topology      : CONSTRUCTED ORDERED SLICES (Shuffle=False)")
    print(f"  Data Leakage Status : Train-only preprocessing; constructed-order limitations apply")
    print(f"  Test Setup          : {len(obs_test)} constructed-sequence flow records")
    print(f"  Initial Log-LL      : {ll_initial:.4f}")
    print(f"  Selected Log-LL     : {ll_final:.4f} (Early Stopping)")
    print(f"  Test Accuracy       : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Attack F1 (HMM)     : {attack_f1:.4f}")
    print(f"  Attack F1 (RF Base) : {baselines['Random Forest']['f1_attack']:.4f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
