"""
Loading, cleaning, and discretization of the CICIDS2017 dataset.

"""

import numpy as np
import pandas as pd


# Features used for discretization
FEATURES = [
    'Flow Packets/s',
    'Flow Duration',
    'Total Fwd Packets',
    'Bwd Packet Length Std',
    'Flow IAT Max',
    'Packet Length Variance',
    'Bwd IAT Mean'
]
N_CLUSTERS = 64         # increased observation alphabet size (fission of ambiguous clusters)
KMEANS_SEED = 42


def load_and_prepare(filepath, n_samples=None):
    """
    Loads and prepares the dataset for the HMM sequentially.

    Returns the raw log-transformed features (X_log) WITHOUT Scaling or K-Means discretization,
    so the caller can fit the scaler and K-Means strictly on the training set
    to prevent data leakage into the validation/test subsets.

    """

    print(f"[data_loader] Reading file: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)

    df.columns = df.columns.str.strip()
    print(f"[data_loader] Raw shape: {df.shape}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    required_cols = FEATURES + ['Label']

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError("Requested features not found.")

    df.dropna(subset=required_cols, inplace=True)

    # --- Remove negative values for Flow Packets/s ---
    df = df[df['Flow Packets/s'] >= 0].copy()

    # --- Balanced CHUNKED Temporal Selection ---
    benign_pool = df[df['Label'] == 'BENIGN']
    ddos_pool   = df[df['Label'].str.contains('DDoS', case=False)]

    TOTAL_SAMPLES_PER_CLASS = 3000
    CHUNK_SIZE = 100
    n_chunks = TOTAL_SAMPLES_PER_CLASS // CHUNK_SIZE

    chunks = []
    for i in range(n_chunks):
        chunks.append(benign_pool.iloc[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE])
        chunks.append(ddos_pool.iloc[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE])

    df = pd.concat(chunks).reset_index(drop=True)

    print(f"[data_loader] Extracted temporally interleaved chunks (Chunk size={CHUNK_SIZE})")
    print(f"[data_loader] Shape after cleaning and chunk interleaving: {df.shape}")

    # --- Log transformation ONLY ---
    X = df[FEATURES].values.astype(float)
    X = np.clip(X, 0, None)
    X_log = np.log1p(X)

    # --- Map labels to hidden states ---
    def map_label(label):
        label_clean = str(label).strip().upper()
        if label_clean == 'BENIGN':
            return 0
        elif 'DDOS' in label_clean:
            return 2
        else:
            return 1

    df['State'] = df['Label'].apply(map_label)

    # --- Diagnostics ---
    print("\n[data_loader] Real state distribution:")
    state_counts = df['State'].value_counts().sort_index()
    state_names  = {0: 'Normal', 1: 'Suspicious', 2: 'Attack'}
    for s, count in state_counts.items():
        print(f"  State {s} ({state_names[s]}): {count} samples ({100*count/len(df):.1f}%)")

    print("[data_loader] Unscaled log-transformed features ready. Scaler MUST be applied in main.py on Train only.")

    states_seq = df['State'].tolist()

    return X_log, states_seq, df
