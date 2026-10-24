"""
Evaluation metrics for HMM intrusion detection.

"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier


STATE_NAMES = ['Normal', 'Suspicious', 'Attack']


def baseline_comparison(X_train, states_train, X_test, states_test):
    """
    Computes three baseline models to contextualize HMM performance:

      1. Majority-class: always predicts the most frequent class in the train set.
      2. Random: randomly draws a class proportionally to the train distribution.
      3. Random Forest (RF): static nonlinear comparator. A single comparison
         does not isolate the contribution of sequence memory.

    Prints a comparison table and returns the results as a dict.
    """
    rng = np.random.default_rng(42)
    train = np.array(states_train, dtype=int)
    test  = np.array(states_test,  dtype=int)
    n     = len(test)

    classes, counts = np.unique(train, return_counts=True)
    majority_class  = classes[np.argmax(counts)]
    probs           = counts / counts.sum()

    majority_pred = np.full(n, majority_class, dtype=int)
    random_pred   = rng.choice(classes, size=n, p=probs)

    print("[evaluation] Fitting RandomForest baseline (n_estimators=50)...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, train)
    rf_pred = rf.predict(X_test)

    attack_label = 2
    results = {}
    print("\n" + "=" * 65)
    print("BASELINE COMPARISON (Static Models)")
    print("=" * 65)
    
    models = [
        ("Majority-class", majority_pred), 
        ("Random", random_pred),
        ("Random Forest", rf_pred)
    ]
    
    for name, pred in models:
        acc = accuracy_score(test, pred)
        f1  = f1_score(test, pred, labels=[attack_label], average=None, zero_division=0)[0]
        print(f"  {name:20s} — Accuracy: {acc:.4f}  |  F1 Attack: {f1:.4f}")
        results[name] = {'accuracy': acc, 'f1_attack': f1}
        
    print("=" * 65)
    return results


def evaluate(states_real, states_pred, state_names=None):
    """
    Evaluates the HMM predictions against ground truth labels.

    Parameters
    ----------
    states_real -> real states (0/1/2)
    states_pred -> predicted states by Viterbi (0/1/2)
    state_names -> state names (default: STATE_NAMES)

    """
    if state_names is None:
        state_names = STATE_NAMES

    real = np.array(states_real, dtype=int)
    pred = np.array(states_pred, dtype=int)

    all_classes = set(range(len(state_names)))
    present_real = set(real.tolist())
    absent = all_classes - present_real
    if absent:
        absent_names = [state_names[c] for c in sorted(absent)]
        print(f"\n[evaluation] ⚠  Classes absent from test (no ground-truth): "
              f"{absent_names}")
        print("            These classes will not be evaluated in the report.")

    present_classes = sorted(set(real.tolist() + pred.tolist()))
    present_names   = [state_names[c] for c in present_classes]

    print("\n" + "=" * 65)
    print("HMM CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(
        real, pred,
        labels=present_classes,
        target_names=present_names,
        zero_division=0
    ))

    conf_matrix = confusion_matrix(real, pred, labels=list(range(len(state_names))))
    accuracy    = accuracy_score(real, pred)

    print(f"Global Accuracy : {accuracy:.4f}")

    # --- Recall, Precision, F1 for the Attack class (State 2) ---
    tp_attack = conf_matrix[2, 2]
    fn_attack = conf_matrix[2, :].sum() - tp_attack
    fp_attack = conf_matrix[:, 2].sum() - tp_attack

    attack_recall = tp_attack / (tp_attack + fn_attack) if (tp_attack + fn_attack) > 0 else 0.0
    attack_precision = tp_attack / (tp_attack + fp_attack) if (tp_attack + fp_attack) > 0 else 0.0

    if (attack_precision + attack_recall) > 0:
        attack_f1 = 2 * attack_precision * attack_recall / (attack_precision + attack_recall)
    else:
        attack_f1 = 0.0

    print(f"Attack Recall    : {attack_recall:.4f}  ← IDS sensitivity (true positives)")
    print(f"Attack Precision : {attack_precision:.4f}  ← IDS false alarms (misclassified normal traffic)")
    print(f"Attack F1        : {attack_f1:.4f}")

    return conf_matrix, accuracy, attack_recall, attack_precision, attack_f1
