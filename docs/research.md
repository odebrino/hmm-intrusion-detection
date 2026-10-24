# Ordered-stream intrusion-detection study

Author: **Rafael Ode Brino**. Report: [HMM-IDS for DDoS Detection](../report/Report.pdf), April 2026, six pages. The author line was corrected on 24 October 2026 following the author's explicit sole-authorship confirmation. Scientific text, numerical tables, and diagnostic figures were preserved.

## Source material

The public repository contains the supplied report, five study modules, and five diagnostic figures. `report/source-manifest.json` records source-file hashes and the limited CLI/terminology changes made when packaging the study. The maintained generic algorithms are in `hmm_ids/`; their differences are documented in [numerics](numerics.md).

The raw CSV and fitted experiment checkpoint were not supplied. The current release verifies its generic numerical implementation and generated example; it does not claim an independent reproduction of every reported number.

## Protocol

The source is the CICIDS2017 Friday afternoon DDoS flow CSV, described by the report as 225,745 raw rows. The experiment cleans invalid entries, retains seven numerical features, and takes 3,000 benign and 3,000 DDoS records. It alternates 100-record class-selected blocks, then uses ordered 70%/15%/15% slices: 4,200 training, 900 validation, and 900 test records.

`RobustScaler` and a 64-cluster K-Means codebook are fitted on training data. The three model states are label-anchored Normal and Attack plus an unlabeled intermediate state called Suspicious. Supervised initialization is followed by the manual transition adjustment and constrained Baum–Welch procedure described in the report. Validation Attack recall selects the checkpoint. Viterbi decodes the complete test sequence.

This preserves order inside extracted blocks but constructs the global sequence using labels. It is not the untouched observation chronology. The report's streaming/chronological language should be read with that boundary. Offline Viterbi predictions can use observations later in the supplied test sequence.

## Reported results and trade-off

The report's confusion matrix is:

| True state / predicted state | Normal | Suspicious | Attack |
| --- | ---: | ---: | ---: |
| Normal | 369 | 29 | 2 |
| Suspicious | 0 | 0 | 0 |
| Attack | 0 | 0 | 500 |

This gives 869/900 exact matches (96.56%), attack recall 100%, attack precision 500/502 (99.60%), and attack F1 99.80%. These are report-grounded values, not confidence bounds or independent deployment measurements.

RF is reported at 97.89% overall accuracy and 98.13% attack F1. The HMM's attack-oriented result is stronger in that run, while overall accuracy is lower. The 29 intermediate-state outputs count as errors under exact matching; treating them as alerts changes the practical interpretation of false alarms.

The generic package does not enforce the IDS middle-state ceiling or select parameters by attack recall. Its synthetic EM demonstration cannot be substituted for the study's validation trajectory or its selected checkpoint.

## Data and reproduction

Use the [official CICIDS2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html) to obtain the machine-learning CSV distribution, retaining its citation to Sharafaldin, Lashkari, and Ghorbani (2018). Select `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`. Record the actual file hash and environment; the research dataset is external to this code license.

```bash
python -m pip install '.[study]'
python study/main.py --csv /path/to/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
  --output-dir outputs/study
```

The study expects enough cleaned rows to construct its fixed 6,000-record subset. The software example and tests require no external traffic data. Additional two-state ablations, multiple splits, genuinely chronological evaluation, and uncertainty analysis remain future experiments.
