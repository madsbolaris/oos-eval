[![Clinc](clinc_logo.png)](https://clinc.com)

# An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction
Repository that accompanies [An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://www.aclweb.org/anthology/D19-1131/).

## GitHub data-to-decision example

This fork adds a reproducible pipeline that converts CLINC150's travel domain into labelled requests for [Kev](https://github.com/jaredpalmer/kev), an open Jev-style decision model. The original files under `data/` remain unchanged.

```bash
python3 scripts/build_kev_dataset.py
python3 -m unittest discover -s tests -v
python3 scripts/build_kev_dataset.py --check
```

Generated partitions are written to `generated/kev/`:

- `train.jsonl`: the native travel training examples and a class-balanced out-of-scope sample
- `calibration.jsonl`: half of each label in the native validation split, used only to fit temperature
- `development.jsonl`: the other half of each validation label, used for model comparison
- `test.jsonl`: a class-balanced sample from the native test split, marked as locked in `manifest.json`

Every record is a TypeSafe-compatible request with a labelled `choice` question. Out-of-scope rows are selected in source order to match one in-domain class in each native split; no row moves between native splits. `model.json` pins the upstream revision, license, domain, and question definition. The manifest records the sampling rule, partition counts, and SHA-256 hashes.

Pull requests run `.github/workflows/validate-kev-data.yml` without secrets. After merging, dispatch **Train Kev** with a unique run name. The protected `kev-training` environment must provide `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`; optional deployment also expects a Modal secret named `kev-serve-key` containing `KEV_API_KEY`.

After selecting one trained run from development results, dispatch **Locked Test Kev** exactly once with a unique result prefix. It evaluates that checkpoint and the released baseline on `test.jsonl`; both temperatures are fitted only on the unchanged calibration partition. Modal result names are immutable to discourage repeated test-set tuning.

The transformed records remain derived from CLINC150 and are distributed under the repository's CC BY 3.0 license. Retain the citation below when using them.


## FAQs
### 1. What are the relevant files?
See `data/data_full.json` for the "full" dataset. This is the dataset used in Table 1 (the "Full" columns). This file contains 150 "in-scope" intent classes, each with 100 train, 20 validation, and 30 test samples. There are 100 train and validation out-of-scope samples, and 1000 out-of-scope test samples. 

### 2. What is the name of the dataset?
The dataset was not given a name in the original paper, but [others](https://arxiv.org/pdf/2003.04807.pdf) have called it `CLINC150`.

### 3. What is this dataset for?
This dataset is for evaluating the performance of intent classification systems in the presence of "out-of-scope" queries. By "out-of-scope", we mean queries that do not fall into any of the system-supported intent classes. Most datasets include only data that is "in-scope". Our dataset includes both in-scope and out-of-scope data. You might also know the term "out-of-scope" by other terms, including "out-of-domain" or "out-of-distribution". 

### 4. What language is the dataset in?
All queries are in English.

### 5. How does your dataset/evaluation handle multi-intent queries?
All samples/queries in our dataset are single-intent samples. We consider the problem of multi-intent classification to be future work.

### 6. How did you gather the dataset?
We used crowdsourcing to generate the dataset. We asked crowd workers to either paraphrase "seed" phrases, or respond to scenarios (e.g. "pretend you need to book a flight, what would you say?"). We used crowdsourcing to generate data for both in-scope and out-of-scope data.

## Citation

If you find our dataset useful, please be sure to cite:

```
@inproceedings{larson-etal-2019-evaluation,
    title = "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
    author = "Larson, Stefan  and
      Mahendran, Anish  and
      Peper, Joseph J.  and
      Clarke, Christopher  and
      Lee, Andrew  and
      Hill, Parker  and
      Kummerfeld, Jonathan K.  and
      Leach, Kevin  and
      Laurenzano, Michael A.  and
      Tang, Lingjia  and
      Mars, Jason",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-1131"
}
```
