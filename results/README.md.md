# Result files

The 26 files in this folder are the results behind the paper "CamSpokenNLI: A Pilot Resource for Cross-Variety Transfer Evaluation in Cameroon Pidgin English and Camfranglais" (v1.0). The top-level `README.md` explains how to run the code and lists the settings used. This page lists every file here, which script makes it, and what it contains. The files are in the same order as the run table in that README. `run_manifest.json` is in the repository root.

The files in this folder are released under the PolyForm Noncommercial License 1.0.0, like the code. `asr_transcripts_raw.csv` and `appendix_c_asr_transcripts.csv` contain the premise texts, which are part of the dataset

## Setup and data checks

| File | Made by | Contents |
|---|---|---|
| `shared_item_folds.csv` | `00b` | The one shared 5-fold split: Pidgin id, Camfranglais id, label, fold (fold seed 42). |
| `environment_nli.json` | `00b` | GPU and package versions of the original run. |
| `audio_preprocessing_log.csv` | `01_A5` (provenance) | One row per clip: source file, source length, where speech starts and ends, lead removed, tail kept, final length, flags. |
| `content_check.csv` | `01_A8` | Does each clip say its own sentence? A quick Whisper-small pass, not a reported result; flagged clips are marked. |
| `content_check_transcripts.csv` | `01_A8` | The transcripts used by that check. |
| `audio_checksums.csv` | `04_E` | SHA-256, length, sample rate, channels and format of each of the 228 released audio files. |

## NLI runs

| File | Made by | Contents |
|---|---|---|
| `fold_results.csv` | `02_C3`, `02_C4`, `02_C5`, `02_C6` | One row per training run (276): experiment, split type, direction, train and test variety, condition, epochs, seed, fold, correct, n_test, accuracy, seconds. |
| `item_predictions.csv` | `02_C3`, `02_C4`, `02_C5`, `02_C6` | One row per test item per run: true label, predicted label, correct, class probabilities. |

How to read `fold_results.csv`:

| Column | Values |
|---|---|
| `experiment` | `within_native`, `within_gloss`, `within_premise_only`, `within_hypothesis_only` (30 runs each), `transfer_disjoint` (60), `epoch_sensitivity` (90), `transfer_old_counterpart_leak` (6) |
| `condition` | `native_full`, `english_gloss`, `premise_only`, `hypothesis_only` |
| `direction` | `within`, `pidgin_to_camfranglais`, `camfranglais_to_pidgin` |
| `epochs`, `seed`, `fold` | 15 for the main runs and 5, 10, 20 for epoch sensitivity; seeds 42, 123, 2024; folds 0 to 4 |

## NLI tables and statistics

| File | Made by | Contents |
|---|---|---|
| `table1_main_nli_matrix.csv` | `02_C7` | The main NLI table: training data by test variety. |
| `baselines_within.csv` | `02_C7` | Within-variety accuracy for each condition (native, English gloss, premise-only, hypothesis-only), over the three seeds. |
| `baselines_hypothesis_only_transfer.csv` | `02_C7` | Hypothesis-only baseline in the two transfer directions. |
| `appendix_b_per_seed_main_conditions.csv` | `02_C7` | Per-seed accuracy for the main conditions (the three repeated runs). |
| `appendix_d_disjoint_transfer_per_seed.csv` | `02_C7` | Per-seed accuracy for disjoint-item transfer. |
| `appendix_d_old_counterpart_leak.csv` | `02_C7` | Counterpart-leak audit (Appendix D.4): per-seed accuracy when a test item's counterpart is in the training set. |
| `binomial_tests_all_seed_results.csv` | `02_C7` | Every per-seed result with the above-chance test (exact one-sided binomial test against 1/3). |
| `epoch_sensitivity.csv` | `02_C7` | Accuracy at 5, 10, 15 and 20 epochs (the 15-epoch values are the main runs). |
| `epoch_sensitivity_per_seed.csv` | `02_C7` | The same, per seed. |
| `mcnemar_full_vs_hypothesis_only.csv` | `02_C8` | Paired comparison (exact McNemar test) of full input against hypothesis only, on the same test items, one test per seed. |

## ASR (Whisper zero-shot)

| File | Made by | Contents |
|---|---|---|
| `asr_transcripts_raw.csv` | `03_D` | The 2,052 raw transcripts: variety, model, condition, id, audio path, reference, hypothesis, detected language. |
| `appendix_c_asr_transcripts.csv` | `03_D` | The same transcripts with normalised reference and hypothesis and the per-clip alignment counts (hits, substitutions, deletions, insertions). |
| `asr_summary.csv` | `03_D` | WER, CER, MER, WIL and WIP for each variety, model and condition (18 rows), with word counts and the language tags. |
| `table2_asr_wer_all_conditions.csv` | `03_D` | The ASR table: WER (%) by variety, condition and model. |
| `asr_wer_bootstrap_ci.csv` | `03_D2` | 95% bootstrap intervals for WER (recordings resampled 2,000 times, seed 0). |
| `asr_script_and_empty_counts.csv` | `03_D2` | Script and empty-transcript checks. |
| `asr_examples_large_v3.csv` | `03_D2` | Example Large-v3 transcripts. |
| `asr_raw_wer.csv` | `03_D3` | WER on the unnormalised transcripts (whitespace-split; `<\|...\|>` tags removed from hypotheses). |

## One note on the audio path

In `asr_transcripts_raw.csv` and `appendix_c_asr_transcripts.csv` the `audio_path` column reads `processed_audio/pidgin/...` or `processed_audio/camfranglais/...`. That was the name of the audio folder during the run. The released folder is `audio/`, with the same files. The column is a text label only.
