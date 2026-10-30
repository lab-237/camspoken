# camspoken-nli (v1.0): code and results

Code and result files for the paper **"CamSpokenNLI: A Pilot Resource for Cross-Variety Transfer Evaluation in Cameroon Pidgin English and Camfranglais"** (Tiani Pekins Ebika, University of Buea).

CamSpokenNLI v1.0 has 114 scenario-matched natural language inference items per variety (Cameroon Pidgin English and Camfranglais). The premises are spoken by a single speaker; the hypotheses are text only. The code fine-tunes XLM-RoBERTa-base on the text and runs Whisper zero-shot on the spoken premises.

- **Data and audio:** Hugging Face dataset, [huggingface.co/datasets/lab-237/camspoken](https://huggingface.co/datasets/lab-237/camspoken) (release tag `v1.0`; the original CSV and audio files are in the `source/` folder). Access is gated (free account, accept the non-commercial terms, request access). The data is stored only there, not in this repository.
- **Paper:** arXiv link to be added after posting.

## Contents of this repository

```
camspoken/
├─ README.md                  this file
├─ LICENSE                    code licence (PolyForm Noncommercial 1.0.0)
├─ run_manifest.json          settings, run counts, versions and file hashes of the original run
├─ SHA256SUMS.txt             SHA-256 hashes of pidgin.csv and camfranglais.csv, to check your download
├─ code/                      21 Python scripts that run on the released data, in run order
│  └─ provenance/             5 Python scripts that produced the released data (documentation only)
└─ results/                   26 result files, plus a README.md that describes each one
```

The audio is not in this repository. It is on Hugging Face, in the `source/audio/` folder, as `pidgin/wes_NNNNN.wav` and `camfranglais/camf_NNNNN.wav` (228 files, already trimmed).

## Requirements

The code is **Python 3.13**, written as notebook cells (one `.py` file per cell) for **Google Colab with a T4 GPU**. The NLI and ASR scripts need a GPU; the setup script prints a warning on any other GPU, because the protocol was fixed to a T4.

| Item                                          | Version                                                     |
| --------------------------------------------- | ----------------------------------------------------------- |
| Python                                        | 3.13 (3.13.15 for every run in the paper)                   |
| PyTorch                                       | 2.11.0 with CUDA 13.0 (preinstalled in Colab)               |
| transformers                                  | 5.16.1 (pinned in `00a`)                                    |
| datasets                                      | 4.8.5 (pinned)                                              |
| scikit-learn                                  | 1.6.1 (pinned)                                              |
| scipy                                         | 1.16.3 (pinned)                                             |
| openai-whisper                                | 20250625 (pinned)                                           |
| jiwer                                         | 4.0.0 (pinned; the original run did not record the version) |
| numpy, pandas                                 | numpy 2.1.3, pandas 2.2.3 (as installed in the Colab image) |
| soundfile, librosa, accelerate, sentencepiece | not pinned                                                  |
| ffmpeg                                        | preinstalled in Colab (Whisper uses it to read audio)       |

The versions recorded for the paper are in `results/environment_nli.json` and `run_manifest.json`. A later Colab image with Python 3.13.16 reproduced the setup results (the data checks and the shared-fold fingerprint).

Time on a T4, taken from the scripts: setup about 4 minutes; within-variety NLI (120 runs) about 1 to 1.5 hours; transfer (60 runs) about 55 minutes; epoch sensitivity (90 runs) about 70 minutes; leak audit (6 runs) about 6 minutes; Whisper (2,052 transcripts) about 1 hour. The whole pipeline takes about 4.5 hours. The training scripts are resumable.

## Data

- 114 items per variety, balanced 38 entailment / 38 neutral / 38 contradiction.
- CSV columns: `id`, `premise_text`, `premise_text_en`, `premise_audio`, `hypothesis_text`, `hypothesis_text_en`, `label`. The English glosses are the same in both files, and the two varieties are matched item by item (`wes_NNNNN` and `camf_NNNNN`, same number).
- Audio: 16 kHz, mono, 16-bit WAV, one speaker. Each clip starts 0.10 s before the speech and keeps a 1.0 s tail. Clips are 2.0 to 6.8 s long (mean 3.66 s for Pidgin and 3.40 s for Camfranglais; 417 s and 387 s in total).

## Settings used

**NLI (fine-tuning)**

| Setting    | Value                                                                                                                                                             |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model      | `xlm-roberta-base`, with a new 3-way classification head                                                                                                          |
| Input      | raw CSV text as stored (no lowercasing, no punctuation removal); premise and hypothesis as a sentence pair, truncated to 64 tokens                                |
| Conditions | native text, English gloss, premise-only, hypothesis-only                                                                                                         |
| Optimizer  | AdamW (`adamw_torch`), learning rate 2e-5, weight decay 0.0, linear learning-rate schedule                                                                        |
| Batch size | 4 for training, 8 for evaluation                                                                                                                                  |
| Precision  | mixed precision (fp16)                                                                                                                                            |
| Epochs     | 15 for the main results; 5, 10 and 20 for the epoch-sensitivity study                                                                                             |
| Selection  | none: no early stopping and no checkpoint selection; predictions come from the final epoch                                                                        |
| Seeds      | 42, 123, 2024 (three repeated runs)                                                                                                                               |
| Folds      | one shared stratified 5-fold split at item level (fold seed 42), copied from Pidgin to Camfranglais by item number                                                |
| Transfer   | train on the source-variety items outside fold k, test on the target-variety items in fold k (disjoint items); native text and hypothesis-only, both directions   |
| Runs       | 276: 120 within-variety, 60 transfer, 90 epoch sensitivity, 6 counterpart-leak audit                                                                              |
| Statistics | exact one-sided binomial test against chance (1/3); exact paired McNemar test, full input against hypothesis-only on the same test items, one per seed (12 tests) |

**ASR (Whisper, zero-shot)**

| Setting             | Value                                                                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Models              | Whisper Base, Small and Large-v3 (`openai-whisper` 20250625)                                                                                                     |
| Language conditions | automatic detection, forced English, forced French                                                                                                               |
| Decoding            | Whisper defaults (`task=transcribe`, fp16); `torch.manual_seed(0)` before every clip                                                                             |
| Reference           | the premise text in the CSV, unchanged                                                                                                                           |
| Scoring text        | tags `<\|...\|>` removed, Unicode NFC, lowercase, curly apostrophes made straight, other non-letters, digits and marks replaced by a space, spaces collapsed     |
| Metrics             | WER, CER, MER, WIL, WIP (corpus level); WER on the unnormalised text with `jiwer`; 95% bootstrap intervals for WER (the 114 clips resampled 2,000 times, seed 0) |
| Size                | 2,052 transcripts = 3 models x 3 conditions x 2 varieties x 114 clips                                                                                            |

## How to run

The scripts share variables (data frames, paths, seeds, helper functions), so they must run **in the same Colab session, in the order below**.

1. **Get access to the data on [Hugging Face](https://huggingface.co/datasets/lab-237/camspoken).** The dataset is gated: log in to a free Hugging Face account, open the dataset page, accept the terms (CC BY-NC 4.0, non-commercial research use) and send the access request. Once it is approved, open the **Files and versions** tab and go into the **`source/`** folder. Download `pidgin.csv`, `camfranglais.csv` and the `audio/` folder, which holds `pidgin/` and `camfranglais/` with 114 WAV files each. Use the files in `source/`, not the parquet folders (`pidgin_nli/`, `camfranglais_nli/`): those are for `load_dataset`, and the scripts check the original CSVs against SHA-256 hashes. You can compare the two CSVs with `SHA256SUMS.txt` from this repository; `00a` also checks them. Keep the file names exactly as downloaded, in lowercase (`pidgin.csv`, `camfranglais.csv`); the scripts look for these names.

2. **Prepare your Google Drive.** In My Drive, create a folder `CamSpokenNLI_run1` (this is `DRIVE_ROOT` in `00a_session_and_data.py`) with three folders inside it, and fill them like this:

   ```
   CamSpokenNLI_run1/
   ├─ final_csv/    pidgin.csv, camfranglais.csv                 (from Hugging Face)
   ├─ audio/        pidgin/ and camfranglais/ (WAV files)        (from Hugging Face)
   └─ results/      audio_preprocessing_log.csv                  (from results/ in this repository)
   ```

   Drive accepts whole folders by drag and drop. The scripts create every other result file themselves; `audio_preprocessing_log.csv` is the only one you must bring, because the audio checks read it.

3. **Open a Colab notebook with a T4 GPU** (Runtime, Change runtime type) and run the three setup scripts first, one cell each, in this order. Repeat them after every Colab reset:
   - `00a_session_and_data.py` installs the packages, mounts Drive, sets the paths, seeds and settings, and checks both CSVs against their stored SHA-256 hashes (it stops if they differ);
   - `00b_shared_folds.py` builds the shared folds (`shared_item_folds`, `environment_nli.json`);
   - `00c_nli_helpers_and_smoke_test.py` defines the NLI helper functions and runs a one-epoch smoke test that is not saved.
4. **Run the other scripts** in the order of the table below, each as a cell in the same session.

`00b` and `00c` stop with a clear message if an earlier setup script has not been run in the session. The NLI scripts (`02_*`) need only the CSVs; the ASR scripts and the audio checks also need the audio.

Outside Colab, replace the `drive.mount` line and set `DRIVE_ROOT` to your own folder. The raw recordings are not released; `code/provenance/00z_raw_audio_unpack.py` is needed only to rerun `01_A5_trim_audio.py` from raw files.

## Run order, and which results each script produces

Result names are files in `results/`; `results/README.md` says what is in each.

| Stage                     | Script                                                     | Result files                                                                                                                                                                                                                                                                                |
| ------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Setup                     | `00a_session_and_data.py`                                  | none (installs packages, mounts Drive, sets paths, seeds and settings, checks the CSV hashes)                                                                                                                                                                                               |
|                           | `00b_shared_folds.py`                                      | `shared_item_folds`, `environment_nli.json`                                                                                                                                                                                                                                                 |
|                           | `00c_nli_helpers_and_smoke_test.py`                        | none (defines the NLI helper functions; one-epoch smoke test, not saved)                                                                                                                                                                                                                    |
| Checks on the data        | `01_A6_audio_check.py`                                     | none (asserts that the audio matches the CSV ids and `audio_preprocessing_log`)                                                                                                                                                                                                             |
|                           | `01_A8_content_check.py`                                   | `content_check`, `content_check_transcripts`                                                                                                                                                                                                                                                |
|                           | `01_A9_listen_flagged.py`                                  | none (optional: plays the flagged clips)                                                                                                                                                                                                                                                    |
|                           | `03_H_dataset_and_audio_checks.py`                         | none (prints the dataset and audio statistics of Appendix H)                                                                                                                                                                                                                                |
| NLI runs                  | `02_C3_within_variety_cv.py`                               | `fold_results`, `item_predictions` (120 within-variety runs)                                                                                                                                                                                                                                |
|                           | `02_C4_transfer_disjoint.py`                               | `fold_results`, `item_predictions` (60 transfer runs)                                                                                                                                                                                                                                       |
|                           | `02_C5_counterpart_leak_audit.py`                          | `fold_results`, `item_predictions` (6 `transfer_old_counterpart_leak` runs)                                                                                                                                                                                                                 |
|                           | `02_C6_epoch_sensitivity.py`                               | `fold_results`, `item_predictions` (90 epoch-sensitivity runs)                                                                                                                                                                                                                              |
| NLI tables and statistics | `02_C7_nli_statistics.py`                                  | `table1_main_nli_matrix`, `baselines_within`, `baselines_hypothesis_only_transfer`, `appendix_b_per_seed_main_conditions`, `appendix_d_disjoint_transfer_per_seed`, `appendix_d_old_counterpart_leak`, `binomial_tests_all_seed_results`, `epoch_sensitivity`, `epoch_sensitivity_per_seed` |
|                           | `02_C8_mcnemar_full_vs_hypothesis_only.py`                 | `mcnemar_full_vs_hypothesis_only`                                                                                                                                                                                                                                                           |
|                           | `03_E_print_epoch_per_seed.py`, `03_G_fold_table_print.py` | none (print tables)                                                                                                                                                                                                                                                                         |
| ASR                       | `03_D_whisper_asr.py`                                      | `asr_transcripts_raw`, `appendix_c_asr_transcripts`, `asr_summary`, `table2_asr_wer_all_conditions`                                                                                                                                                                                         |
|                           | `03_D2_asr_checks.py`                                      | `asr_wer_bootstrap_ci`, `asr_script_and_empty_counts`, `asr_examples_large_v3`                                                                                                                                                                                                              |
|                           | `03_D3_raw_wer.py`                                         | `asr_raw_wer`                                                                                                                                                                                                                                                                               |
|                           | `03_D4_zero_error_and_leak_check.py`                       | none (prints the Large-v3 zero-error counts and the per-seed leak-audit values)                                                                                                                                                                                                             |
|                           | `03_C8_candidate_examples.py`                              | none (prints non-Latin and zero-error Large-v3 transcripts, the candidates for Appendix C.8)                                                                                                                                                                                                |
| Manifest                  | `04_E_manifest_and_backup.py`                              | `audio_checksums`, `run_manifest.json` (written to `results/`; kept at the top of this repository)                                                                                                                                                                                          |

Provenance, in `code/provenance/`. It starts from raw recordings and source files that are not released, so it cannot be rerun from this release:

| Script                                                                              | Purpose                                                                    | Result                                            |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------- |
| `01_A3a_repair_csv_commas.py`, `01_A3b_tidy_text.py`, `01_A3c_standardize_gloss.py` | repair, tidy and standardise the item CSVs                                 | the released `pidgin.csv` and `camfranglais.csv`  |
| `00z_raw_audio_unpack.py`                                                           | unpack the raw recordings for the next script (run after `00a`)            | none                                              |
| `01_A5_trim_audio.py`                                                               | convert to 16 kHz mono WAV, cut to 0.10 s before speech, keep a 1.0 s tail | `audio_preprocessing_log`, and the released audio |

## Notes

- **Counterpart-leak audit:** the `transfer_old_counterpart_leak` runs train on all 114 items of one variety and test on all 114 of the other, so a test item's counterpart is in the training set. They are an audit only and are not used in the main results. They are summarised in `appendix_d_old_counterpart_leak` (Appendix D.4 of the paper).
- **Raw WER:** `asr_raw_wer` scores the unnormalised transcripts (whitespace-split, `<|...|>` language tags removed from the hypotheses). `asr_summary` and `table2_asr_wer_all_conditions` use the normalised scoring described in Appendix C.
- **Runs:** every result file comes from the final data. The training scripts skip runs that are already in `fold_results`, so interrupted runs were resumed without repeating any, and `04_E` asserts that no run appears twice and that the counts are exact (276 NLI runs, 2,052 transcripts).
- **Checksums:** `SHA256SUMS.txt` (the two CSVs) and `results/audio_checksums.csv` (the 228 audio files) let you verify the data files. `run_manifest.json` holds the fold-file hash.
- **Extra files from `02_C7`:** the script also writes `per_seed`, `summary` and `baselines`. These repeat or combine the files listed above (`per_seed` is identical to `binomial_tests_all_seed_results`), so they are not included here.
- **Audio durations:** the longest Camfranglais clip is 5.555 s. The paper reports it as 5.55 s (`03_H`) and `run_manifest.json` as 5.56 s, because they round it differently.
- **Rerunning setup:** `00b_shared_folds.py` rewrites `results/environment_nli.json` with the current session's package versions. The versions used for the paper are stored under `recorded_at_session_start` in `run_manifest.json`.
- **Audio folder name:** during the original run the audio folder in Drive was called `processed_audio/`. The scripts here use `audio/`, the name used by the dataset. This changes nothing but the folder name. The `audio_path` column in `asr_transcripts_raw` and `appendix_c_asr_transcripts` still reads `processed_audio/...`, as written during the run; it is a text label and is never used to open files.
- **Code log:** each script ends with `save_code(...)`, which copies the cell's code into a `code` folder and updates `INDEX.csv` in your session. You can ignore it.

- The `# saved:` line at the top of a script is the time it was last saved in Colab. Scripts dated after 5 October were edited after the results were produced (setup split, listening tool, manifest); they do not change any reported number.

## Licences

- Data and audio: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (attribution, non-commercial use only)
- Code: [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) (see `LICENSE`; non-commercial use only)

All materials are intended for research use. They may not be used for commercial purposes. This is not an open-source licence in the OSI sense, because it excludes commercial use.

## Citation

```bibtex
@misc{ebika2026camspokennli,
  title        = {CamSpokenNLI: A Pilot Spoken Natural Language Inference Resource for Cameroon Pidgin English and Camfranglais},
  author       = {Ebika, Tiani Pekins},
  year         = {2026},
  publisher    = {LAB 237 -- Open AI Research Lab},
  howpublished = {Hugging Face Datasets},
  url          = {https://huggingface.co/datasets/lab-237/camspoken}
}
```
