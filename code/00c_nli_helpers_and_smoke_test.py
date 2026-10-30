# 00c_nli_helpers_and_smoke_test
# AIM: Session start (step 3 of 3): NLI helpers and a 1-epoch smoke test
# saved: 2026-10-24 13:40

# 00c_nli_helpers_and_smoke_test
# AIM: Session start (step 3 of 3): NLI helpers (tokenizer, dataset, model, resumable result files, train/eval runner) and a 1-epoch smoke test
# Run after 00a and 00b, in the same session (about 1-2 min).
# Wait for: 'smoke test ran end to end'.

# Guard: stops with a clear message if 00a or 00b has not been run in this session.
assert "fold_table" in globals(), "Run 00a_session_and_data.py and 00b_shared_folds.py first, in the same session."

# C1+C2: NLI helpers (tokenizer, dataset, model, resumable result files, one train/eval runner) + a 1-epoch smoke test.
# Nothing is saved to results; the smoke test only proves the library versions work together.
# Raw CSV text is used exactly as stored (no lowercasing, no punctuation removal).

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)



def get_text_columns(df, condition):
    if condition == "native_full":
        return df["premise_text"].tolist(), df["hypothesis_text"].tolist()
    if condition == "english_gloss":
        return df["premise_text_en"].tolist(), df["hypothesis_text_en"].tolist()
    if condition == "premise_only":
        return df["premise_text"].tolist(), None
    if condition == "hypothesis_only":
        return df["hypothesis_text"].tolist(), None
    raise ValueError(f"Unknown condition: {condition}")


def make_tokenized_dataset(df, condition):
    first, second = get_text_columns(df, condition)
    if second is None:
        enc = tokenizer(first, truncation=True, max_length=MODEL_MAX_LENGTH)
    else:
        enc = tokenizer(first, second, truncation=True, max_length=MODEL_MAX_LENGTH)
    enc["labels"] = [LABEL2ID[x] for x in df["label"].tolist()]
    return Dataset.from_dict(enc)


def load_model_for_seed(seed):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model


def _training_args_kwargs(output_dir, seed, epochs):
    kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=epochs,
        weight_decay=WEIGHT_DECAY,
        optim=OPTIMIZER,
        lr_scheduler_type=LR_SCHEDULER,
        seed=seed,
        data_seed=seed,
        fp16=True,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    # Transformers renamed evaluation_strategy -> eval_strategy.
    params = set(__import__("inspect").signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "no"
    return kwargs


FOLD_RESULTS_FILE = RESULTS_DIR / "fold_results.csv"
PREDICTIONS_FILE = RESULTS_DIR / "item_predictions.csv"
FAILURES_FILE = RESULTS_DIR / "nli_failures.csv"

RESULT_COLUMNS = [
    "experiment", "split_type", "direction", "train_variety", "test_variety",
    "condition", "epochs", "seed", "fold", "correct", "n_test",
    "accuracy", "fold_mean_component", "status", "seconds", "counterpart_overlap"
]


def read_csv_or_empty(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[columns]


def result_key(row):
    return "|".join(str(row.get(k, "")) for k in [
        "experiment", "split_type", "direction", "train_variety", "test_variety",
        "condition", "epochs", "seed", "fold"
    ])


def existing_result_keys():
    df = read_csv_or_empty(FOLD_RESULTS_FILE, RESULT_COLUMNS)
    if df.empty:
        return set()
    return set(df.apply(result_key, axis=1).tolist())


def append_row(path, row, columns):
    df = read_csv_or_empty(path, columns)
    new = pd.DataFrame([row])
    for c in columns:
        if c not in new:
            new[c] = np.nan
    df = pd.concat([df, new[columns]], ignore_index=True)
    df.to_csv(path, index=False)


def append_prediction_rows(rows):
    if not rows:
        return
    cols = [
        "experiment", "split_type", "direction", "train_variety", "test_variety",
        "condition", "epochs", "seed", "fold", "id", "true_label", "pred_label",
        "correct", "p_entailment", "p_contradiction", "p_neutral"
    ]
    old = pd.read_csv(PREDICTIONS_FILE) if PREDICTIONS_FILE.exists() else pd.DataFrame(columns=cols)
    new = pd.DataFrame(rows)
    for c in cols:
        if c not in new:
            new[c] = np.nan
    pd.concat([old, new[cols]], ignore_index=True).to_csv(PREDICTIONS_FILE, index=False)


def run_training_once(train_df, test_df, condition, seed, epochs, run_tag):
    # The only stochastic controls are the requested seed, model init, and trainer data seed.
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = load_model_for_seed(seed)
    train_ds = make_tokenized_dataset(train_df.reset_index(drop=True), condition)
    test_ds = make_tokenized_dataset(test_df.reset_index(drop=True), condition)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    out_dir = TMP_TRAIN_DIR / run_tag
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(**_training_args_kwargs(out_dir, seed, epochs))
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer_params = set(__import__("inspect").signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    pred = trainer.predict(test_ds)

    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = np.argmax(logits, axis=-1)
    true_ids = np.asarray([LABEL2ID[x] for x in test_df["label"].tolist()])
    correct_mask = pred_ids == true_ids
    correct = int(correct_mask.sum())
    n_test = int(len(test_df))
    accuracy = correct / n_test

    pred_rows = []
    for i, (_, r) in enumerate(test_df.reset_index(drop=True).iterrows()):
        pred_rows.append({
            "id": r["id"],
            "true_label": ID2LABEL[int(true_ids[i])],
            "pred_label": ID2LABEL[int(pred_ids[i])],
            "correct": int(correct_mask[i]),
            "p_entailment": float(probs[i, 0]),
            "p_contradiction": float(probs[i, 1]),
            "p_neutral": float(probs[i, 2]),
        })

    del trainer, model, train_ds, test_ds, pred, logits, probs
    gc.collect()
    torch.cuda.empty_cache()
    return correct, n_test, accuracy, pred_rows


def run_training_with_retry(train_df, test_df, condition, seed, epochs, run_tag):
    errors = []
    for attempt in [1, 2]:
        try:
            return run_training_once(train_df, test_df, condition, seed, epochs, run_tag)
        except Exception as exc:
            errors.append(repr(exc))
            print(f"⚠️ NLI run failed on attempt {attempt}: {exc}")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
    raise RuntimeError(f"NLI run failed twice and was NOT dropped. Errors: {errors}")


# ---- smoke test: 1 epoch, 24 train / 12 test items, NOT saved ----
_c, _n, _a, _ = run_training_once(pidgin_df.iloc[:24].copy(), pidgin_df.iloc[24:36].copy(), "native_full", 42, 1, "smoke")
shutil.rmtree(TMP_TRAIN_DIR / "smoke", ignore_errors=True)
print(f"✅ smoke test ran end to end ({_c}/{_n} correct after 1 epoch; accuracy here is meaningless).")
print("Result files that will be written:", FOLD_RESULTS_FILE.name, PREDICTIONS_FILE.name)

save_code("00c_nli_helpers_and_smoke_test", "Session start (step 3 of 3): NLI helpers and a 1-epoch smoke test")