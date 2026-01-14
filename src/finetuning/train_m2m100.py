import json
import torch
import sacrebleu
from pathlib import Path
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)


TRAIN_PATH = Path("data/jojajovai_train.jsonl")
DEV_PATH = Path("data/jojajovai_dev.jsonl")
TEST_PATH = Path("data/jojajovai_test.jsonl")

MODEL_NAME = "facebook/m2m100_418M"
SRC_LANG = "en" # on ment volontairement au modèle sur l’étiquette de langue source.
# l'étiquette en de l'anaglais est utilisé comme alias pour le guarani qui n'est pas supporté nativement   
TGT_LANG = "es"

MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 128


def load_jsonl_as_dataset(path: Path) -> Dataset:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            rows.append({"src": ex["translation_source"], "tgt": ex["translation_target"]})
    return Dataset.from_list(rows)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = load_jsonl_as_dataset(TRAIN_PATH)
    dev_ds = load_jsonl_as_dataset(DEV_PATH)
    test_ds = load_jsonl_as_dataset(TEST_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG

    model.config.forced_bos_token_id = tokenizer.get_lang_id(TGT_LANG)

    def preprocess(examples):
        inputs = tokenizer(
            examples["src"],
            max_length=MAX_SOURCE_LEN,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])
    dev_tok = dev_ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir="models/m2m100-ft",
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        generation_max_length=128,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[(t if t != -100 else tokenizer.pad_token_id) for t in seq] for seq in labels]
        refs_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = sacrebleu.corpus_bleu(preds_text, [refs_text])
        return {"bleu": bleu.score}

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.eval()
    srcs = test_ds["src"]
    refs = test_ds["tgt"]

    hyps = []
    forced_bos_token_id = tokenizer.get_lang_id(TGT_LANG)

    for i in range(0, len(srcs), 4):
        batch = srcs[i:i+4]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=128)
        hyps.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print("M2M100 fine-tuned BLEU (test):", bleu.score)

    out_path = Path("data/m2m100_ft_predictions.txt")
    out_path.write_text("\n".join(hyps), encoding="utf-8")


if __name__ == "__main__":
    main()
