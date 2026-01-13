import json
import torch
import sacrebleu
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

TEST_PATH = Path("data/jojajovai_test.jsonl")
MODEL_NAME = "facebook/nllb-200-distilled-600M"

SRC_LANG = "grn_Latn"
TGT_LANG = "spa_Latn"

MAX_NEW_TOKENS = 128
BATCH_SIZE = 2  

def load_jsonl(path: Path):
    srcs, refs = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            srcs.append(ex["translation_source"])
            refs.append(ex["translation_target"])
    return srcs, refs


@torch.inference_mode()
def translate_batch(model, tokenizer, texts, device):
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    srcs, refs = load_jsonl(TEST_PATH)

    hyps = []
    for i in range(0, len(srcs), BATCH_SIZE):
        batch = srcs[i:i+BATCH_SIZE]
        hyps.extend(translate_batch(model, tokenizer, batch, device))
        if i % (BATCH_SIZE * 100) == 0:
            print(f"Progress: {i}/{len(srcs)}")


    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print("NLLB OOB BLEU:", bleu.score)

    out_path = Path("data/nllb_oob_predictions.txt")
    out_path.write_text("\n".join(hyps), encoding="utf-8")

if __name__ == "__main__":
    main()
