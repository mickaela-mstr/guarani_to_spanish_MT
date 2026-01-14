import json
import sacrebleu
from pathlib import Path

ROOT = Path("data")
TEST_PATH = ROOT /"jojajovai_test.jsonl"

NLLB_PRED = ROOT / "nllb_oob_predictions.txt"
M2M_PRED  = ROOT / "m2m100_ft_predictions.txt"
NLLB_FT_PRED = ROOT / "nllb_ft_predictions.txt"

def load_refs(test_path: Path):
    refs = []
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            refs.append(ex["translation_target"])
    return refs

def load_hyps(path: Path):
    return path.read_text(encoding="utf-8").splitlines()

def score(name: str, hyps, refs):
    assert len(hyps) == len(refs), f"{name}: nb hyps != nb refs ({len(hyps)} vs {len(refs)})"

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    ter  = sacrebleu.corpus_ter(hyps, [refs]).score

    print(f"BLEU : {bleu:.4f}")
    print(f"chrF : {chrf:.4f}")
    print(f"TER  : {ter:.4f}")

def main():
    refs = load_refs(TEST_PATH)
    score("NLLB (OOB)", load_hyps(NLLB_PRED), refs)
    score("M2M100 (fine-tuned)", load_hyps(M2M_PRED), refs)
    score("NLLB (fine-tuned)", load_hyps(NLLB_FT_PRED), refs)

if __name__ == "__main__":
    main()