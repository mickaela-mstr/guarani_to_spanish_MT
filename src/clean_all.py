# script de nettoyage du dataset jojajovai_all.csv

import re
import unicodedata
import pandas as pd


IN_PATH = "../data/jojajovai_all.csv"
OUT_PATH = "../data/jojajovai_all_clean.csv"


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s



def parse_tokens(x: str) -> list[str]:
    """
    tokens_gn / tokens_es semblent être des strings déjà prêtes.
    On fait une tokenisation simple robuste (mots) au cas où.
    """
    x = normalize_text(x).lower()
    return re.findall(r"\w+", x, flags=re.UNICODE)


def jaccard(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def main():
    df = pd.read_csv(IN_PATH)
    for col in ["gn", "es"]:
        df[col] = df[col].map(normalize_text) #on normalise les textes

    df = df[(df["gn"] != "") & (df["es"] != "")].copy() # on enlève les paires vides

    # on utilise les tokens du fichier si présents, sinon on re-tokenise
    df["tok_gn"] = df["gn"].map(parse_tokens)
    df["tok_es"] = df["es"].map(parse_tokens)

    # on filtre les phrases trop courtes qui font moins de 2 tokens
    df = df[(df["tok_gn"].map(len) >= 2) & (df["tok_es"].map(len) >= 2)].copy()

    # on filtre les phrases trop longues (>100 tokens)
    df["len_gn_tok"] = df["tok_gn"].map(len)
    df["len_es_tok"] = df["tok_es"].map(len)
    df["len_ratio"] = df[["len_gn_tok", "len_es_tok"]].max(axis=1) / df[["len_gn_tok", "len_es_tok"]].min(axis=1)

    # si une phrase est >3x plus longue que l'autre on jette
    df = df[df["len_ratio"] <= 3.0].copy()

    # si le jaccard entre les tokens est trop élevé on jette (risque de phrases identiques)
    df["jaccard_gn_es"] = [jaccard(a, b) for a, b in zip(df["tok_gn"], df["tok_es"])]
    df = df[df["jaccard_gn_es"] < 0.60].copy()
    keep_cols = ["split", "source", "gn", "es"]
    df_out = df[keep_cols].copy()

    print("Paires finales:", len(df_out))
    print("Répartition split:", df_out["split"].value_counts().to_dict())
    print("Top sources:", df_out["source"].value_counts().head(10).to_dict())

    df_out.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
