# script pour exporter les splits en jsonl pour entraînement MT

import pandas as pd

df = pd.read_csv("../data/jojajovai_all_clean.csv")
df = df[["split", "source", "gn", "es"]].copy()

def export_split(name: str):
    out = df[df["split"] == name].copy()
    out = out.rename(columns={"gn": "translation_source", "es": "translation_target"})
    out.to_json(f"../data/jojajovai_{name}.jsonl", orient="records", lines=True, force_ascii=False)
    print(name, "->", len(out))

export_split("train")
export_split("dev")
export_split("test")
