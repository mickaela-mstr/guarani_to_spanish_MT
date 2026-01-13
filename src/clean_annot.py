import pandas as pd

df = pd.read_csv("jojajovai_sample_annotations.csv")

clean_annot = df[
    (df["dialect_unified"].str.lower() == "guarani") &
    (df["correctness_unified"].isin(["A", "B", "C"]))
].copy()

print("Total annoté:", len(df))
print("Gardé après nettoyage:", len(clean_annot))
print(clean_annot["correctness_unified"].value_counts())
 
