import torch
from transformers import pipeline


pipe = pipeline(task="translation", model="facebook/nllb-200-distilled-600M", src_lang= "grn_Latn", tgt_lang="fra_Latn", dtype=torch.float16, device=0)
print(pipe("Omopotîvo hikuái tetãme vicio política, ko'ã itaugüeño he'íva ombotovévo pokarême umi elemento omopotîva."))