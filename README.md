# Mini-projet : Traduction automatique du guarani vers l’espagnol

Ce projet est réalisé dans un cadre universitaire en Master 2 traitement automatique des langues.
Il porte sur la traduction automatique du guarani vers l’espagnol et la comparaison de deux modèles multilingues de traduction neuronale sur un même corpus parallèle.

L’objectif principal est d’étudier la capacité d’un modèle multilingue généraliste à apprendre une langue absente de son pré-entraînement via un fine-tuning supervisé, et de comparer ses performances à celles d’un modèle spécialisé censé déjà prendre en charge cette langue.

---

## Objectifs du projet

- Préparer et nettoyer un corpus parallèle guarani–espagnol
- Fine-tuner un modèle de traduction multilingue
- Évaluer un modèle spécialisé en mode *out-of-the-box*
- Comparer les performances des deux approches
- Analyser les résultats à l’aide de métriques automatiques de traduction

---

## Données

Le corpus utilisé est **Jojajovai**, un corpus parallèle guarani–espagnol aligné par phrase, contenant environ 30 000 paires de phrases.
🔗 [https://github.com/pln-fing-udelar/jojajovai/tree/main/data](https://github.com/pln-fing-udelar/jojajovai/tree/main/data)

> Luis Chiruzzo, Santiago Góngora, Aldo Alvarez, Gustavo Giménez-Lugo, Marvin Agüero-Torales, Yliana Rodríguez. (2022). *Jojajovai: A Parallel Guarani-Spanish Corpus for MT Benchmarking.* Proceedings of the 13th Language Resources and Evaluation Conference, LREC 2022.

Le corpus est déjà divisé en trois partitions :

* train
* test
* dev


Un sous-ensemble du jeu de test est annoté manuellement par des locuteurs natifs du guarani. Ces annotations indiquent le dialecte présent dans la phrase ainsi que la qualité de l’alignement entre la phrase source et la traduction.

---

## Modèles utilisés

### M2M100

M2M100 est un modèle multilingue de traduction basé sur une architecture Transformer encodeur–décodeur. Il permet la traduction directe entre plus de cent langues sans passer par une langue pivot comme l’anglais.

Le guarani n’étant pas pris en charge nativement par ce modèle, une stratégie d’alias de langue est utilisée. Une étiquette de langue existante est employée comme substitut pour représenter le guarani lors du fine-tuning. Le modèle est ensuite affiné de manière supervisée sur le corpus Jojajovai.

### NLLB-200

NLLB-200 est un modèle multilingue spécialisé dans la traduction de langues peu dotées. Il annonce un support natif pour le guarani et l’espagnol, intégrés dans son vocabulaire et ses données de pré-entraînement.

Dans ce projet, NLLB-200 est utilisé sans affinement supplémentaire afin d’évaluer ses performances *out-of-the-box* sur la traduction guarani–espagnol.

---

## Évaluation

Les traductions produites par les modèles sont évaluées sur le jeu de test à l’aide de métriques automatiques :

- **BLEU** : mesure de similarité basée sur les n-grammes de mots
- **chrF** : mesure de similarité au niveau des caractères
- **TER** : taux d’édition nécessaire pour transformer une traduction en référence

Ces métriques permettent d’analyser à la fois la proximité lexicale, la robustesse morphologique et le coût de post-édition des traductions.

---

## Résultats principaux

Les expériences montrent que le modèle **M2M100 fine-tuné** obtient de meilleures performances que **NLLB-200 utilisé sans adaptation**, notamment en termes de BLEU et de TER.

Le score chrF reste similaire entre les deux modèles, ce qui indique que les deux systèmes produisent déjà des formes espagnoles correctes sur le plan orthographique. Les gains observés concernent principalement l’alignement traductionnel et le choix lexical.

Ces résultats suggèrent qu’une adaptation supervisée ciblée peut compenser, au moins en partie, l’absence de support natif d’une langue dans un modèle multilingue.

---

## Arborescence du projet

```text
guarani_to_spanish_MT/
├── data/
│   ├── csv/
│   │   ├── jojajovai_all.csv
│   │   ├── jojajovai_all_clean.csv
│   │   └── jojajovai_sample_annotations.csv
│   └── jsonl/
│       ├── jojajovai_train.jsonl
│       ├── jojajovai_dev.jsonl
│       └── jojajovai_test.jsonl
│
├── resultats/
│   ├── m2m100_ft_predictions.txt
│   └── nllb_oob_predictions.txt
│
├── src/
│   ├── nettoyage/
│   │   ├── clean_all.py
│   │   ├── clean_annot.py
│   │   └── export_jsonl.py
│   │
│   ├── finetuning/
│   │   └── train_m2m100.py
│   │
│   ├── eval/
│   │   └── eval_nllb.py
│   │
│   └── metrics/
│       └── metrics.py
│
├── README.md

```
