# Projet : Traduction automatique Guarani → Espagnol

## 1. Données utilisées

### 1.1 Corpus principal : Jojajovai (Guarani ↔ Espagnol)

Nous utilisons le corpus parallèle **Jojajovai**, composé d’environ **30 000 paires de phrases alignées** guarani–espagnol :

🔗 [https://github.com/pln-fing-udelar/jojajovai/tree/main/data](https://github.com/pln-fing-udelar/jojajovai/tree/main/data)

> Luis Chiruzzo, Santiago Góngora, Aldo Alvarez, Gustavo Giménez-Lugo, Marvin Agüero-Torales, Yliana Rodríguez. (2022). *Jojajovai: A Parallel Guarani-Spanish Corpus for MT Benchmarking.* Proceedings of the 13th Language Resources and Evaluation Conference, LREC 2022.

### 1.2 Composition du corpus

Les textes du corpus Jojajovai proviennent majoritairement de :

* journaux et articles contemporains,
* mythes et légendes,
* contenus culturels ou éducatifs.

### 1.3 Nettoyage des données

Une étape de *data cleaning* est nécessaire pour :

* retirer les dialectes ou variétés qui ne relèvent pas du guarani,
* homogénéiser la langue source,
* éliminer les exemples bruités ou non alignés.

---

## 2. Modèles de traduction explorés

Notre objectif est de comparer deux modèles de traduction multilingue adaptés aux langues peu dotées : **M2M100** et **NLLB-200**.

### 2.1 M2M100 (Meta) — Fine-tuning prévu

M2M100 est un modèle couvrant plus de 100 langues et capable de traduire directement entre n’importe quelle paire de langues, sans pivot par l’anglais.

Nous prévoyons de fine-tuner M2M100 sur nos données guarani–espagnol afin de spécialiser le modèle pour notre tâche.

### 2.2 NLLB-200 (Meta) — Test direct sans fine-tuning

NLLB-200 est conçu pour les langues peu dotées, dont le guarani (grn_Latn).
Le vocabulaire et la représentation du guarani sont déjà inclus dans son pré-entraînement.

Plan d’expérimentation :

* Tester NLLB-200 directement, sans fine-tuning, sur nos données.
* Comparer ses performances avec le M2M100 fine-tuné.

---

## 3. Evaluation prévue

Nous utiliserons la métrique semi-automatique de traduction automatique BLEU et les scores seront calculés sur un jeu de test séparé issu du corpus nettoyé.

---

## 4. Premiers tests effectués

### 4.1 Vérification du modèle NLLB-200

Pour s'assurer du bon fonctionnement du modèle sur nos machines (CPU uniquement), nous avons d’abord reproduit l’exemple HuggingFace :

* Traduction **anglais → français** (codes `eng_Latn` → `fra_Latn`)
* Résultat conforme aux attentes.

### 4.2 Premier test Guarani → Français

Nous avons ensuite choisi une phrase au hasard dans le corpus Jojajovai et testé la traduction **guarani → français** (code `grn_Latn`).

**Phrase source (guarani) :**
« *Omopotîvo hikuái tetãme vicio política, ko'ã itaugüeño he'íva ombotovévo pokarême umi elemento omopotîva.* »

**Sortie NLLB-200 :**
« *Lorsqu'ils ont nettoyé le pays des vices politiques, ces sculpteurs ont déclaré qu'ils rejetaient la pureté des éléments qui le nettoyaient.* »

### 4.3 Observations

* La traduction est grammaticalement correcte, mais le sens semble décalé.
* Cela suggère soit une ambiguïté sémantique, soit une interprétation incorrecte due à la complexité morphologique du guarani.
* Des locuteurs natifs vérifieront cette première impression.
