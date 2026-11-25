# PLAN_FEATURES.md — Plan d'implémentation : features, baselines et AutoML (Lot B)

> Plan du lot B (epic Jira KAN-5) : transformer les vues causales du lot A en premiers détecteurs mesurés honnêtement, tester H1 et H3, livrer v0.0, v0.1 et v0.2.
> Références : [CONTEXT.md](CONTEXT.md) §3.3–3.4, §4, §7, §8 ; [JOURNAL.md](JOURNAL.md) ; [INSTRUCTIONS.md](INSTRUCTIONS.md) Step 2.3 et 2.5 ; [PLAN_DONNEES.md](PLAN_DONNEES.md) (lot A).
> Version : 25/11/2026 (rédaction initiale, reprise du lot B). Toute modification du protocole est notée dans JOURNAL.md.

---

## Vue d'ensemble

| # | Étape | Ticket | Statut |
|---|---|---|---|
| — | Vues scanner / portail causales | KAN-16 (B.1) | ✅ fait (20–21/11) |
| — | Vues sur texte anonymisé + variantes `masked` / `raw` | KAN-60 (B.1b) | ✅ fait (23/11) |
| — | Règle de masquage H3 + EDA B5 | KAN-59 (B.4a) | ✅ fait (23/11) |
| — | Harnais d'expérience + run v0.0 majoritaire | KAN-61 (B.0) | ✅ fait (23/11) |
| **0** | **Extensions du harnais** (vue passée à la fabrique, seuils fournis par l'estimateur, cache, `run_baseline`) | KAN-61 (suite) | ✅ fait (25/11) |
| **1** | **Bloc texte** `features/text.py` | KAN-17 (B.2) | ✅ fait (25/11) |
| **2** | **Baselines v0.0 (repères, raccourcis) et v0.1 (TF-IDF + LR calibrée)** | KAN-20 (B.5) | à faire |
| **3** | **Blocs structure, nom d'outil, provenance, indices d'injection** | KAN-18 (B.3) | à faire |
| **4** | **Test de H1** (texte / structure / les deux, CV et domaine exclu) | KAN-62 (B.7) | à faire |
| **5** | **AutoML v0.2** (CV imbriquée, Optuna) | KAN-21 (B.6) | à faire |
| **6** | **Test de H3** (échelle d'ablation) | KAN-19 (B.4) | à faire |
| ∥ | Sections B du rapport, gouvernance, `CONTRIBUTIONS.md` | KAN-41, KAN-49 | au fil de l'eau |

**Ordre :** 0 → 1 → 2 → 3 → 4 → 5 → 6, tel que fixé dans KAN-5. Deux exceptions :
- L'étape 3 peut démarrer pendant l'étape 2, car elle ne dépend que des vues.
- L'étape 6 peut tourner dès la fin de l'étape 2 sur v0.1, puis être relancée avec v0.2.

**Fini quand (KAN-5) :**
- v0.0, v0.1 et v0.2 sont évaluées sur les deux vues, avec IC.
- H1 et H3 sont testées.
- Les résultats sont dans `results/runs/` et les sections B du rapport sont rédigées.
- `CONTRIBUTIONS.md` est à jour.

---

## État des lieux (25/11)

### Ce qui existe et sur quoi tout le reste s'appuie

- **Vues** (`features/views.py`, `pipelines/build_views.py`)
  - Volumes : scanner 58 488 lignes (12,0 % de positifs), portail 71 024 lignes (17,4 %). En développement : 50 414 et 61 228 lignes.
  - Colonnes de variables, toutes textuelles :
    - scanner : `task`, `history_text`, `content_source_tool`, `content_source_args`, `current_content` ;
    - portail : `task`, `history_text`, `last_content`, `proposed_call_tool`, `proposed_call_args`.
  - Historique balisé ligne par ligne (`<thought>`, `<call> outil {args}`, `<result>`). Les espaces réservés de l'anonymisation y sont conservés.
  - Chaque fichier porte sa variante de texte. `read_view` refuse une variante absente ou différente.
- **Masquage** (`features/masking.py`) : `<HEADER>`, `<CODE>`, `<AMOUNT_k>`.
- **Harnais** (`models/experiment.py`, `pipelines/run_experiment.py`)
  - Une version = un YAML + une fabrique `module:fonction`. La fabrique est appelée à neuf par pli.
  - Une fabrique qui déclare `inner_cv` reçoit les splits internes groupés par gabarit.
  - Les seuils sont fixés sur les scores hors pli internes (`inner_oof`).
  - `check_columns` refuse toute colonne interdite. La position n'est permise qu'à un modèle `kind: shortcut`.
  - Chiffre principal : moyenne par pli, IC par bootstrap de gabarits stratifié par pli, apparié entre modèles.
  - Sorties : `oof.parquet` et `metrics.json`, avec rappel par type d'attaque et taux d'alerte par classe négative.
- **Bootstrap** (`evaluation/bootstrap.py`), **plis figés** (`folds.parquet`, empreinte `f69d2d09f08332a9`), **annotations** (`data/annotations.attach_tool_risk`, table `tool_risk.yaml` en brouillon).
- **Tests :** 177 passent, 1 est ignoré tant que les vues ne sont pas construites localement.

### Ce qui manque

- `features/text.py`, `pipelines/run_baseline.py` et `pipelines/run_audit.py` ne contiennent qu'une docstring.
- `features/structural.py`, `features/provenance.py`, `models/automl.py` et `pipelines/run_automl.py` n'existent pas.
- `models/baselines.py` ne contient que `majority`.
- `evaluation/metrics.py` et `decision/calibration.py` sont vides. Ils relèvent du lot D, mais la calibration de v0.1 a besoin de Brier et de l'ECE.

### Mesures exploratoires du 25/11 (à refaire dans le harnais)

Conditions :
- TF-IDF mots 1–2 grammes par champ (`min_df=2`, `sublinear_tf`, espaces réservés gardés comme jetons) + `LogisticRegression(class_weight="balanced", C=1)`.
- Pli `cv0` seul pour la CV ; 5 domaines pour le domaine exclu.
- Seuil fixé **sur les scores d'entraînement** : c'est la variante optimiste.

Script hors dépôt. Ces chiffres orientent le plan et ne sont pas des résultats.

| Mesure | Scanner | Portail |
|---|---|---|
| Temps d'ajustement par pli (mots / caractères 3–5) | 2,2 s / 11,9 s | 2,6 s / 14,5 s |
| Dimension (mots / caractères) | 111 k / 169 k | 124 k / 183 k |
| PR-AUC `cv0`, texte anonymisé | **0,969** | **0,975** |
| PR-AUC `cv0`, texte masqué | 0,970 | 0,974 |
| PR-AUC domaine exclu (banking … web) | 0,909–0,962 | **0,863**–0,965 |
| FPR observé pour une cible de 1 %, seuil fixé sur l'entraînement (CV) | 2,0 % | 2,9–3,0 % |
| PR-AUC avec un seul champ | `current_content` 0,943 ; `history_text` 0,613 ; `task` 0,128 | `history_text` **0,947** ; `last_content` 0,886 ; `proposed_call_args` **0,664** ; `proposed_call_tool` 0,485 ; `task` 0,181 |

**Ce que ces chiffres changent au plan**

1. **La baseline sature presque la distribution d'entraînement.** v0.2 ne pourra pas faire beaucoup mieux que v0.1 en CV : les écarts tiendront dans les IC. Les comparaisons qui départagent les modèles sont :
   - le domaine exclu ;
   - le rappel à 0,2 % de FPR ;
   - le rappel sur les attaques partielles et retardées ;
   - le portail limité à l'action (point 3) ;
   - plus tard, ATBench (D.6).

   Le rapport doit le dire, plutôt que de présenter un gain de 0,005 de PR-AUC comme un progrès.
2. **Le masquage ne change rien à v0.1** (0,969 contre 0,970). Cela confirme la sonde du type d'attaque de l'EDA B5 : la fonction de l'injection reste lisible. Le test de H3 ne peut donc pas se limiter aux marches 3 → 5. Il faut des ablations par champ et la variante « action seule » (étape 6).
3. **Le portail lit surtout l'injection déjà présente dans l'historique**, pas l'action proposée. `history_text` seul donne 0,947 ; `proposed_call_*` seul donne 0,49–0,66. Tel quel, le portail est en partie un second scanner. Sa vraie valeur (détecter le détournement quand l'injection est invisible, comme dans les attaques partielles ou ATBench) se mesure sur une **variante « action seule »** : `task`, `proposed_call_tool`, `proposed_call_args`. Le harnais le permet déjà par la clé `columns:`.
4. **Un seuil fixé sur les scores d'entraînement donne 2 à 3 fois le FPR cible.** Cela confirme `threshold_source: inner_oof` (JOURNAL 23/11). Il faut le documenter avec les runs du harnais.
5. **Le coût de calcul est faible pour le texte par mots et dix fois plus élevé pour les caractères.** Le budget de l'AutoML doit en tenir compte (étape 0).

---

## Étape 0 — Extensions du harnais (KAN-61, suite) ✅ fait le 25/11

Implémenté dans `models/experiment.py` (`view`, `memory`, `threshold_source: estimator`, `paired_difference`) et `pipelines/run_baseline.py` ; 13 tests ajoutés (190 verts) ; essai de bout en bout noté dans le JOURNAL du 25/11. Le plan ci-dessous reste la spécification.

But : que les étapes 1 à 6 se ramènent à « une fabrique + un YAML », sans contourner le harnais.

1. **Vue passée à la fabrique.** Une fabrique qui déclare un argument `view` le reçoit, sur le modèle de `inner_cv`. Sans cela, un bloc texte ne sait pas s'il doit lire `current_content` ou `last_content`.
2. **Seuils fournis par l'estimateur** (`threshold_source: estimator`).
   - Après `fit`, un estimateur qui expose `inner_oof_scores_` fournit lui-même ses scores hors pli internes. Le harnais ne refait alors pas trois ajustements.
   - Indispensable pour l'AutoML : avec `inner_oof`, chaque essai Optuna serait relancé dans 3 × 3 sous-plis.
   - Biais à documenter : ces scores sont ceux de la configuration retenue sur ces mêmes plis, donc légèrement optimistes. Il faut le comparer une fois à `inner_oof` sur v0.1.
3. **Cache des vectorisations** (`Pipeline(memory=…)` dans un dossier temporaire par run). Le TF-IDF par caractères domine le temps d'ajustement, et la recherche AutoML le recalcule à chaque essai avec les mêmes paramètres.
4. **`pipelines/run_baseline.py`** : lance une liste de configs (`configs/models/v0_0_*.yaml`, `v0_1_*.yaml`) en CV et en domaine exclu, puis écrit un tableau comparatif. Le tableau donne la PR-AUC, le ROC-AUC et le rappel à FPR 1 % et 0,2 %, avec IC, ainsi que les différences appariées avec la référence. Il est écrit dans `results/runs/comparisons/v0_x.csv`.
5. **Différence appariée entre deux runs** : une fonction qui relit deux `oof.parquet` et calcule l'IC de la différence avec les mêmes tirages (`split_resamples` est déjà apparié). À coordonner avec D.4 (KAN-31, Oscar) pour ne pas écrire deux fois le même test.

**Tests :**
- une fabrique avec `view` reçoit la bonne vue ;
- `estimator` refuse un estimateur sans `inner_oof_scores_` ;
- les scores internes couvrent toutes les lignes d'entraînement et aucune ligne évaluée ;
- la différence appariée d'un modèle avec lui-même vaut 0 avec un IC [0 ; 0].

## Étape 1 — Bloc texte (KAN-17, B.2) ✅ fait le 25/11

Implémenté : `features/text.py::TextBlock` et `text_block`, fabrique `models/baselines.py::tfidf_logreg`, 25 tests. Différence avec la spécification ci-dessous : les champs sont pris au `fit` parmi les colonnes transmises par le harnais (pas de liste `fields` à répéter dans le YAML). Dimensions, temps et premier run noté dans le JOURNAL du 25/11, dont le portail « action seule » (PR-AUC 0,744, rappel@1 % 0,058 sur les partielles).

Fichier `features/text.py`, fabrique de transformeur `text_block(view, fields=None, word=…, char=…, weights=None, svd=None)`.

- Un `ColumnTransformer` avec un vectoriseur par champ de la vue. Le même mot n'a pas le même sens dans la tâche, l'historique et l'action. La pondération des champs est passée par `transformer_weights`.
- TF-IDF mots (1–2 grammes) et/ou caractères (`char_wb`, 3–5), avec `sublinear_tf` ; `min_df` et `max_features` sont réglables.
- `token_pattern` garde les espaces réservés comme jetons entiers : `<EXT_EMAIL_3>`, `<HEADER>`, `<AMOUNT_2>`, `<call>`… Il les garde aussi entiers pour l'analyseur par caractères.
- Les arguments JSON sont lus comme texte (clés et valeurs), sans nettoyage.
- Option `TruncatedSVD` pour les modèles denses de l'AutoML (k-NN).
- Les embeddings de phrases sont hors périmètre : piste notée pour le lot C.

**Tests :**
- le vocabulaire n'apprend que le train : un mot vu seulement en validation est absent ;
- aucune colonne hors `feature_columns(view)` n'est lue ;
- le résultat est déterministe ;
- les espaces réservés sont intacts ;
- les champs sont séparés : un même mot dans deux champs donne deux colonnes.

**Fini quand :** bloc utilisable par une ligne de YAML, et dimension et temps par pli notés dans le JOURNAL.

## Étape 2 — Baselines v0.0 et v0.1 (KAN-20, B.5)

### v0.0 : repères à battre (`models/baselines.py`, `kind: trivial | shortcut`)

| Config | Contenu | Attendu |
|---|---|---|
| `v0_0_majority` | ✅ existe | ROC-AUC 0,5, PR-AUC = prévalence |
| `v0_0_random` | score uniforme, graine fixe | idem, sert de contrôle du bootstrap |
| `v0_0_position` | `uses_position: true`, régression logistique sur t encodé en indicatrices | reproduire 0,876 (`y_scan`) et 0,779 (`y_gate`) de l'EDA, sinon expliquer l'écart |
| `v0_0_keyword` | règle de surface de l'article AgentDrift **si elle est reconstructible fidèlement** ; sinon une liste de mots-clés d'injection fixée a priori, déclarée comme telle | repère « règle » ; ne jamais inventer la règle de l'article |

Les raccourcis sont étiquetés `shortcut` partout et ne sont jamais sélectionnés comme modèle.

### v0.1 : baseline officielle (`v0_1_baseline.yaml`, deux vues, texte anonymisé, sans `thought`)

- Modèle : `text_block` (mots) + `LogisticRegression(class_weight="balanced")`. La régularisation C est réglée par `GridSearchCV(cv=inner_cv, scoring="average_precision")` sur une petite grille (5 valeurs).
- Coût estimé :
  - environ 3 min par pli externe avec `inner_oof`, soit environ 15 min par vue ;
  - environ 4 min au total avec `threshold_source: estimator`.
- Choix mots ou caractères : une seule comparaison sur `cv`, puis choix fixé et noté. Les caractères coûtent 5 fois plus.
- **Calibration justifiée** (exigence Step 2.3) :
  - comparer sans calibration, Platt et isotonique, ajustés sur les plis internes par `CalibratedClassifierCV(cv=inner_cv)`, jamais sur le pli évalué ;
  - rapporter le Brier, l'ECE (bins à effectifs égaux, nombre justifié) et le diagramme de fiabilité.
- **Argument à écrire :**
  - `class_weight="balanced"` gonfle les probabilités des positifs : le score n'est plus une probabilité.
  - Une calibration monotone **ne change ni la PR-AUC, ni le ROC-AUC, ni le rappel à FPR fixé** quand le seuil est fixé sur les mêmes scores.
  - Elle sert à la décision à 3 niveaux (D.3), à la combinaison scanner + portail (v0.4) et au transfert de seuil (D.5).
  - Il faut le vérifier numériquement : mêmes métriques de rang avant et après calibration.
- **Brier et ECE** : à placer dans `evaluation/metrics.py`, propriété du lot D. Les écrire avec Oscar (KAN-28 / KAN-29), ou les écrire dans le lot B et les lui faire relire. À trancher (décision 2).
- **Analyses :**
  - comparaison avec la position seule (différence appariée) ;
  - rappel par type d'attaque et taux d'alerte par classe négative (déjà dans `metrics.json`) ;
  - **ablation par champ** (tableau du 25/11 refait dans le harnais) ;
  - **variante portail « action seule »** (`v0_1_gate_action.yaml`, colonnes `task`, `proposed_call_tool`, `proposed_call_args`) ;
  - domaine exclu (`evaluation: lodo`) ;
  - top 30 des coefficients par champ, **vérifiés à la main** : jetons de position, tics du générateur, espaces réservés suspects. Tout raccourci repéré va au JOURNAL et alimente A.5 (KAN-14) et l'étape 6 ;
  - courbes d'apprentissage (10, 25, 50 et 100 % des gabarits d'entraînement) et écart entraînement / évaluation (déjà dans `metrics.json`).

**Livrables :**
- `configs/models/v0_0_*.yaml` et `v0_1_*.yaml` ;
- `results/runs/v0_0_*/` et `results/runs/v0_1_*/` ;
- les figures dans `report/latex/figures/` (fiabilité, courbes PR, ablation par champ, courbes d'apprentissage) ;
- une entrée dans le JOURNAL ;
- la comparaison prudente avec les repères publiés de CONTEXT §5 (protocole différent).

## Étape 3 — Blocs non textuels (KAN-18, B.3)

Fichiers `features/structural.py` et `features/provenance.py` : des transformeurs scikit-learn.

**Principe :** chaque variable est calculée **à partir des seules colonnes de la vue**. On analyse les lignes `<call> outil {args}` et `<result> …` de `history_text`, et les champs de l'appel courant.
- La causalité est héritée des vues, déjà testée par perturbation. Le contrat du harnais reste intact, sans table externe.
- Seule exception : la table fixe `tool_risk.yaml`, lue et non ajustée.
- Toute statistique apprise (encodage, échelle, P(catégorie | tâche)) l'est dans `fit`, donc sur le pli d'entraînement.

| Bloc | Vue | Variables |
|---|---|---|
| `structure` | les deux | catégorie de risque de l'outil proposé ou source (6 catégories + `unknown`) ; n-grammes (1–3) de la séquence des catégories passées ; appels `external_send` / `payment` déjà faits (présence, puis compte plafonné) ; changement de catégorie par rapport à l'appel précédent |
| `tool_name` | les deux | nom d'outil en indicatrices, bloc **séparé** : inutilisable en domaine exclu, contrôle de H1 |
| `provenance` | portail | pour chaque valeur des arguments de a_t (adresse, URL, montant, contact, identifiant), d'où elle vient : la tâche, un `<result>` antérieur, un `<call>` antérieur, ou nulle part. Détails : destinataire interne (`<INT_…>`), externe (`<EXT_…>`) ou contact connu (`<CONTACT_k>`) ; nouveau ou déjà vu ; montant déjà vu dans la tâche. Les espaces réservés portent exactement ces relations : on s'en sert pour comparer à l'identique |
| `injection_cues` | scanner | adresse ou URL externe nouvelle dans `current_content` (absente de la tâche et de l'historique) ; marqueurs impératifs (liste fixée a priori) ; `<HEADER>` et `<CODE>` présents (variante masquée seulement) ; longueur relative du contenu |
| `coherence` (option) | portail | surprise de la catégorie de l'outil sachant la tâche : P(catégorie \| TF-IDF de la tâche), apprise **sur les lignes bénignes du pli d'entraînement seulement**. `expected_tools` n'est jamais lu |

**Points de vigilance**

- **Position déguisée.** Tout compte qui croît avec t (nombre d'appels, longueur de l'historique) réintroduit le raccourci de position (0,876 sur `y_scan`).
  - Pas de compte brut d'étapes.
  - Comptes de risque plafonnés.
  - Contrôle : corrélation de chaque variable avec t, et modèle « structure + position » comparé au modèle « structure ».
- **Brouillon de la table de risque.** `tool_risk.yaml` est un brouillon (KAN-58, relecture d'Oscar). Son statut (`draft` ou `validated`) est écrit dans chaque `metrics.json`, et les runs sont relancés après validation. Ne jamais ajuster une catégorie en regardant les labels.
- **Contrôle de la provenance.** Reproduire l'EDA : argument issu d'une réponse antérieure pour 46 % des H contre 19 % des B, et 58 % des H partielles. L'EDA est sur texte brut et le bloc sur texte anonymisé : un écart de quelques points est attendu. Il faut l'expliquer, pas l'ajuster.

**Tests :**
- valeurs sur des exemples écrits à la main pour chaque variable ;
- causalité : on reconstruit la vue après avoir modifié `obs_t` et les étapes > t, et les variables restent identiques ;
- un outil inconnu donne `unknown` (cas d'ATBench) ;
- aucune statistique calculée sur la validation ;
- un historique tronqué donne une provenance « inconnue », pas « nulle part ».

## Étape 4 — Test de H1 (KAN-62, B.7)

Même estimateur pour toutes les variantes (la régression logistique de v0.1, même grille de C) : seul l'effet des variables est isolé.

| Variante | Blocs |
|---|---|
| `h1_text` | `text_block` |
| `h1_struct` | `structure` + `provenance` / `injection_cues` (sans nom d'outil) |
| `h1_struct_tool` | idem + `tool_name` (contrôle) |
| `h1_text_struct` | concaténation |

- 4 variantes × 2 vues × 2 protocoles (CV par gabarit, domaine exclu), plus la variante « action seule » du portail. Coût estimé : moins d'une heure en tout avec les seuils fournis par l'estimateur.
- **Mesure clé :** la perte entre CV et domaine exclu pour chaque variante (PR-AUC, rappel à FPR 1 %), avec IC apparié.
  - H1 est soutenue si la perte de `h1_struct` est significativement plus faible que celle de `h1_text`.
  - Les mesures du 25/11 montrent une perte faible pour le texte : H1 peut très bien **ne pas** être soutenue sur AgentDrift. Il faudra le dire.
- Rappel par type d'attaque : vérifier que la provenance aide surtout les **partielles**. Ne pas le supposer.
- Seuils transférés tels quels en domaine exclu : écart de FPR observé (le domaine email du portail était déjà à 3,4 % le 25/11), premier élément pour H5 (D.5, KAN-32).
- Banking est lu avec son taux d'attaque plus bas (34 % contre 46 %, EDA C).

**Livrables :** `results/runs/h1/` (tableau variante × protocole × vue, figure), conclusion « soutenue / non soutenue / non tranchée » dans le JOURNAL.

## Étape 5 — AutoML v0.2 (KAN-21, B.6)

Fichiers `models/automl.py`, `pipelines/run_automl.py`, `configs/models/v0_2_automl.yaml`.

**Les quatre briques exigées par le cours, dans un seul `Pipeline` ajusté par pli :**

1. **Préparation**
   - Imputation : AgentDrift n'a pas de valeur manquante, mais la vue tronquée et ATBench si. On garde donc l'imputation et on le dit.
   - Encodage des catégories et mise à l'échelle des variables numériques.
2. **Feature engineering et sélection automatiques**
   - Le choix des blocs est un hyperparamètre : champs texte, mots / caractères, structure, provenance, indices.
   - Les paramètres TF-IDF sont aussi des hyperparamètres.
   - Sélection `SelectKBest(chi2)` ou L1 **dans** le pipeline.
3. **Sélection de modèles**
   - Familles : régression logistique, arbre de décision, SVM linéaire (calibré pour obtenir un score), forêt aléatoire / ExtraTrees (sur SVD ou sur les blocs non textuels), k-NN (sur SVD), Naive Bayes complémentaire (sur TF-IDF).
   - Pas de boosting : c'est v0.3 (KAN-23).
   - SVM à noyau RBF exclu, ce qu'il faut justifier : coût quadratique sur environ 50 000 lignes, et pas de gain attendu sur un problème déjà linéairement séparable à environ 0,97.
4. **Hyperparamètres**
   - Optuna 5.1.0 (épinglé), TPE à graine fixe.
   - Objectif : PR-AUC moyenne sur `inner_cv`.
   - Budget en essais **et** en temps dans le YAML.
   - Élagage des essais (pruner) après le premier pli interne.

**Protocole et coût**

- CV imbriquée : la recherche est relancée dans chaque pli externe, et le meilleur pipeline est réajusté puis évalué sur le pli externe.
- Les seuils viennent des scores hors pli internes du meilleur essai (`threshold_source: estimator`, étape 0).
- Coût estimé : 5 plis × N essais × 3 plis internes. Avec environ 5 s par ajustement en mots, N = 60 donne environ 1 h 30 par vue. Les caractères n'entrent que si le budget le permet.
- Rapporter :
  - **la famille et les blocs choisis par chaque pli** (si les plis choisissent des familles différentes, le dire) ;
  - le meilleur essai par famille ;
  - le temps de calcul.
- Calibrer le meilleur pipeline comme v0.1.

**Résultats :**
- comparaison v0.2 / v0.1 / position seule, avec IC et différence appariée, en CV et en domaine exclu ;
- **limites de v0.2 écrites**, exigées pour ouvrir v0.3 et pour R.1 (KAN-34).

Vu la saturation de la baseline, l'apport de v0.2 se lira sur le domaine exclu, le rappel à 0,2 % et le portail « action seule », pas sur la PR-AUC en CV.

## Étape 6 — Test de H3 (KAN-19, B.4)

Même modèle (v0.1, puis v0.2), mêmes plis, différences appariées entre marches successives.

| Marche | Vue | Commande des vues |
|---|---|---|
| 2 bis | brut, **avec** `thought` | `build_views --raw` (variante `_thought_raw`) |
| 3 | brut, sans `thought` | `build_views --raw` |
| 4 | anonymisé, sans `thought` | défaut |
| 5 (référence) | anonymisé + masqué, sans `thought` | défaut (`masked`) |

Le masquage seul ne bouge pas v0.1 (mesure du 25/11). Deux ablations s'ajoutent donc pour éviter de conclure « pas d'artefact » à tort :

- **Ablation par champ** (étape 2) : l'injection elle-même (`current_content`, `last_content`) porte presque tout le score. C'est attendu pour un scanner. Cela ne prouve pas que le score vient du style du générateur, mais rend l'évaluation externe indispensable.
- **Variante « action seule » du portail** : ce que le portail sait faire sans lire l'injection.

**Mesures :**
- PR-AUC et rappel à FPR 1 % par marche et par vue ;
- rappel par type d'attaque : la chute attendue est la plus forte sur les complètes et les échouées, là où les tics étaient les plus forts ;
- sonde du type d'attaque sur les scores, en rappelant qu'une chute faible **ne prouve pas** que le style est neutralisé (95 % du type reste lisible, JOURNAL 23/11) ;
- repères des raccourcis (position seule).

**Livrables :** `results/runs/h3/` (tableau, figure de l'échelle), conclusion prudente dans le JOURNAL, chaîne prête à être rejouée sur ATBench par D.6 (KAN-33).

---

## Transversal

- **Rapport (KAN-41)**, en anglais et en LaTeX, rédigé au fil des étapes :
  - vues et blocs → `04_methodology.tex` ;
  - v0.0, v0.1, v0.2 et la calibration → `05_models.tex` ;
  - H1 et H3 → `06_results.tex`.

  Chaque chiffre vient d'un `metrics.json`.
- **Gouvernance (KAN-49)**
  - Au moins 1 entrée dans PROMPT_LOG (feature engineering sous la contrainte « x_t seulement », ou espace de recherche de l'AutoML) et au moins 1 entrée dans AI_AUDIT (cible 2), notées **au moment où l'erreur arrive**.
  - A5 (ROC-AUC mis en commun) compte déjà pour le lot B.
  - Candidat déjà observé : un seuil fixé sur les scores d'entraînement donne 2 à 3 fois le FPR cible. Ce n'est une entrée d'AI_AUDIT que si une sortie de l'IA l'a réellement proposé.
- **`CONTRIBUTIONS.md`** : propriétaire et relecteur du lot B, versions v0.0–v0.2. La charge doit rester équilibrée (INSTRUCTIONS §2) : Victor porte déjà C2 (KAN-25–27), D.3 et D.5.

## Dépendances externes

| Dépend de | Pour | Contournement |
|---|---|---|
| KAN-58 (Oscar) — validation de `tool_risk.yaml` | blocs `structure`, H1 | utiliser le brouillon, noter le statut, relancer |
| KAN-28 (Oscar) — format des prédictions hors pli | toute comparaison B / C / D | format proposé le 23/11 (JOURNAL), à faire valider avant l'étape 2 |
| KAN-29 / KAN-31 (Oscar) — calibration, tests appariés | étape 2 (Brier, ECE), étapes 4–6 | différence appariée du harnais ; Brier et ECE dans `evaluation/metrics.py`, relus par D |
| KAN-14 (A.5) — raccourcis | étapes 2 et 6 | coefficients de v0.1 vérifiés à la main |
| Décision n° 5 (anonymisation d'ATBench) | rejouer H3 et H1 sur ATBench | hors lot B (D.6) |

## Décisions à trancher

1. **Propriétaire du lot B.** Jira attribue KAN-17 à KAN-21 et KAN-62 à Jean-Baptiste. La reprise par Victor doit être notée dans Jira et dans `CONTRIBUTIONS.md` : qui défend le module à l'oral (critère 5) ?
2. **Brier et ECE** : écrits par le lot B dans `evaluation/metrics.py` (relecture D), ou attendus de KAN-29 ?
3. **`threshold_source: estimator`** pour v0.1 aussi, ou seulement pour l'AutoML ? Recommandation : l'utiliser partout, après une comparaison unique avec `inner_oof` sur v0.1.
4. **Portail « action seule »** comme variante rapportée systématiquement à côté du portail complet. Recommandation : oui, car c'est la seule mesure de la valeur propre du portail.
5. **Mots ou caractères** pour v0.1 : à fixer sur une seule comparaison en CV, puis figé.
