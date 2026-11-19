# INSTRUCTIONS.md — Consignes du projet Machine Learning (ESILV 2026)

> Synthèse de **toutes** les consignes officielles : livrables, étapes, barème, règles de gouvernance IA et pénalités.
> Ce fichier dit **ce qui est exigé**. Le fichier [CONTEXT.md](CONTEXT.md) dit **ce que nous construisons**.
> Principe d'équipe : on construit le meilleur projet possible, puis on vérifie qu'il coche le barème. On ne bride pas le projet pour coller au barème.

---

## 0. Les règles à ne jamais oublier

| # | Règle | Conséquence si violée |
|---|---|---|
| 1 | **Rapport 100 % en anglais** | Note **divisée par 2** |
| 2 | Aucun scaler, imputer, TF-IDF, SMOTE ou sélection de variables ajusté **avant** le split train/validation (sauf fuite trouvée puis corrigée et documentée dans `AI_AUDIT.md`) | Note technique **plafonnée à 10/20** |
| 3 | Chaque étudiant écrit **au moins 1 entrée** dans `PROMPT_LOG.md` **et** au moins 1 dans `AI_AUDIT.md` | **−2 pts individuels** |
| 4 | Dossiers et fichiers de gouvernance présents dans l'archive Moodle, avec la hiérarchie exacte | **−2 pts** |
| 5 | Affirmer que l'IA n'a fait aucune erreur | **0/5** sur l'Error Auditing |
| 6 | Responsabilité totale : chaque test statistique, prétraitement et ligne de code nous engage | Hallucination non vérifiée ou fuite = faute méthodologique majeure |

---

## 1. Structure obligatoire du dépôt / de l'archive

```
<racine>/
├── requirements.txt          # versions épinglées (==), rien de non épinglé
├── src/                      # code, chemins relatifs uniquement
├── report/
│   └── Report_Group_XX.pdf   # XX = numéro du groupe
└── ai_governance/
    ├── CONTRIBUTIONS.md      # matrice de responsabilité par module
    ├── PROMPT_LOG.md         # 4 à 6 prompts critiques
    └── AI_AUDIT.md           # au moins 4 erreurs de l'IA détectées et corrigées
```

- On peut ajouter d'autres dossiers (`configs/`, `notebooks/`, `tests/`, `results/`…). Les quatre éléments ci-dessus restent obligatoires et ne doivent pas être renommés.
- Livrable final : **dépôt Git complet** + **archive `.zip`** sur Moodle avec la même hiérarchie.

---

## 2. Déroulé du projet (étapes du cours)

### STEP 1 — Choix du sujet et du jeu de données ✅ (rendu)
- Sujet : détection précoce du détournement d'agents IA par injection de prompt indirecte.
- Jeu principal : AgentDrift. Jeux externes : ATBench, AgentDojo (cf. CONTEXT.md §6).

### STEP 2 — Mise en place des solutions standard
1. **Analyse des données brutes** : qualité, statistiques descriptives, définition des variables, déséquilibre des classes, corrélations, réduction de dimension. Proposer un prétraitement **justifié**.
2. **Plan d'apprentissage et de test** : choix des données, construction des ensembles, méthodes de contrôle du sur-apprentissage, du sous-apprentissage et des fuites de données.
3. **Baseline** implémentée, avec justification de la calibration du modèle.
4. **Modèles de référence (benchmark)** issus de l'état de l'art, avec leurs performances publiées.
5. **Premier pipeline AutoML** ; définir le protocole d'évaluation, les risques et modes d'échec, et un **plan priorisé** pour la suite.

**Le pipeline AutoML doit couvrir :**
- **Préparation des données** : traitement des valeurs manquantes, encodage, mise à l'échelle.
- **Feature engineering** : création et sélection automatiques des variables les plus utiles.
- **Sélection de modèles** : plusieurs algorithmes standard (arbres de décision, régression logistique, SVM, etc.).
- **Optimisation des hyperparamètres**.

### STEP 3 — Amélioration de l'AutoML (versions 0.2, 0.3, 0.4, …)
1. Définir les **limites** de la version précédente.
2. Implémenter des **versions avancées** des algorithmes vus en cours.
3. Redéfinir le plan d'apprentissage et de test (données, ensembles, contrôle du sur-apprentissage).
4. **Analyser et critiquer** les résultats avec un protocole d'évaluation.
5. **Combiner plusieurs algorithmes** (ensemble learning) pour la décision.
6. Choisir **un algorithme hors programme** (deep learning possible : réseau de neurones, DNN, MoE…).
7. **Expliquer** cet algorithme et **justifier** ce choix avec un **article scientifique de référence** (Google Scholar…).
8. L'**implémenter**, l'évaluer et le comparer aux résultats précédents **et** à la baseline.

> Chaque amélioration est une **version numérotée** (v0.1 baseline, v0.2, v0.3…). Chaque version a ses limites écrites, ses résultats, et se compare à la précédente et à la baseline.

### Étape d'évaluation finale (semaine 4 de notre plan)
Notre dossier de cadrage prévoit une 4e phase (calibration, transfert de seuil, généralisation, intervalles de confiance, tests appariés, analyse critique). Elle alimente directement les critères « Scientific Report » et « Statistical Rigor ».

---

## 3. Barème officiel (/20)

### Partie A — Livrables de groupe (14 pts, note commune)

| Critère | Source | Pts | Niveau attendu pour la note maximale | Niveau insuffisant |
|---|---|---|---|---|
| **1. Prompt Guardrails** | `PROMPT_LOG.md` | 4 | Les prompts précisent le **schéma des données**, les **limites anti-fuite**, les **hypothèses statistiques** et des **tests de repli**. On voit l'**itération** d'un prompt nu vers un prompt contraint. | Prompts vagues en un coup (« Build a model for this CSV ») ; log absent ou moins de 4 entrées. |
| **2. Error Auditing** | `AI_AUDIT.md` + code | 5 | **Au moins 4 erreurs de fond** détectées (fuite de prétraitement, mauvaise métrique pour le déséquilibre, hypothèses de test violées…) avec **analyse de cause** et **correctif vérifié**. | « L'IA n'a fait aucune erreur » ; seulement des fautes de frappe ; fuite ou API inventée restée dans le pipeline final. |
| **3. Reproducibility** | Archive + code | 3 | Hiérarchie `.zip` exacte ; dépendances épinglées ; **graines fixées** ; scripts ou notebooks propres et modulaires ; charge de travail **équilibrée** dans `CONTRIBUTIONS.md`. | Chemins cassés, dépendances manquantes, métriques non reproductibles, gros blocs de code non attribués. |
| **4. Scientific Report** | `Report_Group_XX.pdf` | 2 | Synthèse analytique rigoureuse ; **intervalles de confiance / incertitude** ; discussion honnête des **modes d'échec** du modèle, au-delà du texte générique. | Texte LLM non retravaillé, affirmations non étayées, tableaux de métriques sans raisonnement métier. |

### Partie B — Soutenance individuelle (6 pts, note par étudiant, démonstration en direct de 3 min)

| Critère | Pts | Niveau attendu pour la note maximale | Niveau insuffisant |
|---|---|---|---|
| **5. Code Ownership** (module attribué) | 3 | Explique avec aisance **chaque fonction, hyperparamètre et appel de bibliothèque** de son module ; distingue ce qui vient de l'équipe et ce qui vient de l'IA. | Incapable de lire ou d'expliquer son propre code ; copier-coller aveugle évident. |
| **6. Statistical Rigor** (questions « what-if » en direct) | 3 | Justifie les compromis de métriques et de tests ; **prédit le comportement du pipeline** sous perturbation (covariate shift, valeurs aberrantes, hypothèses violées). | Incapable de définir les notions utilisées (p-value, fuite en validation croisée, compromis précision-rappel). |

### Bonus / malus du rapport
- **+1 ou +2 pts** si le rapport est écrit en **LaTeX** (selon la qualité). → **On le fait en LaTeX.**
- Bonus si le rapport est particulièrement bien écrit ; malus s'il est difficile à lire ou à comprendre.
- **Rapport non anglais → note /2.**

---

## 4. Règles de gouvernance IA

### 4.1 Responsabilité totale
Nous assumons chaque test statistique, chaque étape de prétraitement et chaque ligne de code. Une hallucination non vérifiée (API inventée, chiffre inventé, référence inexistante) ou une fuite de données compte comme une **faute méthodologique majeure**.

### 4.2 `CONTRIBUTIONS.md` — matrice de responsabilité
- Chaque module de `src/` a **un responsable principal** (celui qui le défend à l'oral) et éventuellement un relecteur.
- La charge doit être **équilibrée** et visible (modules, versions, sections du rapport, entrées de gouvernance).
- Le responsable doit pouvoir expliquer chaque fonction et chaque hyperparamètre de son module (critère 5).

### 4.3 `PROMPT_LOG.md` — interactions critiques
- **4 à 6 entrées**, les plus importantes du projet, **au moins 1 par étudiant**.
- On ne trace pas les recherches de syntaxe. On trace les prompts des **décisions à fort enjeu** :
  - feature engineering et **prévention des fuites** ;
  - choix des **tests statistiques** et vérification des hypothèses (normalité, colinéarité, déséquilibre des classes) ;
  - **sélection de modèle**, réglage des hyperparamètres, **choix des métriques**.
- Chaque entrée montre l'**itération** d'un prompt nu (v0) vers un prompt contraint (v1, v2…), ce que la réponse du LLM avait de faux ou d'incomplet, et comment on l'a vérifiée.
- **Chaque prompt final suit une structure complète et fixe** (modèle dans `ai_governance/PROMPT_LOG.md`) :

| Bloc | Contenu |
|---|---|
| **Rôle** | Expertise attendue du LLM (ex. « ML engineer spécialisé en évaluation de classifieurs déséquilibrés ») |
| **Contexte** | Projet, étape, version du pipeline, ce qui existe déjà |
| **Tâche** | Ce qu'on demande, en une phrase précise |
| **Objectif** | Pourquoi, et quel critère de réussite |
| **Schéma des données** | Colonnes, types, unité d'observation, clé de groupe, prévalence |
| **Contraintes anti-fuite** | Ce qui est interdit (fit hors pli, futur de la trajectoire, seuil sur le test…) |
| **Hypothèses statistiques** | Ce qu'on suppose, ce qu'il doit vérifier, ce qui est violé (dépendance des étapes…) |
| **Contraintes techniques** | Bibliothèques et versions, graines, API autorisées, style de code |
| **Format de sortie** | Structure exacte attendue (code, tableau, justification) |
| **Tests de repli / vérification** | Tests que la réponse doit passer ; quoi faire si une hypothèse échoue |
| **Auto-critique demandée** | Lister les limites, hypothèses non vérifiables et risques de la réponse |

- Le prompt final est donné **en anglais** (copie exacte), avec un résumé en français de la réponse et de l'audit.

### 4.4 `AI_AUDIT.md` — rapport « Catch & Correct »
- **Au moins 4 erreurs** (une autre version dit ≥3 : on vise **6 à 8**, au moins 1 par étudiant, idéalement 2).
- Erreurs **de fond** sur la logique ML, pas des fautes de frappe. Exemples cités par les consignes : scaler ajusté avant le split train/test, accuracy utilisée sur un jeu déséquilibré 99:1, diagnostics de résidus oubliés.
- Chaque entrée contient : la sortie fautive de l'IA, la **cause racine**, le **correctif**, la **preuve que le correctif marche** (test, métrique avant/après) et la **version** du code concernée (commit ou tag).
- Les erreurs doivent être **réelles**. On les note au fil de l'eau, on ne les invente pas après coup.

### 4.5 Red flags et pénalités
| Red flag | Pénalité |
|---|---|
| Fuite de données non détectée (scaler, imputer, SMOTE ajusté avant le split) et non corrigée dans `AI_AUDIT.md` | Note technique **plafonnée à 10/20** |
| Un étudiant sans entrée dans `PROMPT_LOG.md` **et** dans `AI_AUDIT.md` | **−2 pts** individuels |
| Archive non conforme (dossiers ou fichiers de gouvernance manquants) | **−2 pts** |

---

## 5. Rapport (`report/Report_Group_XX.pdf`)

**Langue : anglais. Format : LaTeX (bonus).**

### Contenu exigé
1. **Business case** : problème précis, objectif, lien direct avec notre spécialisation. Ce cadrage détermine les données nécessaires.
2. **Description du jeu de données** et de sa source.
3. **Exploration** : graphiques et figures.
4. **Formalisation** du problème.
5. **Présentation des modèles**.
6. **Traitement de chaque obstacle rencontré** : optimisation des paramètres, sur-apprentissage, sous-apprentissage, déséquilibre…
7. **Comparaison des résultats** des modèles.
8. **Conclusion** sur la réponse apportée au business case.
9. **Références** : articles scientifiques et sources externes.

### Plan type proposé par les consignes (non exhaustif)
- Business scope
- Problem formalisation and methods — Algorithm description — Limitations
- Methodology — Data description and exploration — Missing values — Imbalanced data — Outliers — Data splitting for train/test
- Algorithm implementation and hyperparameters
- Results — Metrics — Overfitting / underfitting / imbalance — Evaluation and comparison with the baseline
- Discussion and conclusion

### Ce qui fait gagner les points (critère 4)
- Intervalles de confiance sur **toutes** les métriques principales.
- Discussion des **modes d'échec** propres à notre problème (attaques retardées, transfert entre domaines, artefacts du générateur), pas une liste générique de limites.
- Chaque tableau de métriques commenté avec un raisonnement métier (coût d'une fausse alerte, coût d'une attaque manquée).

---

## 6. Soutenance (3 min par étudiant)

- Démonstration en direct de **son** module.
- Se préparer à :
  - expliquer chaque fonction, hyperparamètre et appel de bibliothèque de son module ;
  - dire ce qui a été conçu par l'équipe et ce qui a été généré par l'IA (et ce qu'on a corrigé) ;
  - répondre à des « what-if » : que se passe-t-il en cas de covariate shift, de valeurs aberrantes, d'hypothèse violée, de changement de prévalence, de seuil transféré sur un autre domaine ;
  - définir : p-value, fuite en validation croisée, compromis précision-rappel, calibration, intervalle bootstrap, validation croisée groupée.

---

## 7. Checklist de rendu

**Archive et reproductibilité**
- [ ] `requirements.txt` entièrement épinglé (`pip freeze` filtré), version de Python indiquée dans le README.
- [ ] `src/`, `report/Report_Group_XX.pdf`, `ai_governance/{CONTRIBUTIONS,PROMPT_LOG,AI_AUDIT}.md` présents.
- [ ] Chemins relatifs uniquement ; aucun chemin absolu (`/Users/...`).
- [ ] Graines fixées (Python, NumPy, scikit-learn, PyTorch, CUDA si utilisé) ; résultats reproduits par un membre qui n'a pas écrit le code.
- [ ] Une commande documentée par résultat du rapport (README).
- [ ] Données brutes non commitées si trop lourdes, avec script de téléchargement + checksum.

**Gouvernance**
- [ ] `PROMPT_LOG.md` : 4–6 entrées les plus importantes, ≥1 par étudiant, structure complète (rôle → auto-critique), itération nu → contraint visible.
- [ ] `AI_AUDIT.md` : ≥4 (cible 6–8) erreurs de fond, ≥1 par étudiant, cause racine + correctif + preuve + version.
- [ ] `CONTRIBUTIONS.md` : un responsable par module, charge équilibrée.

**Méthodologie**
- [ ] Toutes les transformations ajustées **dans** les plis (pipelines scikit-learn).
- [ ] Split groupé par gabarit de tâche ; seuils choisis sur la validation, jamais sur le test.
- [ ] Métriques adaptées au déséquilibre (pas d'accuracy seule).
- [ ] Intervalles de confiance et tests appariés pour les comparaisons de modèles.
- [ ] Un test automatique anti-fuite dans `tests/`.

**Rapport**
- [ ] En anglais, en LaTeX, toutes les sections du §5.
- [ ] Références vérifiées (aucune référence inventée par l'IA).
