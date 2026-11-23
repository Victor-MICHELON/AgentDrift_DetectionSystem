# PLAN_DONNEES.md — Plan d'implémentation : préparation des données et EDA (Lot A)

> Plan de la première phase du Step 2 : obtenir des données propres, auditées, anonymisées et découpées, et une EDA présentable qui justifie chaque choix de prétraitement.
> Références : [CONTEXT.md](CONTEXT.md) §3.3, §5, §6, §8.1, §9 ; [JOURNAL.md](JOURNAL.md) (décisions du 19/11) ; [INSTRUCTIONS.md](INSTRUCTIONS.md) Step 2.1–2.2.
> Version : 21/11/2026 (couverture de l'étape 10, dépendances de l'étape 7, étape 6 lancée). Toute modification du protocole est notée dans JOURNAL.md.

---

## Vue d'ensemble

| # | Étape | Nature | Statut |
|---|---|---|---|
| 1 | Téléchargement reproductible + checksums | mise en place | ✅ fait (20/11) |
| 2 | Parsing : JSONL → `trajectories.parquet` / `steps.parquet` + cibles | mise en forme | ✅ fait (20/11) |
| 3 | Schéma validé (pandera) + invariants | mise en forme | ✅ fait (20/11) |
| **4** | **EDA, partie A : données brutes** | analyse | ✅ fait (20/11) |
| 5 | Anonymisation (masquage H3 → Lot B, `features/masking.py`) | prétraitement | ✅ anonymisation faite (20/11) |
| 6 | Annotations manuelles (risque des outils, `injection_visible`) | enrichissement | 🔄 brouillons prêts (21/11), relecture humaine à faire |
| **7** | **EDA, partie B : effet du prétraitement** | analyse | 🔄 anonymisation faite (22/11) ; masquage (KAN-19) et annotations (KAN-58) en attente |
| 8 | Splits groupés + `folds.parquet` | protocole | ✅ fait (20/11) |
| **9** | **EDA, partie C : protocole d'évaluation + synthèse** | analyse | ✅ fait (23/11) |
| 10 | Tests anti-fuite | protocole | 🔄 en grande partie couvert (20/11) ; reste L12 (attend `features/views.py`) |
| 11 | Adaptateur ATBench → schéma d'étapes (`data/external.py`) | mise en forme | à faire (avant la phase 4) |

**Ordre :** 2 → 3 → 4 → 5 → (6 ∥ 8) → 7 → 9 → 10 → 11. L'étape 8 ne dépend que des métadonnées des trajectoires ; l'étape 6 est un travail humain (préparation possible par l'IA, validation et relecture par l'équipe) et ne doit pas bloquer le reste.

L'EDA vient **avant** le prétraitement : elle le justifie (consigne Step 2.1, « proposer un prétraitement justifié »). Elle est complétée après le prétraitement (preuve que les parades marchent) et après les splits.

---

## Étape 1 — Téléchargement reproductible ✅

- `python -m agentdrift.data.download [--atbench] [--verify-only]` (`src/agentdrift/data/download.py`).
- AgentDrift récupéré par SHA (`014a514`) ; refus si commit différent, fichier suivi modifié ou checksum faux.
- ATBench figé : révision HF `4476ef9`, `ATBench/test.json` (1 000 trajectoires), SHA-256 dans `configs/data.yaml`.
- Tests : `tests/test_download.py`.

## Étape 2 — Parsing (`src/agentdrift/data/load.py`) ✅

Commande : `python -m agentdrift.data.load` (≈ 4 s ; vérifie les checksums, construit, valide, écrit). Lecteurs : `read_trajectories()`, `read_steps()` (test fixe exclu par défaut), `read_templates()`. Tests : `tests/test_data.py`.

**Sorties** (`data/interim/`, non commité) :

- `trajectories.parquet` — 1 ligne par trajectoire : `traj_id`, `origin_split`, `domain` (= `agent`), `category`, `source_category`, `dataset_category`, `attack_type`, `compliance`, `task`, `template_id`, `complexity`, `world_id`, `world_json`, `n_steps`, `injection_step`, `first_h_step`, `y_traj`.
- `steps.parquet` — 1 ligne par étape : `traj_id`, `t`, `thought`, `tool`, `args_json` (JSON canonique, clés triées), `obs`, `label`, cibles `y_scan`, `y_gate`, `is_first_h`, plus les clés de groupe (`origin_split`, `domain`, `template_id`, `world_id`, `source_category`).
- `templates.parquet` — les 250 gabarits (`template_id`, `domain`, `instruction`, `complexity`, `expected_tools`). **Oracle uniquement (L12).**

**Cibles** (CONTEXT §3.3) :

- `y_scan(t) = label_{t−1} ∈ {injection_point, failed_injection}`, 0 pour t = 0 ;
- `y_gate(t) = label_t == hijacked` ;
- `is_first_h` = premier H de la trajectoire (métriques « à la première alerte ») ;
- `y_traj = category == attacked` — **provisoire, [À CONFIRMER]**.

`world_id` = hash SHA-1 (12 caractères) du JSON canonique de l'objet `world` hors `date` (la date est unique par monde : 1 141 mondes avec ou sans elle). Indice d'étape `t` 0-based.

## Étape 3 — Schéma validé (`src/agentdrift/data/schema.py`, pandera) ✅

- Types, valeurs autorisées (`label`, `domain`, `category`, `source_category`), unicité `(traj_id, t)`, `t` contigu à partir de 0.
- Invariants par trajectoire :
  - `benign`, `hard_negative` : uniquement des étapes `benign` ;
  - `attacked` : exactement un `injection_point`, au moins un `hijacked`, jamais de H avant l'injection ;
  - `failed_attack` : exactement un `failed_injection`, aucun `hijacked` ;
  - écart injection → premier H : 1 pour `attacked_full` et `attacked_partial`, ≥ 2 pour `attacked_delayed` (observé : 2 → 837, 3 → 60, 4 → 3 ; corrige CONTEXT §6.2, cf. JOURNAL 20/11) ;
  - injection (réussie ou non) toujours à t = 1–3 ; `y_scan` cohérent avec le label précédent.
- Résultat sur le corpus : **aucune violation** ; totaux conformes (71 024 étapes, 7 036 `y_scan`, 12 349 `y_gate`, 5 536 premiers H).
- Le chargement valide avant d'écrire. Un invariant violé est **rapporté**, jamais corrigé en silence.

## Étape 4 — EDA, partie A : données brutes ✅

Fait le 20/11 : `notebooks/01_eda.ipynb` (≈ 12 s, VS Code, interpréteur `.venv`), `python -m agentdrift.pipelines.run_eda` (métriques dans `results/runs/eda/`, 17 figures). Écarts avec les mesures du 19/11 notés dans JOURNAL. Entrée PROMPT_LOG P1.

### Livrables
- `notebooks/01_eda.ipynb` — notebook présentable, **en anglais** (figures et texte réutilisés dans la section 3 du rapport et pour la soutenance).
- `src/agentdrift/data/audit.py` — tous les calculs (fonctions qui renvoient des DataFrames).
- `src/agentdrift/viz/plots.py` — style commun (palette, tailles), fonctions de figures, export.
- `src/agentdrift/pipelines/run_eda.py` — régénère `results/runs/eda/metrics.json` et toutes les figures sans le notebook.
- Figures retenues exportées en PDF dans `report/latex/figures/`.

### Règles de présentation (révisées le 23/11, valables pour les trois notebooks EDA)
- Notebooks **de présentation** des données et de leur préparation : chaque section suit **pourquoi c'est important pour le garde-fou → figure → table lisible → ce qu'on voit → ce qu'on en fait**, en langage clair relié au projet. Les codes internes (fuites L…, hypothèses H…) ne figurent qu'en note de traçabilité en bas de section.
- **Explications en cellules Markdown**, jamais du code qui affiche du texte. Les chiffres cités sont donc écrits à la main ; la **dernière cellule** (code replié) les recalcule tous et arrête le notebook si l'un d'eux ne correspond plus aux données (`viz/notebook.py::check`). Toute modification des données ou du code impose de réexécuter et de corriger le texte signalé.
- Aucune logique métier dans le notebook (calculs dans `data/audit.py`, figures dans `viz/plots.py`) ; « Restart & Run All » en quelques minutes ; enregistré avec ses sorties ; titres des figures identiques à ceux de `run_eda`.
- Le test fixe n'oriente aucun choix : statistiques descriptives par partie (train / val / test côte à côte), baselines « raccourci » ajustées sur train et évaluées sur val.

### Sections

| § | Section | Contenu et figures | Ce que la section doit montrer |
|---|---|---|---|
| A0 | Provenance | Sources, licences, versions figées, checksums | Reproductibilité |
| A1 | Vue d'ensemble | Trajectoires et étapes par domaine × catégorie, 103 outils, 250 gabarits ; frises d'exemples (bénigne, complète, partielle, retardée) | Comprendre l'objet étudié |
| A2 | Qualité | Valeurs manquantes (`thought` / `obs` / `args` vides), doublons exacts, quasi-doublons par gabarit, 263 adresses avec espace, valeurs aberrantes (longueurs) | Exigé par le Step 2.1 |
| A3 | Déséquilibre | Prévalence de `y_scan` / `y_gate` par étape, trajectoire, domaine ; étapes H contre premiers H | Choix des métriques (PR-AUC), `class_weight` |
| A4 | Structure temporelle | Longueurs par catégorie ; carte de chaleur label × position ; position de l'injection ; écart injection → premier H par type | Raccourci de position ; rôle scanner / portail (H2) |
| A5 | Outils et actions | Fréquence des outils par label ; aucun outil réservé aux H ; part hors `expected_tools` | H1, H8 ; oracle jamais en entrée |
| A6 | Contenu textuel | Longueurs par champ ; préfixes d'injection par catégorie ; injections invisibles ; mots d'aveu dans `thought` | Masquage H3 ; variante sans `thought` |
| A7 | Arguments et provenance | Adresse inconnue, destinataire du même domaine, argument issu d'une `obs` antérieure ; B contre H | Variables de provenance (H1, H8) |
| A8 | Fuites et raccourcis | Concentration des catégories par monde ; baselines « monde seul » et « position seule » ; V de Cramér métadonnées × label ; longueur totale | L1–L12 → décisions de prétraitement |
| A9 | Réduction de dimension | TF-IDF + SVD puis t-SNE (scikit-learn), coloré par label puis par domaine | Regroupement par domaine plutôt que par attaque ? (style du générateur) |
| A10 | Test externe | ATBench contre AgentDrift : longueurs, nombre d'outils, catégories retenues | Ampleur du changement de distribution |

## Étape 5 — Anonymisation ✅ (masquage H3 déplacé dans le Lot B)

Commande : `python -m agentdrift.data.anonymize` (≈ 8 s) → `data/interim/{trajectories,steps}_anon.parquet`. Lecture : `read_steps(anonymized=True)`, `read_trajectories(anonymized=True)` — **obligatoire pour tout modèle** (L9). Code : `src/agentdrift/data/anonymize.py` ; tests : `tests/test_anonymize.py`.

- **Ce qui est remplacé** (`task`, `thought`, `args_json`, `obs`) : utilisateur → `<USER>`, contact k → `<CONTACT_k>` (nom complet, et prénom ou nom seul quand il désigne une seule personne du monde) ; société et libellé du domaine de l'utilisateur → `<COMPANY>` ; adresses → `<USER_EMAIL>`, `<CONTACT_k_EMAIL>`, `<INT_EMAIL_k>` (même domaine que l'utilisateur) ou `<EXT_EMAIL_k>` ; URL → `<INT_URL_k>` / `<EXT_URL_k>` ; domaines nus → `<INT_DOMAIN_k>` / `<EXT_DOMAIN_k>`. Les 263 adresses avec espace sont réparées avant. `world_json` est retiré des tables anonymisées.
- **Pourquoi aussi les prénoms et le libellé de domaine** (mesuré le 20/11, part des trajectoires) : nom complet 41 %, prénom seul 7 %, société 13 %, libellé de domaine hors adresse 44 %, adresse du monde 15 %.
- **Propriétés testées** : déterministe et ligne par ligne (rien d'ajusté, compatible avec la règle 2) ; **causale** (la numérotation k suit l'ordre tâche → thought → args → obs, donc le texte de l'étape t ne dépend jamais des étapes suivantes, L3) ; générique (règles tirées de l'objet monde et d'expressions régulières, jamais d'une liste extraite du corpus) ; JSON des arguments toujours valide ; lignes, labels et cibles inchangés.
- **Les noms de personnes des gabarits** (« Rachel Lee ») ne sont pas touchés : ils n'appartiennent pas au monde et sont couverts par le split par gabarit.

**Critère de réussite (corrigé).** L'ancien critère « la recherche du monde retombe au hasard » ne mesurait rien : cette baseline utilise `world_id`, hash du monde brut, que le texte ne change pas. Nouveaux critères :

| Critère | Résultat (20/11) |
|---|---|
| Aucun identifiant résiduel (noms complets, adresses du monde, nom de société) | 0 trajectoire ; libellé de domaine : 3 faux positifs du contrôle (sous-chaînes « unitedhealthcare », un mot de passe, un nom de classe) |
| Sonde textuelle (1-plus-proche-voisin TF-IDF, ajusté sur train, val) qui retrouve le monde | 12,2 % → **3,1 %** ; hasard à domaine × catégorie égaux : 2,1 % (le résidu vient du contenu propre à la catégorie) |
| Provenance de l'EDA A7 conservée | tous les indicateurs identiques à ≤ 0,3 point près |

**Masquage H3** (`mask_artifacts`, variante du rapport) : déplacé dans le Lot B (`features/masking.py`, décision 2). La liste de préfixes tirée de l'EDA sur tout le corpus contredit la règle « générique » et manque des variantes présentes dans le train seul (`COMPLIANCE NOTICE:`, `SECURITY NOTICE:`, `ALERT:`, `IMMEDIATE ACTION REQUIRED:`…). Règle à utiliser : toute étiquette en majuscules suivie de « : », `Per policy <code>`, montants `$…` → `<AMOUNT>`, appliquée à tout le texte quel que soit le label.

## Étape 6 — Annotations manuelles (`configs/annotations/`, versionnées)

Code : `src/agentdrift/data/annotations.py` ; tests : `tests/test_annotations.py` ; consigne : `configs/annotations/README.md`. Vérification : `python -m agentdrift.data.annotations` (couverture, avancement, valeurs).

- `tool_risk.yaml` : 103 outils → 6 catégories (`read` 51, `internal_write` 29, `external_send` 11, `payment` 5, `delete` 4, `execute` 3). **Brouillon IA (21/11)**, 20 outils marqués `ambiguous` avec une alternative. Catégories fixées d'après le nom et l'effet déclaré de l'outil, sans regarder les labels. Relecture croisée par un deuxième membre, puis `status: validated`.
- `injection_visible.csv` : 300 étapes `injection_point` (100 complètes, 100 partielles, 100 retardées), tirées du train + val avec la graine 42 (`--sample`, qui n'écrase jamais une fiche existante). **Fiche en aveugle** : ordre mélangé, sans catégorie ni `traj_id` (le `traj_id` contient la catégorie). La clé `sample_id → (traj_id, t)` est dans `injection_visible_key.csv`. Colonnes `oui` / `non` + commentaire, à remplir par l'équipe.
- **Écart avec le plan initial** : les annotations sont jointes **à la lecture** (`attach_tool_risk`, `attach_injection_visible`), pas écrites dans `steps.parquet` par `load.py`. Elles changent à la main après la construction des tables, et reconstruire les tables (et leurs copies anonymisées) à chaque modification serait fragile. Les jointures échouent au lieu de compléter en silence (outil sans catégorie, annotation sur une étape qui n'est pas une injection).

**Reste :** relecture de `tool_risk.yaml` et annotation des 300 lignes (idéalement à deux, avec un kappa de Cohen sur les lignes communes). Relecteur à désigner (décision n° 3).

## Étape 7 — EDA, partie B : effet du prétraitement

Fait le 22/11 (partie anonymisation) : `notebooks/02_eda_preprocessing.ipynb` (notebook séparé : la partie A reste figée avec ses sorties, la partie B se complète à mesure que le masquage et les annotations arrivent), `python -m agentdrift.pipelines.run_eda --part B` (≈ 7 s, `results/runs/eda_b/`, figures `eda_b1`–`eda_b3`). Calculs dans `data/audit.py` (section « part B »), figures dans `viz/plots.py`.

| § | Contenu | Résultat (22/11) | Statut |
|---|---|---|---|
| B1 | Même texte brut / anonymisé (/ masqué), un exemple par catégorie d'attaque ; occurrences des espaces réservés | `<CONTACT_k>` le plus fréquent ; contenu de l'attaque intact | ✅ (colonne « masqué » en attente) |
| B2 | Identifiants du monde restants | 41 % / 15 % / 27 % / 44 % des trajectoires → 0 ; 3 faux positifs du libellé de domaine | ✅ |
| B3 | Sonde textuelle du monde (`world_probe`) contre le hasard à domaine × catégorie égaux | 12,2 % → 3,1 % (hasard 2,1 %) | ✅ |
| B4 | Provenance avant / après (`provenance_before_after`) | écart maximal 0,33 point | ✅ |
| B5 | Masquage H3 | — | ⏳ `features/masking.py` (KAN-19) ; la section se remplit automatiquement |
| B6 | Catégories de risque des outils × labels | effectifs et 20 cas ambigus affichés ; **croisement avec les labels masqué tant que `tool_risk.yaml` n'est pas validé** (sinon la relecture pourrait s'ajuster sur les labels) | ⏳ KAN-58 |
| B7 | `injection_visible` par catégorie, accord avec l'union des expressions régulières de A6 (kappa de Cohen) | 0 / 300 annotées | ⏳ KAN-58 |

La baseline « monde seul » par `world_id` n'est **pas** un critère (indépendante du texte). Le notebook 01 a été corrigé sur ce point et sur la liste de préfixes (remplacée par la règle générique), puis réexécuté : tous les tableaux de la partie A sont inchangés.

**Pour finir l'étape :** réexécuter le notebook 02 et `run_eda --part B` quand KAN-19 et KAN-58 sont faits, puis rédiger les constats de B5 à B7.

## Étape 8 — Splits (`src/agentdrift/splits/grouped.py`) ✅

Commande : `python -m agentdrift.splits.grouped` (≈ 1 s) → `data/processed/folds.parquet` (empreinte `f69d2d09f08332a9`, à comparer entre membres). Tests : `tests/test_splits.py`. Configuration : `configs/split.yaml` (clés alignées sur les tables intermédiaires : `template_id`, `domain`, `world_id`).

- **Test fixe** : `fold = -1` (35 gabarits, 1 722 trajectoires) ; jamais renvoyé par `outer_splits` ni `lodo_splits`, exclu par défaut des lecteurs `read_*()`.
- **Validation croisée externe** : `StratifiedGroupKFold(5, shuffle=True, random_state=42)` sur train + val (215 gabarits), groupe = `template_id`, strate = `source_category × domain`. Les étapes héritent du pli de leur trajectoire (`attach_folds`).
- **Validation interne** : `inner_splits(partie_train)` (3 plis groupés par gabarit), générée à la demande dans chaque pli externe.
- **Leave-one-domain-out** : `lodo_splits` (5 configurations, pas de colonne stockée).
- **Contrôle** : `world_fold` = `GroupKFold(5)` par monde (ordre des mondes permuté avec la graine).

Résultat (20/11) :

| Pli | Gabarits | Trajectoires | `y_traj` | `y_gate` (étape) | `y_scan` (étape) |
|---|---|---|---|---|---|
| 0 | 43 | 2 165 | 43,3 % | 16,6 % | 9,9 % |
| 1 | 44 | 2 174 | 44,6 % | 17,9 % | 9,9 % |
| 2 | 42 | 2 140 | 44,6 % | 17,3 % | 10,0 % |
| 3 | 45 | 2 215 | 43,2 % | 17,3 % | 9,8 % |
| 4 | 41 | 2 120 | 44,5 % | 17,4 % | 9,8 % |
| test | 35 | 1 722 | 45,0 % | 17,9 % | 10,1 % |

Parts de catégories et de domaines à ± 2 points entre plis ; aucun gabarit dans deux plis (L1). 1 060 des 1 136 mondes de dev apparaissent dans plusieurs plis : attendu (on ne peut pas grouper à la fois par gabarit et par monde, JOURNAL 19/11), d'où l'anonymisation obligatoire. Les plis par monde ont une prévalence plus variable (`y_traj` 39,6–49,5 %), conséquence de la pureté des mondes (EDA A8).

## Étape 9 — EDA, partie C : protocole et synthèse ✅

Fait le 23/11 : `notebooks/03_eda_protocol.ipynb`, `python -m agentdrift.pipelines.run_eda --part C` (≈ 5 s, `results/runs/eda_c/`, figures `eda_c1`–`eda_c3`). Calculs dans `data/audit.py` (section « part C »). Bootstrap par gabarit dans `evaluation/bootstrap.py` (vectorisé, testé contre sklearn dans `tests/test_bootstrap.py`, réutilisable par le Lot D). **Règle L5 :** le test fixe ne fournit que des effectifs ; les largeurs d'IC sont mesurées sur les plis de dev avec un score de référence sans contenu (outil × position × domaine, ajusté hors pli), qui n'est pas un modèle du projet.

| § | Contenu | Résultat (23/11) |
|---|---|---|
| C1 | Taille et prévalence par pli ; plis par gabarit vs plis par monde | écarts de parts ≤ 1,2 point, `y_traj` à ± 1,4 point ; plis par monde : 9,9 points. Prévalence du test dans la fourchette des plis |
| C2 | Ce que les plis partagent | 0 gabarit partagé (plis, dev / test) ; 1 060 / 1 136 mondes partagés ; 99,6 % des trajectoires du test dans un monde vu en dev ; ≈ 50 trajectoires par gabarit |
| C3 | Domaine exclu | banking à part : 34 % d'attaques (46 % ailleurs), `y_gate` 9,8 % (18,8 %). À lire avec la prévalence du domaine exclu |
| C4 | Faux positifs attendus aux FPR cibles | 1 % : ≥ 71 partout. 0,2 % : 16 (portail) / 14 (scanner) sur le test, ≈ 27 sur une partie de validation interne |
| C5 | Largeur attendue des IC à 95 % (bootstrap par gabarit) | sur 35 gabarits, portail : ± 0,07 PR-AUC, ± 0,07 rappel à 1 % (score de référence, PR-AUC 0,61) ; effet de plan 5 à 16 ; à 0,2 %, les plis varient ≈ 2 × plus que le bootstrap à seuil fixe (variabilité du seuil non couverte) ; seuil à 1 % → 0,17–1,97 % observé sur le pli suivant |
| C6 | Tableau de synthèse constat → décision → hypothèse | 14 lignes : 8 appliquées, 2 en cours, 3 décidées dans CONTEXT, 1 proposée (0,2 % comme point secondaire) |

Le tableau C6 lit les `metrics.json` des parties A et B : relancer `--part A` et `--part B` avant `--part C` si elles changent. La colonne `status` est tenue à la main (`audit.SYNTHESIS_STATUS`).

## Étape 10 — Tests anti-fuite (`tests/test_splits.py`, `tests/test_data.py`)

| Contrôle | Test existant (20/11) | Statut |
|---|---|---|
| L1 : aucun gabarit partagé entre plis ni entre dev et test | `test_splits.py::test_corpus_l1_no_template_across_folds_or_test`, `test_synthetic_grouping_and_test_isolation` | ✅ |
| Test fixe jamais renvoyé par les générateurs de plis | `test_splits.py::test_outer_and_lodo_splits_never_yield_test_rows`, `test_data.py::test_readers_hide_fixed_test_by_default` | ✅ |
| Plis reproductibles à graine égale | `test_synthetic_grouping_and_test_isolation` (empreinte `folds_hash`) | ✅ |
| Prévalence par pli dans une tolérance | `test_corpus_folds_are_balanced` | ✅ |
| Validation interne groupée par gabarit | `test_inner_splits_are_grouped_by_template` | ✅ |
| Invariants du schéma sur une fixture | `test_data.py::test_toy_tables_pass_schema`, `test_violations_are_reported` | ✅ |
| Aucun nom / adresse / société du monde après anonymisation | `test_anonymize.py::test_corpus_has_no_world_identity_left` | ✅ |
| L12 : aucune colonne de métadonnées dans la liste des variables exposée | — | à faire quand `features/views.py` exposera sa liste de variables (Lot B, KAN-16) |
| L3 / L11 (vues strictes) | — | relève de `features/views.py` (Lot B) ; interface fournie : `steps.parquet` + cibles |

Il reste donc un seul test propre au Lot A (L12), qui dépend d'une interface du Lot B.

---

## Critères de fin de la phase

- `python -m agentdrift.pipelines.run_eda` reproduit les chiffres du notebook ; écarts avec les mesures du 19/11 notés dans JOURNAL.
- `pytest` vert, tests anti-fuite inclus.
- Gouvernance : ≥ 1 entrée `PROMPT_LOG.md` (candidat : split groupé ou anonymisation) ; erreurs d'IA réelles notées au fil de l'eau dans `AI_AUDIT.md` ; modules `data/` et `splits/` attribués dans `CONTRIBUTIONS.md`.

## Décisions d'équipe en attente

1. `y_traj` : `category == attacked` (provisoire) ?
2. ~~Anonymisation dans `data/`, masquage H3 dans `features/`~~ — appliqué le 20/11 (à confirmer en équipe).
3. Relecteur de `tool_risk.yaml` et de `injection_visible.csv` ?
4. **Proposé par l'EDA C (23/11)** : 1 % de FPR par étape comme point de fonctionnement principal ; 0,2 % rapporté comme point secondaire, toujours avec le FPR observé et sa dispersion sur les plis (≈ 15 faux positifs attendus sur le test). Résultats au seuil rapportés aussi en moyenne ± écart-type sur les plis, car l'IC bootstrap à seuil fixe ne couvre pas la variabilité du seuil.
