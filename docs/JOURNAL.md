# JOURNAL.md — Journal des décisions

> Toute décision qui change le protocole, les données, les modèles ou l'organisation est notée ici : date, décision, raison.
> Le cadre de référence reste [CONTEXT.md](CONTEXT.md). Une décision de ce journal l'emporte sur CONTEXT.md tant que celui-ci n'est pas mis à jour.

| Date | Décision | Raison |
|---|---|---|
| 17/11/2026 | Jeu principal AgentDrift ; ClawTrojan exclu | Licence et étiquetage par étape |
| 19/11/2026 | ATBench = test externe principal, AgentDojo = bonus | Coût de génération des traces AgentDojo |
| 19/11/2026 | Dossier de gouvernance nommé `ai_governance/` | Structure obligatoire des consignes |
| 19/11/2026 | Rapport en LaTeX, en anglais | Bonus +1/+2 ; note /2 sinon |
| 19/11/2026 | `PROMPT_LOG.md` limité aux 4–6 prompts les plus importants, structure fixe | Consigne mise à jour ; qualité des prompts plutôt que volume |
| 19/11/2026 | Python 3.12, `.venv` local, `src/` installé en éditable | Compatibilité torch / transformers ; imports propres |
| 19/11/2026 | Documents de cadrage déplacés dans `docs/` ; références et journal séparés de CONTEXT.md | Garder CONTEXT.md centré sur le projet |
| 19/11/2026 | `src/agentdrift/` ne contient que les modules en cours d'usage ; les autres sont créés au moment voulu | Éviter une arborescence de fichiers vides |
| — | Définition finale des cibles `y_scan` / `y_gate` / `y_traj` (CONTEXT.md §3.3) | À fixer après l'audit du schéma |
