# REFERENCE.md — Références

> Toutes les références sont à **vérifier (DOI / arXiv / Google Scholar)** avant d'entrer dans `report/latex/references.bib`. Une référence donnée par un LLM n'est jamais reprise telle quelle.
> Dernière mise à jour : 19/11/2026.

## Problème et benchmarks
- Greshake, K. et al. (2023). *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* AISec '23. arXiv:2302.12173.
- Debenedetti, E. et al. (2024). *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* NeurIPS 2024 Datasets & Benchmarks. arXiv:2406.13352.
- Zhan, Q. et al. (2024). *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents.* Findings of ACL 2024.
- OWASP (2025). *Top 10 for LLM Applications* — LLM01: Prompt Injection.
- Article de la compétition de red teaming d'agents (≈ 100 % de succès d'attaque, cité dans la synthèse) — **[À CONFIRMER]**.
- AgentDrift (dépôt GitHub Asif-0209/AgentDrift) et ATBench (AI45Research) — article associé **[À CONFIRMER]**.

## Méthodes
- He, P. et al. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* ICLR 2021.
- He, P., Gao, J., Chen, W. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.* ICLR 2023.
- Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019.
- Wang, C. et al. (2021). *FLAML: A Fast and Lightweight AutoML Library.* MLSys 2021.
- Wolpert, D. (1992). *Stacked Generalization.* Neural Networks.
- Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). *Isolation Forest.* ICDM 2008.
- Page, E. S. (1954). *Continuous Inspection Schemes.* Biometrika (CUSUM).
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* EMNLP 2014 (GRU).
- Inan, H. et al. (2023). *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.* arXiv:2312.06674.

## Évaluation et statistiques
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML 2017.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.*
- Saito, T., Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE.
- Kaufman, S. et al. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.

## Détecteurs pré-entraînés (benchmarks SOTA, CONTEXT.md §7.4)
Licences, accès et scores publiés **[À CONFIRMER]** avant toute citation :
- Meta Prompt Guard (1 / 2).
- ProtectAI `deberta-v3-base-prompt-injection-v2`.
