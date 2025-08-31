# MuCAL_DPO
This repository is for the paper, "MuCAL: Contrastive Alignment for Preference-Driven Knowledge Graph-to-Text Generation" (EMNLP 2025).
We basically provide three parts:
- Training and evaluation of MuCAL (KG-Text alignment model)
- Data collection for preference data creation.
- DPO training script.

## Models/Baselines
- [BE-MPNet-Hard2 (Hard-MuCAL)](https://huggingface.co/OneFly7/biencoder_ep10_bs32_lr2e5_cosine_annealing_hard_neg_2): A multilingual KG-Text alignment model across 6 languages (Arabic, English, Chinese, French, Spanish, Russian).
- [CE-MPNet](https://huggingface.co/OneFly7/crossencoder_ep10_bs4_trans3): A multilingual KG-Text reranker, which has better retrieval performance in common scenarios (without human-curated corruptions).
- [CLS-MPNet](https://huggingface.co/OneFly7/mucal_cross_encoder_regression_model_best): A multilingual KG-Text alignment model, trained as a binary classifier. (aligned/not aligned)
- [EREDAT](https://huggingface.co/teven/bi_all_bs192_hardneg_finetuned_WebNLG2017): An English KG-Text representation model.
- [FactSpotter](https://huggingface.co/Inria-CEDAR/FactSpotter-DeBERTaV3-Base): A reference-less English KG-Text alignment metric.
- [Data Quest-Eval (DQE)](https://github.com/ThomasScialom/QuestEval): A reference-less English Data-to-Text metric based on QA.


## Authors:
- [Yifei Song](https://melos7.github.io/yifei-website/) (CNRS, Loria, Université de Lorraine)
- [Claire Gardent](https://members.loria.fr/CGardent/) (CNRS, Loria)

## Citation
If you find this repo useful, please cite:
[TO ADD]
