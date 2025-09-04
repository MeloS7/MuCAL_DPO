# MuCAL Training and Evaluation

In this part, we provide the training of MuCAL variants, including bi-encoder, cross-encoder, binary classifier, and also the evaluation codes for alignment models' retrieval performance.

## File Description
- Contrastive:
    - BiEncoder: The codes of bi-encoder architecture model training and evaluation.
    - CrossEncoder: The codes of cross-encoder architecture model training and evaluation.
- Regression:
    - in_batch: The codes of binary classifier baseline model training and evaluation.
- Eval_metric:
    - corrupted_res_analyze: corruption type ananlysis codes, as shown in our Appendix.
    - eval_metric_on_corruption: ditto
    - retrieval_6k: Evaluation code on the Multi-Test-1K-Corr test set.
- Script:
    - train_bi_cr_example: A demonstrated script for Contrastive learning.
    - eval_re_6k: A demonstrated script for corrupted retrieval test.

## Code Usage
Basically, we code all model variants in the similar structures, please check the ```main.py``` file in each component and the demonstrated script before using them.

## Data
The data is compressed into data.zip file.


