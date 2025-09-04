# Preference Collection

In this part, you can find the codes for KG verbalization via few-shot prompting. We fix the same prompt for your different LLMs, and use the following models:
- Deepseek-r1-dstill-7B/-v3
- Llama3-70B
- Qwen2.5-7B/-14B/32B

We also keep the original text from KELM-Q1 (they have not been validated by human annotators).

## File Struture
You can easily visit the ```src/*.py``` files for KG verbalization, either via API or HF models.
Here are the list of data we provide:
- train_generations: the verbalzations of KG in the training set.
- dev_generations: the verbalizations of KG in the dev set.
- prefernece_data: the training/dev pariwise data collected by different alignment models via "best-worst" sampling.



