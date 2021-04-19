# Optimizing Dense Retrieval Model Training with Hard Negatives
Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, Shaoping Ma


This repo provides code, retrieval results, and trained models for our paper [Optimizing Dense Retrieval Model Training with Hard Negatives](https://arxiv.org/abs/2104.08051). The previous version is [Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently.](https://arxiv.org/abs/2010.10469)

Ranking has always been one of the top concerns in information retrieval researches. For decades, the lexical matching signal has dominated the ad-hoc retrieval process, but solely using this signal in retrieval may cause the vocabulary mismatch problem. In recent years, with the development of representation learning techniques, many researchers turn to Dense Retrieval (DR) models for better ranking performance. Although several existing DR models have already obtained promising results, their performance improvement heavily relies on the sampling of training examples. Many effective sampling strategies are not efficient enough for practical usage, and for most of them, there still lacks theoretical analysis in how and why performance improvement happens. 

To shed light on these research questions, we theoretically investigate different training strategies for DR models and try to explain why hard negative sampling performs better than random sampling. Through the analysis, we also find that there are many potential risks in static hard negative sampling, which is employed by many existing training methods. 

Therefore, we propose two training strategies named a Stable Training Algorithm for dense Retrieval (STAR) and a query-side training Algorithm for Directly Optimizing Ranking pErformance (ADORE), respectively. STAR improves the stability of DR training process by introducing random negatives. ADORE replaces the widely-adopted static hard negative sampling method with a dynamic one to directly optimize the ranking performance. Experimental results on two publicly available retrieval benchmark datasets show that either strategy gains significant improvements over existing competitive baselines and a combination of them leads to the best performance.

## Retrieval Results and Trained Models

| Passage Retrieval | Dev MRR@10  | Dev R@100 | Test NDCG@10 | Files |
|---------------- | ------------|-------| ------- | ------ |
| Inbatch-Neg     | 0.264 | 0.837 | 0.583 | [Model](https://drive.google.com/drive/folders/1ncFKzr4lz9qdI9ZXEi4AosQ_A900ELBz?usp=sharing) |
| Rand-Neg     | 0.301 | 0.853 | 0.612 | [Model](https://drive.google.com/drive/folders/1BJNYcUiFh-Ukc2fibw-3NwPNnAYCZgO9?usp=sharing) |
| STAR     | 0.340 | 0.867 | 0.642 | [Model](https://drive.google.com/drive/folders/1bJw8P15cFiV239mTgFQxVilXMWqzqXUU?usp=sharing) [Train](https://drive.google.com/file/d/1FRFTsqmU95IQHYHJ_q5SDCWVnxbWlIXi/view?usp=sharing) [Dev](https://drive.google.com/file/d/1QkPYmpvI3PX3vWZEO4zsEcoxLipIEsHA/view?usp=sharing) [TRECTest](https://drive.google.com/file/d/1VBPJKKrkZw7RAS5gUJX--lb6Qsu-72D8/view?usp=sharing) |
| ADORE (Inbatch-Neg) | 0.316 | 0.860 | 0.658 | [Model](https://drive.google.com/drive/folders/1Kuwnit7SBoMVZ6s2Mz9RAORQlYU6zG8K?usp=sharing) |
| ADORE (Rand-Neg) | 0.326 | 0.865 | 0.661 | [Model](https://drive.google.com/drive/folders/1U8Nq_LLyVZPh26_ldnSAvsYNk7g43IkE?usp=sharing) |
| ADORE (STAR) | 0.347 | 0.876 | 0.683 | [Model](https://drive.google.com/drive/folders/1C1GQGfI4UHg99rfRcPYzQxlGZXvtsDfm?usp=sharing) [Train](https://drive.google.com/file/d/1zJTPwnUdX_1vkcaQ4SLwtF4L-h24oQOg/view?usp=sharing) [Dev](https://drive.google.com/file/d/1pm4pRimapKZDVqnvLpiYFmgBRKdKlJZW/view?usp=sharing) [TRECTest](https://drive.google.com/file/d/19Vp57INLBszrO6qk6eCfcaH3oY-5HDKR/view?usp=sharing) [Leaderboard]|


| Doc Retrieval | Dev MRR@100  | Dev R@100 | Test NDCG@10 | Files |
|---------------- | ------------|-------| ------- | ------ |
| Inbatch-Neg     | 0.320 | 0.864 | 0.544 | [Model](https://drive.google.com/drive/folders/1wQ6bCH8TjNxazVoKW08DmggPOjO2-Y-H?usp=sharing) |
| Rand-Neg     | 0.330 | 0.859 | 0.572 | [Model](https://drive.google.com/drive/folders/15oGEZbOeqWz0k77R_xE26VBwcEo4VJuE?usp=sharing) |
| STAR     | 0.390 | 0.867 | 0.605 | [Model](https://drive.google.com/drive/folders/18GrqZxeiYFxeMfSs97UxkVHwIhZPVXTc?usp=sharing) [Train](https://drive.google.com/file/d/1Gcp7dbAzpslIIaV_KFkZg4UOkp0ev_eL/view?usp=sharing) [Dev](https://drive.google.com/file/d/17yR6BRLzfW1bxr-VaQ2w2cKpClgcFTej/view?usp=sharing) [TRECTest](https://drive.google.com/file/d/1Mh2BtPYvXRnT2Uz2ZYALZuXQQq3IHApS/view?usp=sharing) |
| ADORE (Inbatch-Neg) | 0.362 | 0.884 | 0.580 | [Model](https://drive.google.com/drive/folders/1Lvz8aLZyzqm9faCWzoQfJJwbL4vdZu2l?usp=sharing) |
| ADORE (Rand-Neg) | 0.361 | 0.885 | 0.585 | [Model](https://drive.google.com/drive/folders/1gH5pTfqBkComxPXhFgdbPRD9r1vZWYZO?usp=sharing) |
| ADORE (STAR) | 0.405 | 0.919 | 0.628 | [Model](https://drive.google.com/drive/folders/1p9FZ8iqqZ9rsfgDzntlNBVKNEf7-2Hih?usp=sharing) [Train](https://drive.google.com/file/d/1XGXdo6LG8VwHvtCbevbvmu3GkwPiV_NC/view?usp=sharing) [Dev](https://drive.google.com/file/d/1vY_q-jzU0CHhOtv_UIYSDkU-l6Kj8fEX/view?usp=sharing) [TRECTest](https://drive.google.com/file/d/1N1ouaMgNdPhjqJMXUBOU5Rg28_T7wapv/view?usp=sharing) [Leaderboard]|

If any links fail or the files go wrong, please contact me or open a issue.

## Requirements

To install requirements, run the following commands:

```setup
git clone git@github.com:jingtaozhan/DRhard.git
cd DRhard
python setup.py install
```
However, you need to set up a new python enverionment for data preprocessing (see below).

## Data Download
To download all the needed data, run:
```
bash download_data.sh
```

## Data Preprocess
You need to set up a new environment with `transformers==2.8.0` to tokenize the text. This is because we find the tokenizer behaves differently among versions 2, 3 and 4. To replicate the results in our paper with our provided trained models, it is necessary to use version `2.8.0` for preprocessing. Otherwise, you may need to re-train the DR models. 

Run the following codes.
```bash
python preprocess.py --data_type 0; python preprocess.py --data_type 1
```

## Inference
With our provided trained models, you can easily replicate our reported experimental results.

Instructions are coming soon.

```bash
python ./star/inference.py --data_type passage --max_doc_length 256 --mode dev   
python ./msmarco_eval.py ./data/passage/preprocess/dev-qrel.tsv ./data/passage/evaluate/star/dev.rank.tsv
```

```bash
python ./adore/inference.py --model_dir ./data/passage/trained_models/adore-star --output_dir ./data/passage/evaluate/adore-star --preprocess_dir ./data/passage/preprocess --mode dev --dmemmap_path ./data/passage/evaluate/star/passages.memmap
python ./msmarco_eval.py ./data/passage/preprocess/dev-qrel.tsv ./data/passage/evaluate/adore-star/dev.rank.tsv
```

```bash
python ./cvt_back.py --input_dir ./data/passage/evaluate/star/ --preprocess_dir ./data/passage/preprocess --output_dir ./data/passage/offcial_runs/star --mode dev
python ./msmarco_eval.py ./data/passage/dataset/qrels.dev.small.tsv ./data/passage/offcial_runs/star/dev.rank.tsv
```

## Train
Instructions are coming soon.