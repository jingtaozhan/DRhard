A previous version of our paper is:
+ Zhan et al.  [Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently.](https://arxiv.org/abs/2010.10469)

If you find this github or our paper helpful, please consider citing us:

## Retrieval Results
Coming soon

## Trained Models
Coming soon

## Data Preprocess
Prepare the datasets from TREC DL Track. The following presents an example about the passage dataset. 
```bash
mkdir data
mkdir data/passage
mkdir data/passage/dataset
cd data/passage/dataset
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xvfz collectionandqueries.tar.gz -C ./
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gzip -d ./msmarco-test2019-queries.tsv.gz 
```
The document dataset can be similarly acquired and should be extracted at `data/doc/dataset`.

You need to set up a new environment with `transformers==2.8.0` to tokenize the text. This is because we find the tokenizer behaves differently among versions 2, 3 and 4. To replicate the results in our paper with our provided trained models, it is necessary to use version 2.8.0 for preprocessing. Otherwise, you may need to re-train the DR models. 

Run the following codes.
```bash
python preprocess.py --data_type 0; python preprocess.py --data_type 1
```

## Inference
With our provided trained models, you can easily replicate our reported experimental results.

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