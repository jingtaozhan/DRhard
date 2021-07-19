import sys
sys.path.append('./')
import os
import json
import random
import argparse
import faiss
import logging
import numpy as np
import torch
import subprocess
from transformers import RobertaConfig
from tqdm import tqdm
from model import RobertaDot
from dataset import load_rank, load_rel
from retrieve_utils import (
    construct_flatindex_from_embeddings, 
    index_retrieve, convert_index_to_gpu
)
from star.inference import doc_inference, query_inference
logger = logging.Logger(__name__)


def retrieve_top(args):
    config = RobertaConfig.from_pretrained(args.model_path, gradient_checkpointing=False)
    model = RobertaDot.from_pretrained(args.model_path, config=config)
    output_embedding_size = model.output_embedding_size
    model = model.to(args.device)
    query_inference(model, args, output_embedding_size)
    doc_inference(model, args, output_embedding_size)
    
    model = None
    torch.cuda.empty_cache()

    doc_embeddings = np.memmap(args.doc_memmap_path, 
        dtype=np.float32, mode="r")
    doc_ids = np.memmap(args.docid_memmap_path, 
        dtype=np.int32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, output_embedding_size)

    query_embeddings = np.memmap(args.query_memmap_path, 
        dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, output_embedding_size)
    query_ids = np.memmap(args.queryids_memmap_path, 
        dtype=np.int32, mode="r")

    index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids)
    if torch.cuda.is_available() and not args.not_faiss_cuda:
        index = convert_index_to_gpu(index, list(range(args.n_gpu)), False)
    else:
        faiss.omp_set_num_threads(32)
    nearest_neighbors = index_retrieve(index, query_embeddings, args.topk + 10, batch=320)

    with open(args.output_rank_file, 'w') as outputfile:
        for qid, neighbors in zip(query_ids, nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid}\t{pid}\t{idx+1}\n")


def gen_static_hardnegs(args):
    rank_dict = load_rank(args.output_rank_file)
    rel_dict = load_rel(args.label_path)
    query_ids_set = sorted(rel_dict.keys())
    for k in tqdm(query_ids_set, desc="gen hard negs"): 
        v = rank_dict[k]
        v = list(filter(lambda x:x not in rel_dict[k], v))
        v = v[:args.topk]
        assert len(v) == args.topk
        rank_dict[k] = v
    json.dump(rank_dict, open(args.output_hard_path, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", choices=["passage", 'doc'], type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "lead"], required=True)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--not_faiss_cuda", action="store_true")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    args.preprocess_dir = f"./data/{args.data_type}/preprocess"
    args.model_path = f"./data/{args.data_type}/warmup"
    args.output_dir = f"./data/{args.data_type}/warmup_retrieve"
    args.label_path = os.path.join(args.preprocess_dir, f"{args.mode}-qrel.tsv")

    args.query_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")
    args.queryids_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query-id.memmap")
    args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")
    args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")

    args.output_rank_file = os.path.join(args.output_dir, f"{args.mode}.rank.tsv")
    args.output_hard_path = os.path.join(args.output_dir, "hard.json")
    
    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)

    retrieve_top(args)
    gen_static_hardnegs(args)
    
    results = subprocess.check_output(["python", "msmarco_eval.py", args.label_path, args.output_rank_file])
    print(results)