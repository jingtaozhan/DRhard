# coding=utf-8
import argparse
import subprocess
import sys
sys.path.append("./")
import faiss
import logging
import os
import numpy as np
import torch
from transformers import RobertaConfig
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from model import RobertaDot
from dataset import (
    TextTokenIdsCache, load_rel, SubsetSeqDataset, SequenceDataset,
    single_get_collate_function
)
from retrieve_utils import (
    construct_flatindex_from_embeddings, 
    index_retrieve, convert_index_to_gpu
)
logger = logging.Logger(__name__)


def prediction(model, data_collator, args, test_dataset, embedding_memmap, ids_memmap, is_query):
    os.makedirs(args.output_dir, exist_ok=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size*args.n_gpu,
        collate_fn=data_collator,
        drop_last=False,
    )
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    write_index = 0
    for step, (inputs, ids) in enumerate(tqdm(test_dataloader)):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            logits = model(is_query=is_query, **inputs).detach().cpu().numpy()
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index+write_size] = logits
        ids_memmap[write_index:write_index+write_size] = ids
        write_index += write_size
    assert write_index == len(embedding_memmap) == len(ids_memmap)


def query_inference(model, args, embedding_size):
    if os.path.exists(args.query_memmap_path):
        print(f"{args.query_memmap_path} exists, skip inference")
        return
    query_collator = single_get_collate_function(args.max_query_length)
    query_dataset = SequenceDataset(
        ids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix=f"{args.mode}-query"),
        max_seq_length=args.max_query_length
    )
    query_memmap = np.memmap(args.query_memmap_path, 
        dtype=np.float32, mode="w+", shape=(len(query_dataset), embedding_size))
    queryids_memmap = np.memmap(args.queryids_memmap_path, 
        dtype=np.int32, mode="w+", shape=(len(query_dataset), ))
    try:
        prediction(model, query_collator, args,
                query_dataset, query_memmap, queryids_memmap, is_query=True)
    except:
        subprocess.check_call(["rm", args.query_memmap_path])
        subprocess.check_call(["rm", args.queryids_memmap_path])
        raise


def doc_inference(model, args, embedding_size):
    if os.path.exists(args.doc_memmap_path):
        print(f"{args.doc_memmap_path} exists, skip inference")
        return
    doc_collator = single_get_collate_function(args.max_doc_length)
    ids_cache = TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passages")
    subset=list(range(len(ids_cache)))
    doc_dataset = SubsetSeqDataset(
        subset=subset,
        ids_cache=ids_cache,
        max_seq_length=args.max_doc_length
    )
    assert not os.path.exists(args.doc_memmap_path)
    doc_memmap = np.memmap(args.doc_memmap_path, 
        dtype=np.float32, mode="w+", shape=(len(doc_dataset), embedding_size))
    docid_memmap = np.memmap(args.docid_memmap_path, 
        dtype=np.int32, mode="w+", shape=(len(doc_dataset), ))
    try:
        prediction(model, doc_collator, args,
            doc_dataset, doc_memmap, docid_memmap, is_query=False
        )
    except:
        subprocess.check_call(["rm", args.doc_memmap_path])
        subprocess.check_call(["rm", args.docid_memmap_path])
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", choices=["passage", 'doc'], type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "lead"], required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--faiss_gpus", type=int, default=None, nargs="+")
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    args.preprocess_dir = f"./data/{args.data_type}/preprocess"
    args.model_path = f"./data/{args.data_type}/trained_models/star"
    args.output_dir = f"./data/{args.data_type}/evaluate/star"
    args.query_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")
    args.queryids_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query-id.memmap")
    args.output_rank_file = os.path.join(args.output_dir, f"{args.mode}.rank.tsv")
    args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")
    args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")
    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)

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
    if args.faiss_gpus:
        index = convert_index_to_gpu(index, args.faiss_gpus, False)
    else:
        faiss.omp_set_num_threads(32)
    nearest_neighbors = index_retrieve(index, query_embeddings, args.topk, batch=32)

    with open(args.output_rank_file, 'w') as outputfile:
        for qid, neighbors in zip(query_ids, nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid}\t{pid}\t{idx+1}\n")


if __name__ == "__main__":
    main()
