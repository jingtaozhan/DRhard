import sys
sys.path.append('./')
import os
import json
import random
import argparse
from tqdm import tqdm
from dataset import load_rank, load_rel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--rel_file", type=str, required=True)
    parser.add_argument("--docsize", type=int, required=True)
    args = parser.parse_args()
    assert not os.path.exists(args.output)
    rank_dict = load_rank(args.input)
    rel_dict = load_rel(args.rel_file)
    print("length", len(rank_dict))
    random.seed(1047)
    query_ids_set = sorted(rank_dict.keys() | rel_dict.keys())
    for k in tqdm(query_ids_set, desc="qids"): # the train query size
        try:
            v = rank_dict[k]
        except KeyError:
            print("key error", k)
            v = random.sample(range(args.docsize), 200)
        if len(v) < 200:
            print("origin", len(v), "<200")
        v = list(filter(lambda x:x not in rel_dict[k], v))
        if len(v) < 200:
            print("remove rel ids", len(v), "<200")
        v = v[:200]
        assert all(0<=x<args.docsize for x in v)
        rank_dict[k] = v
        
    json.dump(rank_dict, open(args.output, 'w'))
        