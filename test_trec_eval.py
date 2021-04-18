import os
import sys
import tempfile
import subprocess

def compute_metrics_from_files(path_to_reference, path_to_candidate, trec_eval_bin_path):
    trec_run_fd, trec_run_path = tempfile.mkstemp(text=True)
    try:
        with os.fdopen(trec_run_fd, 'w') as tmp:
            for line in open(path_to_candidate):
                qid, pid, rank = line.split()
                rank = int(rank)
                tmp.write(f"{qid} Q0 {pid} {rank} {1/rank} System\n")
        result = subprocess.check_output([
            trec_eval_bin_path, "-c", "-mndcg_cut.10", path_to_reference, trec_run_path])
        print(result)
    finally:
        os.remove(trec_run_path)
        

def main():
    """Command line:
    python test_trec_eval.py <path_to_reference_file> <path_to_candidate_file>
    """
    print("Eval Started")
    if len(sys.argv) == 3:
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]
        trec_eval_bin_path = "./data/trec_eval"
        assert os.path.exists(trec_eval_bin_path)
        compute_metrics_from_files(path_to_reference, path_to_candidate, trec_eval_bin_path)
    else:
        print('Usage: test_trec_eval.py <reference ranking> <candidate ranking>')
        exit()
    
if __name__ == '__main__':
    main()