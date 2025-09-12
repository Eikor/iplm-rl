from tmtools.io import get_structure, get_residue_data
import os
import glob
import numpy as np
from tmtools import tm_align
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
from functools import partial
import concurrent
import sys


def process_single_id(pdb_id, gt_root, pred_root):


    try:
        gt_file = os.path.join(gt_root, f"{pdb_id}.pdb")
        gt = get_structure(gt_file)
        gt_coords, gt_seq = get_residue_data(next(gt.get_chains()))
        nan_mask = ~np.isnan(gt_coords.sum(axis=-1))
        gt_coords = gt_coords[nan_mask]
        gt_seq_no_nan = ''.join([gt_seq[i] for i, valid in enumerate(nan_mask) if valid])

        pred_dir = os.path.join(pred_root, pdb_id)
        try:
            pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.pdb")), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        except Exception as e:
            print(glob.glob(os.path.join(pred_dir, "*.pdb")))

        results = []
        pred_seqs = []  
        for pred_file in pred_files:
            try:
                s = get_structure(pred_file)
                pred_coords, pred_seq = get_residue_data(next(s.get_chains()))
                pred_seqs.append(pred_seq)

                output = tm_align(pred_coords,
                                gt_coords,
                                pred_seq,
                                gt_seq_no_nan)
                results.append((output.tm_norm_chain1, output.tm_norm_chain2, output.rmsd))#, recovery))
            except Exception as e:
                print(f"Pred error {pred_file}: {str(e)}")
                results.append((0, 0, 0))

        tmscores_norm1 = [x[0] for x in results]
        tmscores_norm2 = [x[1] for x in results]
        rmsds = [x[2] for x in results]
        return (pdb_id, gt_seq, tmscores_norm1, tmscores_norm2, rmsds, pred_seqs)
    
    except Exception as e:
        print(f"ID error {pdb_id}: {str(e)}")
        return (pdb_id, gt_seq, [0], [0], [0], pred_seqs)

if __name__ == "__main__":
    # args
    gt_pdb_path = sys.argv[1]
    pred_pdb_path = sys.argv[2]
    save_name = sys.argv[3]

    
    pdb_ids = sorted([d for d in os.listdir(pred_pdb_path)
              if os.path.isdir(os.path.join(pred_pdb_path, d))])
    print(f"Found {len(pdb_ids)} PDB IDs to process")
    
    worker = partial(process_single_id,
                    gt_root=gt_pdb_path,
                    pred_root=pred_pdb_path)
    
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker, pid): pid for pid in pdb_ids}
        
        with tqdm(total=len(pdb_ids), desc="Processing") as pbar:
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                results.append(res)
                pbar.update(1)
                pbar.set_postfix_str(f"Recent ID: {res[0]}")

    # sort
    id_order = {pid:i for i, pid in enumerate(pdb_ids)}
    results.sort(key=lambda x: id_order[x[0]])
    
    # save results
    with open(f"{save_name}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {save_name}.pkl")
    
    max_tmscore = [np.max(results[i][3]) for i in range(len(results))]
    mean_tmscore_norm1 = [np.mean(results[i][2]) for i in range(len(results))]
    mean_tmscore_norm2 = [np.mean(results[i][3]) for i in range(len(results))]
    mean_rmsd = [np.mean(results[i][4]) for i in range(len(results))]

    print(f'max tmscore: {np.mean(max_tmscore)}')
    print(f'mean tmscore norm1: {np.mean(mean_tmscore_norm1)}')
    print(f'mean tmscore norm2: {np.mean(mean_tmscore_norm2)}')
    print(f'mean rmsd: {np.mean(mean_rmsd)}')
