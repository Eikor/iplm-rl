import torch
import glob
import random
import os
import sys
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

anno_path = sys.argv[1]
fasta_path = sys.argv[2]
save_path = sys.argv[3]

output_dir = './'
GPU_ID = os.environ['CUDA_VISIBLE_DEVICES']
# GPU_ID = 0

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
model = model.eval().cuda()
model.esm = model.esm.half()
torch.backends.cuda.matmul.allow_tf32 = True


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def extract_pdbid(fasta_name):
    # modify this function for different naming rules
    fasta_name = fasta_name.split('/')[-1].split('.')[0]
    if '_' in fasta_name:
        try:
            pdbid = fasta_name.split('_')[1]
            chain = fasta_name.split('_')[2]
        except:
            return fasta_name
        return pdbid + '.' + chain
    else:
        return fasta_name

def read_fasta(url, num_returns=20):
    # read fasta file and randomly return num_returns sequences in a list
    # if seqs in fasta file is less than num_returns, 
    # this will return all seqs in fasta file

    ret = []
    if not os.path.exists(url):
        return ret
    
    with open(url, 'r') as f:
        for l in f:
            if l.startswith('>'):
                continue
            else:
                ret.append(l)
    return ret

def load_seq(url):
    # each structure has multiple designed sequences in one fasta file
    fastas = sorted(glob.glob(url+'/*.fasta'))
    sequence = {}

    for f in fastas:
        seq_name = extract_pdbid(f)
        sequence[seq_name] = read_fasta(f)

    return sequence


def infer_pdb_only(model, fasta_path, save_path_prefix):

    fasta_path = glob.glob(fasta_path+'/*.fasta')

    for fp in fasta_path:
        pdbid = extract_pdbid(fp)
        seq = read_fasta(fp)
        if len(seq) == 0:
            with open(f'{output_dir}GPU_{GPU_ID}.log', 'a') as f:
                f.write(f'GPU {GPU_ID}: empty {pdbid}, abort\n')
            exit()

        save_path = os.path.join(save_path_prefix, pdbid)
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except:
                with open(f'{output_dir}GPU_{GPU_ID}.log', 'a') as f:
                    f.write(f'GPU {GPU_ID}: skip {pdbid} due to mkdir err\n')
                continue

            with open(f'{output_dir}GPU_{GPU_ID}.log', 'a') as f:
                f.write(f'GPU {GPU_ID}: working on {pdbid}\n')
            
            with torch.no_grad():
                outputs = []
                
                for s in seq:
                    # hf infer
                    input_ids = tokenizer(s, add_special_tokens=False)['input_ids']
                    tokenized_input = torch.tensor([token for token in input_ids if token is not None]).unsqueeze(0).cuda()
                    with torch.no_grad():
                        output = convert_outputs_to_pdb(model(tokenized_input))
                    outputs.append(output)
                
                for i in range(len(outputs)):
                    with open(save_path+f'/{i}.pdb', "w") as f:
                        f.write(''.join(outputs[i]))
        else:
            with open(f'{output_dir}GPU_{GPU_ID}.log', 'a') as f:
                f.write(f'GPU {GPU_ID}: skip {pdbid} due to path exist\n')
            continue

infer_pdb_only(model, fasta_path, save_path)
