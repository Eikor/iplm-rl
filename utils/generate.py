from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
import pickle
import argparse
import math
GPU_ID = os.environ['CUDA_VISIBLE_DEVICES']
def prepare_inputs(pyd_name, tokenizer, n_queries, fix_residue=(None, None), text=None, device='cuda'):
    pyd_name = pyd_name.split('/')[-1]
    # 1 start position
    residues, positions = fix_residue
    seq_for_tokenize = pyd_name+'|'
    return_position = False

    if residues is not None and positions is not None:
        seq_for_tokenize = seq_for_tokenize + residues
        return_position = True    

    if text is not None:
        seq_for_tokenize = seq_for_tokenize + '1' + text + '2'
        
    else:
        seq_for_tokenize = seq_for_tokenize + '1'

    encoded = tokenizer([seq_for_tokenize], return_tensors='pt', n_queries=n_queries).to(device)
    labels = encoded.input_ids.clone()
    
    # modify labels
    if n_queries > 0:
        labels[:, :n_queries+1] = -100
    if residues is not None:
        labels[:, np.maximum(n_queries, 0):np.maximum(n_queries, 0) + len(residues)+1] = -100

    # create position_ids
    if positions is not None:
        positions = positions + np.maximum(n_queries, 0) + 1 # start from 1
        position_ids = torch.arange(0, np.maximum(n_queries, 0)) # structure pos
        position_ids = torch.cat([position_ids, torch.tensor(positions)]) # fix residue pos
        if text is not None:
            text_pos = torch.arange(np.maximum(n_queries, 0), len(text) + 2)
        else:
            text_pos = torch.tensor([np.maximum(n_queries, 0)])
        position_ids = torch.cat([position_ids, text_pos]).unsqueeze(0).to(device=device)

    if not return_position:
        position_ids = None

    return encoded.input_ids.view([1, -1]), labels, encoded.attention_mask.view([1, -1]), position_ids

def truncate_seq(text):
    bos = text.find('1')
    eos = text.find('2')
    if eos > bos and bos >= 0:
        return text[bos+1:eos]
    else:
        return text[bos+1:]

def get_args():
    parser = argparse.ArgumentParser(description='generate args')

    parser.add_argument('--structure_path', type=str, default='/home/junde/datasets/test_pyd/fix_residue_test')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--anno_file', type=str, default=None)
    parser.add_argument('--adapter_path', type=str, default=None)

    parser.add_argument('--fix-length', action='store_true')
    parser.add_argument('--fix-residue', action='store_true')
    parser.add_argument('--num_fixed_residue', type=int, default=0)
    parser.add_argument('--total', type=int, default=100, help='total number of designed sequences')
    parser.add_argument('--num_return_sequences', type=int, default=2, help='number of sequences per round')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--max_length', type=int, default=512)

    parser.add_argument('--save_prefix', type=str, default='debug', help='save path')
    parser.add_argument('--save_suffix', type=str, default='res1', help='save suffix')
    args = parser.parse_args()
    return args


def run_design(model, tokenizer, 
               structure_path,
               anno_file=None,
               total=1000, 
               fix_length=False, 
               fix_residue=False,
               num_fixed_residue=0,
               max_length=512, 
               t=0.8, p=0.9, 
               repetition_penalty=1.0, 
               num_return_sequences=10, 
               save_prefix='res', 
               save_suffix=''):

    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix, exist_ok=True)
    
    if os.path.exists(structure_path):
        structure_emb_path = glob(os.path.join(structure_path, '*.pyd'))
        if anno_file is not None:
            with open(anno_file, 'r') as f:
                anno = f.readlines()
            structure_emb_path = [os.path.join(structure_path, rec.split('|')[0]) for rec in anno]

        if len(structure_emb_path) < 1:
            print(f'no preprocessed structure embedding found at {structure_path}')
            exit()
    else:
        print(f'no preprocessed structure embedding found in {structure_path}')
        exit()
    
    print('-------------------------- run design -----------------------------')
    for s in structure_emb_path:
        save_name = s.split('/')[-1].split('.')[0] + '_' + save_suffix
        if os.path.exists(f'{save_prefix}/{save_name}.fasta'):
            print(f'GPU {GPU_ID}: skip {save_name}')
            continue
        else:
            with open(f'{save_prefix}/{save_name}.fasta', 'w') as f:
                f.writelines('')
        print(s)

        with open(s, 'rb') as f:
            pyd = pickle.load(f)

        if pyd['seq'] is not None:
            n_queries=len(pyd['seq'])
            seq_length = len(pyd['seq']) + 1
            
            if seq_length > 510:
                print('overlenth, skip')
                continue

            input_ids, labels, attn_mask, position_ids = prepare_inputs(s, tokenizer, n_queries, text=pyd['seq'], device=model.device)
            with torch.no_grad():
                loss = model(input_ids=input_ids, labels=labels, attention_mask=attn_mask, position_ids=position_ids).loss.item()
            print(f'calculate {s} ref seq loss: {loss}')
            print(f'seq_length: {seq_length} ')

        else:
            max_length_for_generate = max_length
            min_new_tokens = 5


        res = []
        score = []
        pbar = tqdm(total=total, desc=f'generate {s}')

        residues = (None, None) 
        input_ids, labels, attn_mask, position_ids = prepare_inputs(s, tokenizer, n_queries, fix_residue=residues, device=model.device)
        if fix_length:
            max_length_for_generate = n_queries + seq_length
            min_new_tokens = seq_length -1
        else:
            max_length_for_generate = max_length + n_queries
            min_new_tokens = 5

        if residues[0] is not None:
            max_length_for_generate += len(residues[0])

        print(position_ids)
        while len(res) < total:
            with torch.no_grad():
                # use inputs for peft model and automodel 
                returns = model.generate(
                    inputs=input_ids, 
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    do_sample=True,
                    temperature=t, 
                    use_cache=True,
                    max_length=max_length_for_generate,
                    min_new_tokens=min_new_tokens,
                    top_p=p, 
                    num_return_sequences=num_return_sequences, 
                    pad_token_id=0, repetition_penalty=repetition_penalty, 
                    # return_dict_in_generate=True, output_scores=True,
                    bad_words_ids=[[3]] if not fix_length else [[3], [4]]
                    )
            texts = tokenizer.batch_decode(returns)

            for text in texts:
                text = truncate_seq(text)
                if text is not None: # and text not in res:
                    pbar.update(1)
                    res.append(text)
        
        pbar.close()
        scores = np.arange(len(res))
        # with torch.no_grad():
        #     for text in tqdm(res, desc='calculate score'):
        #         input_ids, labels, attn_mask, position_ids = prepare_inputs(s, tokenizer, n_queries, text=text, device=model.device)
        #         score.append(model(input_ids=input_ids, labels=labels, attention_mask=attn_mask).loss.item())

        print('---------------------------------------------------------------')
        save_name = s.split('/')[-1].split('.')[0] + '_' + save_suffix
        with open(f'{save_prefix}/{save_name}.fasta', 'w') as f:
            for i in np.argsort(scores)[:total]:
                f.writelines(f'>{scores[i]}\n'+res[i]+'\n')


if __name__ =='__main__':

    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, n_queries=-1)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, structure={
            "embedding_keys" : ['esm_if_emb', 'chroma_emb_design', 'prorefiner_emb'], 
            "width": 768,
            "output_dim": 4096, # set output dim based on args.model_name
            "max_seqlen": 512,
            "n_queries": -1,
            "num_heads": 16,
            "structure_emb_path_prefix": args.structure_path,
            "projector": 'mlp'
        })
    if args.adapter_path is not None and args.adapter_path != '-':
        model.load_adapter(args.adapter_path)
    model.cuda().eval()
    model.requires_grad_(False)

    run_design(model, tokenizer, 
               structure_path=args.structure_path,
               anno_file=args.anno_file,
               total=args.total, 
               fix_length=args.fix_length,
               fix_residue=args.fix_residue,
               num_fixed_residue=args.num_fixed_residue,
               max_length=args.max_length, 
               t=args.temperature, 
               p=args.top_p, 
               repetition_penalty=args.repetition_penalty, 
               num_return_sequences=args.num_return_sequences, 
               save_prefix=args.save_prefix, 
               save_suffix=args.save_suffix,)
