from datasets import load_dataset, Dataset, DatasetDict
from dpo_trainer import pDPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed
import pickle
import numpy as np
from tqdm import tqdm
import random
random.seed(0)
import sys

subset = int(sys.argv[1])
tmscore_path = sys.argv[2]
save_name = sys.argv[3]

with open(tmscore_path, 'rb') as f:
    esmfold_tmscore = pickle.load(f)

prompt = []
chosen = []
rejected = []


for rec in tqdm(esmfold_tmscore):
    pdb = rec[0]

    # modify this
    pyd_name = pdb + '.pyd'
    
    # assert len(rec) == 6
    WT_seq = rec[1]
    tmscore_norm1 = rec[2]
    tmscore_norm2 = rec[3]
    tmscore = (np.array(tmscore_norm1) + np.array(tmscore_norm2)) / 2
    # rmsd = rec[4] 
    seq = rec[5]


    
    # subset 
    subset_idx = np.arange(len(tmscore))
    np.random.shuffle(subset_idx)
    subset_idx = subset_idx[:subset]
    tmscore = tmscore[subset_idx]
    seq = [seq[i] for i in subset_idx]

    tm_idx = np.argsort(tmscore)[::-1]
    tmscore_sorted = [tmscore[i] for i in tm_idx]
    seq = [seq[i] for i in tm_idx]

    upper = seq[:subset//2]
    lower = seq[subset//2:]

    prompt += [pyd_name + f"|{WT_seq}"]*(subset//2)
    chosen += upper
    rejected += lower

    
train = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
train_ds = Dataset.from_dict(train)
ds = DatasetDict({"train": train_ds})

tokenizer = AutoTokenizer.from_pretrained("/mnt/aaa/junde/InstructPLM/iplm1_5-small", trust_remote_code=True)

fn_kwargs = {
    "processing_class": tokenizer,
    "max_prompt_length": 512,
    "max_completion_length": 512,
    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
    "add_special_tokens": False,
}

def process(row, **fn_kwargs):
    wt = row["prompt"].split('|')[1]
    prompt = row["prompt"].split('|')[0]
    row['prompt'] = prompt + '|DPOprompt_FLAG1' + wt.rstrip() + '2'
    row["chosen"] = prompt + "|DPOchosen_FLAG1" + row["chosen"].rstrip() + '2'
    row["rejected"] = prompt + "|DPOrejected_FLAG1" + row["rejected"].rstrip() + '2'

    return pDPOTrainer.tokenize_row(row, **fn_kwargs)

train_dataset = ds.map(
    process,
    fn_kwargs=fn_kwargs,
    num_proc=16,
    writer_batch_size=10,
    desc="Tokenizing train dataset",
)
print(train_dataset['train'][0])
train_dataset.save_to_disk(save_name)