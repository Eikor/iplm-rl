# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from accelerate import PartialState
import logging
import multiprocessing
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)
from utils.data import PairDatasetForDPO

import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed
from peft import PeftModel, PeftConfig
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    TrlParser,
    ScriptArguments
)
from dpo_trainer import pDPOTrainer
@dataclass
class DPOScriptArguments(ScriptArguments):
    peft_resume: Optional[str] = None
    sft: Optional[bool] = False
    structure_emb_path: Optional[str] = None

if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    training_args.dataset_num_proc = 4
    training_args.save_only_model = True
    training_args.sft = args.sft
    set_seed(training_args.seed)

    # print(args)
    # print(training_args)
    # print(model_config)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    structure_config = AutoConfig.from_pretrained(model_config.model_name_or_path, trust_remote_code=True).structure
    print(structure_config)
    structure_config['structure_emb_path_prefix'] = args.structure_emb_path

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        structure=structure_config
    )

    if args.peft_resume is not None:
        peft_config = PeftConfig.from_pretrained(args.peft_resume)
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
        model = PeftModel.from_pretrained(
            model, args.peft_resume,
            is_trainable=True 
        )
        # check if it's working
        model.print_trainable_parameters()        
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
        model_ref.load_adapter(args.peft_resume)
        training_args.force_use_ref_model = True
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
        peft_config = get_peft_config(model_config)
        model_ref = None
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)

    ################
    # Dataset
    ################
    train_dataset = load_from_disk(args.dataset_name)["train"]

    ################
    # Training
    ################
    trainer = pDPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
    )
        
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.save_model(training_args.output_dir)
