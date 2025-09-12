from typing import List, Optional, Union
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing
from tokenizers import Tokenizer
from transformers.tokenization_utils_base import BatchEncoding, EncodedInput, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import PaddingStrategy, TensorType
import torch
import numpy as np

def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())
    

class iPLMTokenizer(PreTrainedTokenizerFast):
    def __init__(self, parallel=False, **kwargs):
        super().__init__(tokenizer_object=create_tokenizer_custom(kwargs.get('tokenizer_file')), **kwargs)
        self.add_special_tokens({'pad_token': '<|pad|>'})
        self.parallel = parallel
    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            n_queries = -1, # -1 for vary-length prompt, int with larger than 0 for fix-length, 0 for no prompt
            text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
            text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            text_pair_target: Optional[
                Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
            ] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
        ) -> BatchEncoding:
        
        if not isinstance(text, list):
            text = [text]
            batching = False
        else:
            batching = True
        
        # add prompt
        text_with_prompt = []
        for t in text:
            prompt_length = 0
            if '|' not in t:
                # normal seq
                text_with_prompt.append(t)
            else:
                raw_text = t.split('|')[-1]
                if '_FLAG' in raw_text:
                    flag, raw_text = raw_text.split('_FLAG')
                    assert len(raw_text) > 0, 'WT seq must be given to determine number of queries.'
                else:
                    flag = None
                    
                if n_queries > 0: # use fix length prompt
                    prompt_length = n_queries
                elif n_queries < 0:
                    prompt_length = len(raw_text.replace('1', '').replace('2', ''))
                
                if flag is not None:
                    if flag == 'DPOprompt': # dpo prompt only returns structure path
                        text_with_prompt.append('<|bos|>' * prompt_length)
                    elif flag == 'DPOrejected' or flag == 'DPOchosen': # only returns seq
                        text_with_prompt.append(raw_text)
                    else:
                        raise NotImplementedError(f'undefined flag: {flag}')
                else:
                    text_with_prompt.append('<|bos|>' * prompt_length + raw_text)
        
        batch = super().__call__(
            text=text_with_prompt,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation= truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs
            )

        # add structure ids
        for i in range(len(text)):            
            if '|' not in text[i]:
                continue

            structure_ids = text[i].split('|')[0]
            raw_text = text[i].split('|')[-1]
            for idx, input_ids in enumerate(batch['input_ids'][i]): # for left padding
                if input_ids == 1:
                    starts = idx
                    break


            if '_FLAG' in raw_text:
                flag, raw_text = raw_text.split('_FLAG')
                if flag == 'DPOrejected' or flag == 'DPOchosen': # skip add structure path for dpo response 
                    continue
            if return_tensors is None:
                for j in range(len(structure_ids)):
                    batch['input_ids'][i][starts+j] = ord(structure_ids[j])
            else:
                batch['input_ids'][i, starts:starts+len(structure_ids)] = torch.tensor([ord(c) for c in structure_ids])


        if "token_type_ids" in batch:
            del batch["token_type_ids"]

        if batching:
            return batch
        else:
            return {k:v[0] for k, v in batch.items()}
