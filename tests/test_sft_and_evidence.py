from dataclasses import replace

import pytest

from posttrain.training.sft_data import encode_answer,pad_batch
from posttrain.training.evidence import require_real_pilot,validate_freeze


class FakeTokenizer:
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict=False):
        return [1,2,3] if add_generation_prompt else [1,2,3,4,5,6]


def test_assistant_only_labels_and_padding(problem):
    p=replace(problem,validation_status='reference_passed')
    sample=encode_answer(FakeTokenizer(),p,10)
    assert sample['labels']==[-100,-100,-100,4,5,6]
    batch=pad_batch([sample,{'input_ids':[1,4],'attention_mask':[1,1],'labels':[-100,4]}],0)
    assert batch['labels'][1]==[-100,4,-100,-100,-100,-100]
    assert batch['attention_mask'][1]==[1,1,0,0,0,0]
    with pytest.raises(ValueError):encode_answer(FakeTokenizer(),p,5)
    with pytest.raises(ValueError):encode_answer(FakeTokenizer(),problem,10)


def test_cannot_freeze_mock_or_single_update(tmp_path):
    with pytest.raises(ValueError):require_real_pilot({'evidence_level':'mock_contract'})
    with pytest.raises(ValueError):require_real_pilot({'evidence_level':'real_7b_pilot','model_id':'Qwen/Qwen2.5-Coder-7B-Instruct','stable_complete_steps':1})
    with pytest.raises(ValueError):validate_freeze({},tmp_path)


def test_sft_handles_v5_dictionary_default_by_requesting_flat_tokens(problem):
    class V5Tokenizer:
        def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict=True):
            ids=[1,2,3] if add_generation_prompt else [1,2,3,4,5]
            return {'input_ids':ids,'attention_mask':[1]*len(ids)} if return_dict else ids
    sample=encode_answer(V5Tokenizer(),replace(problem,validation_status='reference_passed'),10)
    assert sample['input_ids']==[1,2,3,4,5]
    assert sample['labels']==[-100,-100,-100,4,5]
