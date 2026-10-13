"""SFT microbatching contract tests. They import no torch and execute no model."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from posttrain.common import canonical
from posttrain.training import sft_data
from posttrain.training.sft import checked_config
from posttrain.training.sft_data import (data_length_report,encode_examples,length_distribution,
    microbatch_loss_weights,microbatch_token_counts,pad_batch,partition_micro_batches,
    resolve_pad_token_id,shifted_target_tokens)


class PrefixTokenizer:
    """Template with a 3-token prompt and a 3-token assistant answer."""

    def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict=False):
        return [1,2,3] if add_generation_prompt else [1,2,3,4,5,6]


def config(**overrides):
    cfg={'model_id':'Qwen/Qwen2.5-Coder-7B-Instruct','model_revision':'a'*40,'dtype':'bfloat16',
         'max_steps':20,'batch_size':4,'max_length':2048,'seed':42,'learning_rate':2e-5,
         'lora_rank':32,'lora_alpha':64,'save_steps':[5,10,15,20]}
    cfg.update(overrides)
    return cfg


def samples():
    # shifted targets (labels[1:] without ignore_index): 3, 2 and 4 tokens
    return [{'input_ids':[1,2,3,4,5],'attention_mask':[1]*5,'labels':[-100,-100,3,4,5]},
            {'input_ids':[1,2,3],'attention_mask':[1]*3,'labels':[-100,2,3]},
            {'input_ids':[1,2,3,4,5,6,7],'attention_mask':[1]*7,'labels':[-100,-100,-100,4,5,6,7]}]


def test_partition_micro_batches_keeps_every_example_and_permits_uneven_tail():
    assert [len(batch) for batch in partition_micro_batches([0,1,2,3,4],2)]==[2,2,1]
    assert [index for batch in partition_micro_batches([0,1,2,3,4],2) for index in batch]==[0,1,2,3,4]
    assert [len(batch) for batch in partition_micro_batches([0,1,2,3],3)]==[3,1]
    assert [len(batch) for batch in partition_micro_batches([0,1,2,3],4)]==[4]
    with pytest.raises(ValueError,match='invalid_micro_batch_size'):
        partition_micro_batches([0],0)
    with pytest.raises(ValueError,match='empty_optimizer_update'):
        partition_micro_batches([],1)


def test_config_accepts_optional_micro_batch_size_and_rejects_invalid_values():
    cfg=checked_config(config())
    assert cfg['micro_batch_size']==1 and cfg['batch_size']==4
    assert checked_config(config(micro_batch_size=2))['micro_batch_size']==2
    assert checked_config(config(micro_batch_size=4))['micro_batch_size']==4
    for invalid in (0,-1,5,1.5,'2',True,None):
        with pytest.raises(ValueError,match='invalid_micro_batch_size'):
            checked_config(config(micro_batch_size=invalid))


def test_pilot_config_declares_micro_batch_size_two_for_batch_size_four():
    path=Path(__file__).resolve().parents[1]/'configs/sft_pilot.json'
    cfg=checked_config(json.loads(path.read_text(encoding='utf-8')))
    assert cfg['micro_batch_size']==2 and cfg['batch_size']==4


def test_token_counts_cover_padding_targets_and_microbatch_count():
    counts=microbatch_token_counts(samples(),partition_micro_batches([0,1,2],2))
    assert counts['micro_batch_count']==2
    assert [record['size'] for record in counts['micro_batches']]==[2,1]
    assert [record['width'] for record in counts['micro_batches']]==[5,7]
    assert counts['target_tokens']==9
    assert [record['target_tokens'] for record in counts['micro_batches']]==[5,4]
    assert counts['nonpadding_tokens']==15
    assert counts['padding_tokens']==2
    assert counts['processed_tokens']==17
    assert counts['padding_fraction']==round(2/17,6)
    unpadded=microbatch_token_counts(samples(),partition_micro_batches([0,1,2],1))
    assert unpadded['padding_tokens']==0 and unpadded['micro_batch_count']==3
    assert unpadded['target_tokens']==counts['target_tokens']
    whole=microbatch_token_counts(samples(),partition_micro_batches([0,1,2],3))
    assert whole['padding_tokens']==6 and whole['processed_tokens']==21


def test_microbatch_weights_normalize_over_the_entire_optimizer_update():
    counts=microbatch_token_counts(samples(),partition_micro_batches([0,1,2],2))
    weights=microbatch_loss_weights(counts)
    assert weights==[5/9,4/9] and sum(weights)==pytest.approx(1.0)
    assert microbatch_loss_weights(microbatch_token_counts(samples(),partition_micro_batches([0,1,2],3)))==[1.0]
    per_token=[[1.0,2.0,3.0],[4.0,6.0],[1.0,1.0,1.0,1.0]]
    flat=[value for row in per_token for value in row]
    micro_means=[sum(value for index in batch for value in per_token[index])/
                 sum(len(per_token[index]) for index in batch)
                 for batch in partition_micro_batches([0,1,2],2)]
    assert sum(weight*mean for weight,mean in zip(weights,micro_means))==pytest.approx(sum(flat)/len(flat))
    with pytest.raises(ValueError,match='empty_optimizer_update_targets'):
        microbatch_loss_weights({'target_tokens':0,'micro_batches':[{'target_tokens':0}]})


def test_right_padding_masks_prompt_tokens_and_pad_positions_from_targets():
    long={'input_ids':[1,2,3,4],'attention_mask':[1,1,1,1],'labels':[-100,-100,3,4]}
    short={'input_ids':[1,2],'attention_mask':[1,1],'labels':[-100,2]}
    padded=pad_batch([long,short],0)
    assert padded['input_ids']==[[1,2,3,4],[1,2,0,0]]
    assert padded['attention_mask']==[[1,1,1,1],[1,1,0,0]]
    assert padded['labels']==[[-100,-100,3,4],[-100,2,-100,-100]]
    assert [shifted_target_tokens({'labels':row}) for row in padded['labels']]==[2,1]
    counts=microbatch_token_counts([long,short],[[0,1]])
    assert (counts['nonpadding_tokens'],counts['padding_tokens'],counts['target_tokens'])==(6,2,3)
    assert counts['padding_fraction']==round(2/8,6)


def test_pad_token_uses_tokenizer_pad_token_and_falls_back_to_eos():
    class Padded:
        pad_token_id=5;eos_token_id=9
    class Unpadded:
        pad_token_id=None;eos_token_id=9
    class Neither:
        pad_token_id=None;eos_token_id=None
    assert resolve_pad_token_id(Padded())==(5,'pad_token')
    assert resolve_pad_token_id(Unpadded())==(9,'eos_token_fallback')
    with pytest.raises(ValueError,match='tokenizer_pad_token_and_eos_missing'):
        resolve_pad_token_id(Neither())
    long={'input_ids':[1,2,3,4],'attention_mask':[1,1,1,1],'labels':[-100,-100,3,4]}
    short={'input_ids':[1,2],'attention_mask':[1,1],'labels':[-100,2]}
    pad_token_id,source=resolve_pad_token_id(Unpadded())
    assert pad_batch([long,short],pad_token_id)['input_ids'][1]==[1,2,9,9]


def test_encode_examples_rejects_overlong_examples_without_truncating_targets(problem):
    valid=replace(problem,validation_status='reference_passed')
    accepted,rejected=encode_examples(PrefixTokenizer(),[valid],5)
    assert accepted==[] and rejected==[{'problem_id':valid.problem_id,'reason':'example_exceeds_context_limit'}]
    accepted,rejected=encode_examples(PrefixTokenizer(),[valid],6)
    assert rejected==[] and accepted[0][0]==valid.problem_id
    assert accepted[0][1]['input_ids']==[1,2,3,4,5,6]
    assert accepted[0][1]['labels']==[-100,-100,-100,4,5,6]
    assert shifted_target_tokens(accepted[0][1])==3
    with pytest.raises(ValueError,match='isolated reference backtest'):
        encode_examples(PrefixTokenizer(),[problem],6)


def test_examples_with_zero_shifted_target_tokens_are_rejected(problem,monkeypatch):
    valid=replace(problem,validation_status='reference_passed')
    assert shifted_target_tokens({'labels':[-100,-100]})==0
    # Degenerate template: empty generation prompt and one single assistant token
    # leave nothing to supervise after the causal shift.
    monkeypatch.setattr(sft_data,'chat_tokens',
        lambda tokenizer,messages,*,add_generation_prompt:[] if add_generation_prompt else [7])
    with pytest.raises(ValueError,match='empty_shifted_target'):
        sft_data.encode_answer(None,valid,16)
    accepted,rejected=encode_examples(None,[valid],16)
    assert accepted==[]
    assert rejected==[{'problem_id':valid.problem_id,'reason':'empty_shifted_target'}]


def test_data_length_report_counts_rejections_and_distributions_without_references(problem):
    valid=replace(problem,validation_status='reference_passed')
    accepted,_=encode_examples(PrefixTokenizer(),[valid,replace(valid,problem_id='p2')],6)
    rejected=[{'problem_id':'p3','reason':'example_exceeds_context_limit'}]
    report=data_length_report({'max_length':2048,'batch_size':4,'micro_batch_size':2},
        [sample for _,sample in accepted],rejected,'eos_token_fallback')
    assert report['accepted_examples']==2 and report['rejected_examples']==1
    assert report['rejected_by_reason']=={'example_exceeds_context_limit':1}
    assert report['targets_truncated'] is False
    assert report['truncation_policy']=='reject_examples_beyond_max_length'
    assert report['pad_token_source']=='eos_token_fallback'
    assert report['batch_size']==4 and report['micro_batch_size']==2
    assert report['total_tokens']=={'count':2,'min':6,'max':6,'mean':6.0,'median':6,'p90':6,'bucket':128,
        'histogram':[{'start':0,'end':127,'count':2}]}
    assert report['assistant_tokens']['count']==2 and report['assistant_tokens']['max']==3
    assert report['shifted_target_tokens']['mean']==3.0
    text=canonical(report)
    assert valid.references[0] not in text
    assert 'input_ids' not in text and 'labels' not in text
    assert 'p1' not in text and 'p2' not in text and 'p3' not in text


def test_length_distribution_buckets_and_empty_input():
    distribution=length_distribution([130,4,4,127])
    assert distribution['count']==4 and distribution['min']==4 and distribution['max']==130
    assert distribution['median']==127 and distribution['p90']==130
    assert distribution['histogram']==[{'start':0,'end':127,'count':3},{'start':128,'end':255,'count':1}]
    assert length_distribution([])=={'count':0,'min':0,'max':0,'mean':0.0,'median':0,'p90':0,'bucket':128,'histogram':[]}
