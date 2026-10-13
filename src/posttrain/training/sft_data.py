from __future__ import annotations

from posttrain.data.schema import Problem

IGNORE_INDEX=-100
REJECTED_ENCODING_REASONS=('example_exceeds_context_limit','empty_shifted_target')


def chat_tokens(tokenizer,messages,*,add_generation_prompt):
    # Transformers 5 defaults return_dict=True. Request flat IDs explicitly;
    # len(BatchEncoding) counts fields, not tokens, and cannot enforce a limit.
    ids=tokenizer.apply_chat_template(messages,tokenize=True,
        add_generation_prompt=add_generation_prompt,return_dict=False)
    if not isinstance(ids,list) or not ids or any(type(token) is not int or token<0 for token in ids):
        raise ValueError('flat_chat_token_ids_required')
    return ids


def resolve_pad_token_id(tokenizer):
    """Pad token used for right padding, falling back to EOS when absent.

    Returns (token_id,source) so the run can record which token was used.
    """
    value=getattr(tokenizer,'pad_token_id',None);source='pad_token'
    if value is None:
        value=getattr(tokenizer,'eos_token_id',None);source='eos_token_fallback'
    if type(value) is not int or value<0:
        raise ValueError('tokenizer_pad_token_and_eos_missing')
    return value,source


def shifted_target_tokens(sample: dict) -> int:
    """Supervised tokens that contribute loss after the causal LM shift.

    Transformers shifts internally, so the counted positions are labels[1:]
    without ignore_index: prompt tokens and right padding contribute nothing.
    """
    return sum(1 for value in sample['labels'][1:] if value!=IGNORE_INDEX)


def encode_answer(tokenizer, problem: Problem, max_length: int) -> dict:
    """Qwen prefix template + answer with assistant-only causal labels.

    Transformers causal LM shifts labels; this function never shifts a second time.
    Whole examples exceeding max_length are rejected, not silently truncated targets.
    """
    if problem.validation_status!='reference_passed':
        raise ValueError('SFT requires an isolated reference backtest')
    prefix=chat_tokens(tokenizer,problem.prompt(),add_generation_prompt=True)
    full=chat_tokens(tokenizer,problem.prompt()+[{'role':'assistant','content':problem.references[0]}],
                     add_generation_prompt=False)
    if full[:len(prefix)]!=prefix:
        raise ValueError('chat_template_prefix_mismatch_review_tokenization')
    if len(full)>max_length:
        raise ValueError('example_exceeds_context_limit')
    if len(full)<=len(prefix):
        raise ValueError('empty_assistant_target')
    sample={'input_ids':full,'attention_mask':[1]*len(full),
            'labels':[IGNORE_INDEX]*len(prefix)+full[len(prefix):]}
    if shifted_target_tokens(sample)<1:
        raise ValueError('empty_shifted_target')
    return sample


def encode_examples(tokenizer,rows,max_length):
    """Encode train rows, rejecting unusable examples instead of truncating.

    Only target-length and empty-target rejections are tolerated; anything else
    is a bug or a data-contract violation and propagates.
    """
    examples=[];rejected=[]
    for problem in rows:
        try:
            sample=encode_answer(tokenizer,problem,max_length)
            if shifted_target_tokens(sample)<1:raise ValueError('empty_shifted_target')
        except ValueError as exc:
            if str(exc) not in REJECTED_ENCODING_REASONS:raise
            rejected.append({'problem_id':problem.problem_id,'reason':str(exc)})
        else:
            examples.append((problem.problem_id,sample))
    return examples,rejected


def partition_micro_batches(indices,micro_batch_size: int) -> list:
    """Consecutive partition of one optimizer update's examples.

    batch_size stays the number of examples per optimizer update; the trailing
    microbatch keeps the remainder and may be smaller than micro_batch_size.
    """
    if type(micro_batch_size) is not int or micro_batch_size<1:
        raise ValueError('invalid_micro_batch_size')
    ordered=list(indices)
    if not ordered:
        raise ValueError('empty_optimizer_update')
    return [ordered[start:start+micro_batch_size] for start in range(0,len(ordered),micro_batch_size)]


def microbatch_token_counts(samples,micro_batches) -> dict:
    """Pure per-update token accounting for right-padded microbatches.

    Right padding is inserted per microbatch up to that microbatch's longest
    example, so padded positions are counted per microbatch while supervised
    target tokens stay a property of the examples themselves.
    """
    records=[]
    for batch in micro_batches:
        if not batch:raise ValueError('empty_micro_batch')
        lengths=[len(samples[index]['input_ids']) for index in batch]
        nonpadding=sum(lengths);width=max(lengths)
        records.append({'size':len(batch),'width':width,'nonpadding_tokens':nonpadding,
            'padding_tokens':width*len(batch)-nonpadding,
            'target_tokens':sum(shifted_target_tokens(samples[index]) for index in batch)})
    nonpadding_tokens=sum(record['nonpadding_tokens'] for record in records)
    padding_tokens=sum(record['padding_tokens'] for record in records)
    processed_tokens=nonpadding_tokens+padding_tokens
    return {'micro_batch_count':len(records),
        'target_tokens':sum(record['target_tokens'] for record in records),
        'nonpadding_tokens':nonpadding_tokens,'padding_tokens':padding_tokens,
        'processed_tokens':processed_tokens,
        'padding_fraction':round(padding_tokens/processed_tokens,6) if processed_tokens else 0.0,
        'micro_batches':records}


def microbatch_loss_weights(counts: dict) -> list:
    """Microbatch valid shifted targets divided by the update's valid targets.

    Weights sum to one, so summing (microbatch mean CE x weight) with backward()
    is the assistant-only token mean over the ENTIRE optimizer update.
    """
    total=counts['target_tokens']
    if total<1:raise ValueError('empty_optimizer_update_targets')
    return [record['target_tokens']/total for record in counts['micro_batches']]


def length_distribution(values,bucket: int = 128) -> dict:
    """Length histogram summary; counts only, never sample content or ids."""
    ordered=sorted(int(value) for value in values)
    if not ordered:
        return {'count':0,'min':0,'max':0,'mean':0.0,'median':0,'p90':0,'bucket':bucket,'histogram':[]}
    def quantile(fraction):
        return ordered[min(len(ordered)-1,int(fraction*(len(ordered)-1)+.5))]
    histogram={}
    for value in ordered:
        start=value//bucket*bucket
        histogram[start]=histogram.get(start,0)+1
    return {'count':len(ordered),'min':ordered[0],'max':ordered[-1],
        'mean':round(sum(ordered)/len(ordered),3),'median':quantile(.5),'p90':quantile(.9),
        'bucket':bucket,'histogram':[{'start':start,'end':start+bucket-1,'count':histogram[start]}
            for start in sorted(histogram)]}


def data_length_report(cfg,samples,rejected,pad_token_source: str) -> dict:
    """Accepted/rejected counts and encoded length distributions.

    Aggregate numbers only: no problem ids, no token ids and no reference text,
    so the report can be shared without exposing the hidden answers.
    """
    reasons=sorted({row['reason'] for row in rejected})
    return {'max_length':cfg['max_length'],'batch_size':cfg['batch_size'],
        'micro_batch_size':cfg['micro_batch_size'],'pad_token_source':pad_token_source,
        'accepted_examples':len(samples),'rejected_examples':len(rejected),
        'rejected_by_reason':{reason:sum(1 for row in rejected if row['reason']==reason) for reason in reasons},
        'targets_truncated':False,'truncation_policy':'reject_examples_beyond_max_length',
        'total_tokens':length_distribution([len(sample['input_ids']) for sample in samples]),
        'assistant_tokens':length_distribution([sum(1 for value in sample['labels'] if value!=IGNORE_INDEX)
            for sample in samples]),
        'shifted_target_tokens':length_distribution([shifted_target_tokens(sample) for sample in samples])}


def pad_batch(examples: list[dict], pad_token_id: int) -> dict:
    if not examples:
        raise ValueError('empty_batch')
    longest=max(len(x['input_ids']) for x in examples)
    result={'input_ids':[],'attention_mask':[],'labels':[]}
    for example in examples:
        if len({len(example[k]) for k in result})!=1:
            raise ValueError('inconsistent_sample_lengths')
        padding=longest-len(example['input_ids'])
        for name,pad in [('input_ids',pad_token_id),('attention_mask',0),('labels',IGNORE_INDEX)]:
            result[name].append(example[name]+[pad]*padding)
    return result
