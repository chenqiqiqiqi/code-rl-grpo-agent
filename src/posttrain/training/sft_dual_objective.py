"""CE uses S only; full-vocabulary frozen-teacher KL uses A only."""
from contextlib import nullcontext
from posttrain.training.sft_data import pad_batch,partition_micro_batches,microbatch_token_counts,microbatch_loss_weights
from posttrain.training.sft_kl_objective import teacher_log_probs,masked_forward_kl

def backward_routes(model,samples_s,samples_a,indices_s,indices_a,micro,pad,beta,device,*,autocast=True):
    import torch
    model.train();stats={};ce_total=kl_total=0.
    for route,samples,indices in [('S',samples_s,indices_s),('A',samples_a,indices_a)]:
        batches=partition_micro_batches(indices,micro);counts=microbatch_token_counts(samples,batches);stats[route]=counts
        if route=='A' and beta==0:continue
        for batch,weight in zip(batches,microbatch_loss_weights(counts)):
            inputs={k:torch.tensor(v,device=device) for k,v in pad_batch([samples[i] for i in batch],pad).items()}
            def precision():return torch.autocast('cuda',dtype=torch.bfloat16) if autocast else nullcontext()
            teacher=None
            if route=='A':
                model.eval()
                try:
                    with torch.no_grad(),model.disable_adapter(),precision():
                        result=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
                        teacher=teacher_log_probs(result.logits,inputs['labels']);del result
                finally:model.train()
            with precision():
                if route=='S':
                    result=model(**inputs);loss=result.loss
                    ce_total+=float(loss.detach())*weight
                else:
                    result=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
                    kl=masked_forward_kl(result.logits,inputs['labels'],teacher)
                    kl_total+=float(kl.detach())*weight;loss=beta*kl
                objective=loss*weight
            objective.backward()
            del result,loss,objective,teacher
    return {**stats,'ce_loss':ce_total,'forward_kl':kl_total,'loss':ce_total+beta*kl_total}
