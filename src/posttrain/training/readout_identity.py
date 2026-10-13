"""Parameter-byte identity independent of autograd flag membership."""
import hashlib

def all_parameter_hash(model):
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(f'{name}|{p.dtype}|{tuple(p.shape)}'.encode())
        flat=p.detach().reshape(-1)
        for start in range(0,p.numel(),4*1024*1024):
            h.update(flat[start:start+4*1024*1024].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()
