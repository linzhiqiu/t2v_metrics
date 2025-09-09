import torch
import torch.nn.functional as F
from einops import rearrange

def _unpad_input(input_ids, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    input_ids = rearrange(input_ids, 'b s ... -> (b s) ...')[indices]
    return input_ids, indices, cu_seqlens, max_seqlen_in_batch

def _pad_input(hidden_states, indices, batch, seqlen):
    output = torch.zeros(batch * seqlen, *hidden_states.shape[1:], device=hidden_states.device,
                         dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return rearrange(output, '(b s) ... -> b s ...', b=batch)
