import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_fp8_decode_kernel(
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    O_ptr,
    BlockTables_ptr,
    ContextLens_ptr,
    stride_q_batch,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_o_batch,
    stride_o_head,
    stride_bt_batch,
    scale,
    kv_group_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + batch_idx * stride_q_batch + head_idx * stride_q_head + offs_d
    q = tl.load(q_ptrs).to(tl.float32)

    context_len = tl.load(ContextLens_ptr + batch_idx)
    num_blocks = tl.cdiv(context_len, BLOCK_SIZE)
    kv_head_idx = head_idx // kv_group_size

    m_i = tl.full((), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for block_idx in tl.static_range(0, MAX_NUM_BLOCKS):
        if block_idx < num_blocks:
            block_id = tl.load(BlockTables_ptr + batch_idx * stride_bt_batch + block_idx)
            block_start = block_idx * BLOCK_SIZE
            remaining = context_len - block_start
            tokens_in_block = tl.minimum(remaining, BLOCK_SIZE)
            num_tiles = tl.cdiv(tokens_in_block, BLOCK_TOKENS)

            for tile_idx in tl.static_range(0, BLOCK_SIZE // BLOCK_TOKENS):
                if tile_idx < num_tiles:
                    offs_t = tile_idx * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
                    token_mask = offs_t < tokens_in_block

                    k_ptrs = (
                        K_cache_ptr
                        + block_id * stride_k_block
                        + offs_t[:, None] * stride_k_token
                        + kv_head_idx * stride_k_head
                        + offs_d[None, :]
                    )
                    v_ptrs = (
                        V_cache_ptr
                        + block_id * stride_v_block
                        + offs_t[:, None] * stride_v_token
                        + kv_head_idx * stride_v_head
                        + offs_d[None, :]
                    )

                    k = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
                    scores = tl.sum(k * q[None, :], axis=1) * scale
                    scores = tl.where(token_mask, scores, -float('inf'))

                    m_ij = tl.max(scores, axis=0)
                    m_new = tl.maximum(m_i, m_ij)
                    alpha = tl.exp(m_i - m_new)
                    p = tl.exp(scores - m_new)
                    l_new = l_i * alpha + tl.sum(p, axis=0)

                    v = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
                    pv = tl.sum(p[:, None] * v, axis=0)
                    acc = acc * (l_i * alpha / l_new) + pv / l_new
                    m_i = m_new
                    l_i = l_new

    o_ptrs = O_ptr + batch_idx * stride_o_batch + head_idx * stride_o_head + offs_d
    tl.store(o_ptrs, acc.to(tl.bfloat16))


def _normalize_query_shape(q: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if q.ndim == 4:
        if q.shape[1] != 1:
            raise ValueError(f'decode query must have shape [batch, 1, heads, dim], got {tuple(q.shape)}')
        return q[:, 0].contiguous(), True
    if q.ndim != 3:
        raise ValueError(f'decode query must have shape [batch, heads, dim], got {tuple(q.shape)}')
    return q.contiguous(), False


def _validate_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    q_3d, restore_seq_dim = _normalize_query_shape(q)
    if q_3d.dtype != torch.bfloat16:
        raise TypeError(f'q must be bfloat16, got {q_3d.dtype}')
    if k_cache.dtype != torch.float8_e4m3fn or v_cache.dtype != torch.float8_e4m3fn:
        raise TypeError('k_cache and v_cache must use torch.float8_e4m3fn')
    if k_cache.shape != v_cache.shape:
        raise ValueError('k_cache and v_cache must have the same shape')
    if k_cache.ndim != 4:
        raise ValueError(f'k_cache must be [num_blocks, block_size, num_kv_heads, head_dim], got {tuple(k_cache.shape)}')
    if block_tables.ndim != 2:
        raise ValueError(f'block_tables must be [batch, max_num_blocks], got {tuple(block_tables.shape)}')
    if context_lens.ndim != 1:
        raise ValueError(f'context_lens must be [batch], got {tuple(context_lens.shape)}')
    if not (q_3d.is_cuda and k_cache.is_cuda and v_cache.is_cuda and block_tables.is_cuda and context_lens.is_cuda):
        raise ValueError('all tensors must be on CUDA')
    batch, num_heads, head_dim = q_3d.shape
    _, _, num_kv_heads, cache_head_dim = k_cache.shape
    if batch != block_tables.shape[0] or batch != context_lens.shape[0]:
        raise ValueError('batch dimension mismatch between q, block_tables and context_lens')
    if head_dim != 128 or cache_head_dim != 128:
        raise ValueError('this kernel currently requires head_dim=128')
    if num_heads % num_kv_heads != 0:
        raise ValueError('num_heads must be divisible by num_kv_heads for GQA')
    return q_3d, restore_seq_dim


@torch.inference_mode()
def hilda_paged_attention_fp8_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    restore_seq_dim = q.ndim == 4
    q_3d = q[:, 0] if restore_seq_dim else q
    batch, num_heads, head_dim = q_3d.shape
    _, block_size, num_kv_heads, _ = k_cache.shape

    output = torch.empty((batch, num_heads, head_dim), device=q_3d.device, dtype=torch.bfloat16)
    grid = (batch, num_heads)
    max_num_blocks = block_tables.shape[1]
    if max_num_blocks <= 4:
        num_warps = 4
    else:
        num_warps = 8 if batch <= 16 else 2
    _paged_attention_fp8_decode_kernel[grid](
        q_3d,
        k_cache,
        v_cache,
        output,
        block_tables,
        context_lens,
        q_3d.stride(0),
        q_3d.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        output.stride(0),
        output.stride(1),
        block_tables.stride(0),
        scale,
        num_heads // num_kv_heads,
        BLOCK_SIZE=block_size,
        BLOCK_TOKENS=128,
        HEAD_DIM=128,
        MAX_NUM_BLOCKS=block_tables.shape[1],
        num_warps=num_warps,
        num_stages=2,
    )
    if restore_seq_dim:
        return output.unsqueeze(1)
    return output


@torch.inference_mode()
def reference_paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q_3d, restore_seq_dim = _normalize_query_shape(q)
    batch, num_heads, head_dim = q_3d.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    kv_head_indices = torch.arange(num_heads, device=q_3d.device, dtype=torch.long) // (num_heads // num_kv_heads)
    out = torch.empty((batch, num_heads, head_dim), device=q_3d.device, dtype=torch.bfloat16)

    for batch_idx in range(batch):
        context_len = int(context_lens[batch_idx].item())
        num_blocks = triton.cdiv(context_len, block_size)
        block_ids = block_tables[batch_idx, :num_blocks].to(dtype=torch.long)
        k_tokens = k_cache.index_select(0, block_ids).reshape(-1, num_kv_heads, head_dim)[:context_len].to(torch.float32)
        v_tokens = v_cache.index_select(0, block_ids).reshape(-1, num_kv_heads, head_dim)[:context_len].to(torch.float32)
        q_heads = q_3d[batch_idx].to(torch.float32)
        k_heads = k_tokens.index_select(1, kv_head_indices)
        v_heads = v_tokens.index_select(1, kv_head_indices)
        scores = torch.einsum('hd,thd->ht', q_heads, k_heads) * scale
        probs = scores.softmax(dim=-1)
        out[batch_idx] = torch.einsum('ht,thd->hd', probs, v_heads).to(torch.bfloat16)

    if restore_seq_dim:
        return out.unsqueeze(1)
    return out
