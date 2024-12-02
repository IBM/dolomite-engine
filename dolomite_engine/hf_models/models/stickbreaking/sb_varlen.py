import gc
import math

import torch
import triton
import triton.language as tl


log2 = math.log(2)
inv_log2: tl.constexpr = 1 / log2
ALLOW_TF32: tl.constexpr = False
DEBUG: tl.constexpr = False
BLOCK_M = 64
BLOCK_N = 64


def row_block_counts_and_sequence_ids(cu_seqlens: torch.Tensor, BLOCK_M: int, BLOCK_N: int):
    total_length = cu_seqlens[-1]
    M_COUNT = (total_length - 1) // BLOCK_M + 1
    M_range = torch.arange(M_COUNT, dtype=torch.int32, device=cu_seqlens.device)
    idxs = torch.arange(total_length, dtype=torch.int32, device=cu_seqlens.device)
    sequence_ids = (idxs[:, None] >= cu_seqlens[None, :]).sum(-1)
    M_block_start_ids = sequence_ids[::BLOCK_M]
    first_row_block = torch.where(M_block_start_ids == 0, 0, (cu_seqlens // BLOCK_N)[M_block_start_ids - 1])
    row_blocks = (M_range + 1) * (BLOCK_M // BLOCK_N) - first_row_block
    cu_row_blocks = torch.cumsum(row_blocks, -1)
    return cu_row_blocks, first_row_block, sequence_ids


@triton.jit
def softplus(x):
    out = tl.where(x < 15, tl.math.log2(1 + tl.math.exp2(x)), x)
    return out


@triton.jit
def compute_attn_weights(
    q, k, cm, neg_log_acc, qk_scale, mask, MASK: tl.constexpr, ALLOW_TF32: tl.constexpr = ALLOW_TF32
):
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32)
    qk *= qk_scale
    neg_log = -softplus(qk).to(q.dtype)
    _log_p = qk + neg_log_acc[:, None]
    if MASK:
        neg_log = tl.where(mask, neg_log, 0.0).to(neg_log.dtype)
        _log_p += tl.dot(neg_log, cm, allow_tf32=ALLOW_TF32)
        p = tl.math.exp2(_log_p)
        p = tl.where(mask, p, 0.0).to(p.dtype)
    else:
        _log_p += tl.dot(neg_log, cm, allow_tf32=ALLOW_TF32)
        p = tl.math.exp2(_log_p)
    return neg_log, p


# @triton.autotune(configs=[triton.Config({}, num_stages=4, num_warps=4)], key=['batch_size', 'token_size'],)
@triton.jit
def _forward(
    Q_ptr,
    stride_qh,
    stride_qm,
    stride_qd,
    K_ptr,
    stride_kh,
    stride_kn,
    stride_kd,
    V_ptr,
    stride_vh,
    stride_vn,
    stride_vd,
    O_ptr,
    stride_oh,
    stride_om,
    stride_od,
    M_ptr,
    stride_mh,
    stride_mi,
    stride_mm,
    CRB_ptr,
    R_ptr,
    stride_rh,
    stride_rm,
    batch_ptr,
    CSL_ptr,
    # L_ptr, stride_lm, stride_ln,
    logit_scale,
    batch_size,
    token_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr = ALLOW_TF32,
    inv_log2: tl.constexpr = inv_log2,
    no_grad: tl.constexpr = False,
    MIN_LOG_ACC: tl.constexpr = -1.0,
):

    head_id = tl.program_id(0)
    M_block_id = tl.program_id(1)

    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)

    M_blk_idxs = tl.max_contiguous(tl.multiple_of(M_block_id * BLOCK_M + M_range, BLOCK_M), BLOCK_M)
    NO_M_MASK = ((M_block_id + 1) * BLOCK_M - 1) < token_size

    # Loading thread information
    M_start_idx = tl.load(CRB_ptr + M_block_id)
    end_m = (M_block_id + 1) * BLOCK_M
    N_blk_idxs = tl.max_contiguous(tl.multiple_of(end_m + N_range, BLOCK_N), BLOCK_N)
    last_N_block_id = end_m // BLOCK_N

    # Init pointers
    Q_blk_ptrs = Q_ptr + stride_qh * head_id + stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :]
    K_blk_ptrs = K_ptr + stride_kh * head_id + stride_kn * N_blk_idxs[None, :] + stride_kd * D_range[:, None]
    V_blk_ptrs = V_ptr + stride_vh * head_id + stride_vn * N_blk_idxs[:, None] + stride_vd * D_range[None, :]
    O_blk_ptrs = O_ptr + stride_oh * head_id + stride_om * M_blk_idxs[:, None] + stride_od * D_range[None, :]
    M_blk_ptrs = M_ptr + stride_mh * head_id + stride_mi * M_start_idx + stride_mm * M_range
    R_blk_ptrs = R_ptr + stride_rh * head_id + stride_rm * M_blk_idxs

    # --- Load band vectors ---
    M_mask = M_blk_idxs < token_size
    if NO_M_MASK:
        q = tl.load(Q_blk_ptrs)
        batch_ids = tl.load(batch_ptr + M_blk_idxs)
    else:
        q = tl.load(Q_blk_ptrs, mask=M_mask[:, None])
        batch_ids = tl.load(batch_ptr + M_blk_idxs, mask=M_mask, other=batch_size)
    start_idxs = tl.load(CSL_ptr + batch_ids - 1, mask=batch_ids > 0, other=0).to(tl.int64)
    first_N_block_id = tl.min(start_idxs) // BLOCK_N
    neg_log_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    # --- End band vectors ---
    # Iterate only up to start of sequence
    iters = last_N_block_id - first_N_block_id  # tl.cdiv(tl.max(end_m - start_idxs), BLOCK_N)
    min_start_idxs = tl.min(start_idxs)
    is_same_start = min_start_idxs == tl.max(start_idxs)
    needs_compute = True  # tl.max(neg_log_acc) > MIN_LOG_ACC
    for i in range(iters):
        N_blk_idxs -= BLOCK_N
        K_blk_ptrs -= BLOCK_N * stride_kn
        V_blk_ptrs -= BLOCK_N * stride_vn
        M_blk_ptrs -= stride_mi
        if needs_compute:
            needs_compute = True  # tl.max(neg_log_acc) > MIN_LOG_ACC

        if needs_compute:
            on_band = i < BLOCK_M // BLOCK_N
            is_last_block = i == (iters - 1)
            on_N_edge = on_band and i == 0
            neg_log, p, _, v = compute_block(
                q,
                neg_log_acc,
                min_start_idxs,
                start_idxs,
                token_size,
                cm,
                qk_scale,
                K_blk_ptrs,
                V_blk_ptrs,
                M_blk_idxs,
                N_blk_idxs,
                is_last_block,
                is_same_start,
                on_N_edge,
                on_band,
            )
            # Store intermediate values
            if not no_grad:
                tl.store(M_blk_ptrs, neg_log_acc)
            neg_log_acc += tl.sum(neg_log, axis=1)
            acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=ALLOW_TF32)

    tl.store(O_blk_ptrs, acc.to(O_ptr.type.element_ty), mask=M_mask[:, None])
    tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc), mask=M_mask)


@triton.jit
def compute_block(
    q,
    neg_log_acc,
    min_start_idxs,
    start_idxs,
    token_size,
    cm,
    qk_scale,
    K_blk_ptrs,
    V_blk_ptrs,
    M_blk_idxs,
    N_blk_idxs,
    is_last_block: tl.constexpr,
    is_same_start: tl.constexpr,
    on_N_edge: tl.constexpr,
    on_band: tl.constexpr,
):

    if on_N_edge:
        N_mask = N_blk_idxs < token_size
        k = tl.load(K_blk_ptrs, mask=N_mask[None, :], other=0.0)
        v = tl.load(V_blk_ptrs, mask=N_mask[:, None], other=0.0)
    else:
        k = tl.load(K_blk_ptrs)
        v = tl.load(V_blk_ptrs)

    needs_mask = on_band or not (is_same_start and (not is_last_block))
    if needs_mask:  # On band
        mask = start_idxs[:, None] <= N_blk_idxs[None, :]  # sequence boundary
        if on_band:
            mask &= M_blk_idxs[:, None] > N_blk_idxs[None, :]  # diagonal boundary
            # if on_N_edge:
            #     mask &= (N_blk_idxs < token_size)[None, :]  # K-side block_boundary
        neg_log, p = compute_attn_weights(q, k, cm, neg_log_acc, qk_scale, mask, MASK=True)
    else:
        neg_log, p = compute_attn_weights(q, k, cm, neg_log_acc, qk_scale, None, MASK=False)
    return neg_log, p, k, v


def sb_fwd(q, k, v, cu_seqlens, batch_ids, cu_row_blocks, logit_scale=None, no_grad=False):
    with torch.cuda.device(q.device):
        num_heads = q.size(0)
        batch_size = cu_seqlens.size(0)
        token_size = q.size(1)
        dim_size = q.size(-1)
        BLOCK_D = triton.next_power_of_2(dim_size)
        if logit_scale is None:
            logit_scale = 1 / math.sqrt(dim_size)
        o = torch.zeros_like(q)
        rem = torch.zeros_like(q[:, :, 0], device=q.device)
        if no_grad:
            M = torch.zeros((1, 1, 1), device=q.device, dtype=q.dtype)
        else:
            M = torch.zeros((num_heads, cu_row_blocks[-1], BLOCK_M), device=q.device, dtype=q.dtype)
        M_count = triton.cdiv(token_size, BLOCK_M)
        # N_count = triton.cdiv(token_size, BLOCK_N)
        grid = (num_heads, M_count)
        # att = torch.zeros((token_size, token_size), device=out.device, dtype=torch.int32)
        # att_p = torch.zeros((token_size, token_size), device=batch_ids.device, dtype=q.dtype)
        _forward[grid](
            q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            M,
            M.stride(0),
            M.stride(1),
            M.stride(2),
            cu_row_blocks,
            rem,
            rem.stride(0),
            rem.stride(1),
            batch_ids,
            cu_seqlens,
            # att_p, att_p.stride(0), att_p.stride(1),
            logit_scale=logit_scale,
            batch_size=batch_size,
            token_size=token_size,
            no_grad=no_grad,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        return o, rem, M


@triton.jit
def _backward_dq(
    DO_ptr,
    stride_doh,
    stride_dom,
    stride_dod,
    DR_ptr,
    stride_drh,
    stride_drm,
    Q_ptr,
    stride_qh,
    stride_qm,
    stride_qd,
    K_ptr,
    stride_kh,
    stride_kn,
    stride_kd,
    V_ptr,
    stride_vh,
    stride_vn,
    stride_vd,
    DQ_ptr,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    M_ptr,
    stride_mh,
    stride_mi,
    stride_mm,
    DM_ptr,
    CRB_ptr,
    batch_ptr,
    CSL_ptr,
    # L_ptr, stride_lm, stride_ln,
    logit_scale,
    batch_size,
    token_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    inv_log2: tl.constexpr = inv_log2,
    ALLOW_TF32: tl.constexpr = ALLOW_TF32,
):
    head_id = tl.program_id(0)
    M_block_id = tl.program_id(1)
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    # Start by calculating the output block
    M_blk_idxs = M_block_id * BLOCK_M + M_range
    # NO_M_MASK = (M_block_id + 1) * BLOCK_M - 1 < token_size
    M_mask = M_blk_idxs < token_size
    # Load all sequence boundaries
    batch_ids = tl.load(batch_ptr + M_blk_idxs, mask=M_mask, other=batch_size).to(tl.int32)
    # Loading important thread information
    start_idxs = tl.load(CSL_ptr + batch_ids - 1, mask=batch_ids > 0, other=0).to(tl.int32)

    M_start_idx = tl.load(CRB_ptr + M_block_id - 1, mask=M_block_id > 0, other=0)
    end_m = (M_block_id + 1) * BLOCK_M
    last_N_block_id = end_m // BLOCK_N
    first_N_block_id = tl.min(start_idxs) // BLOCK_N
    N_blk_idxs = first_N_block_id * BLOCK_N + N_range

    # Init pointers
    Q_blk_ptrs = Q_ptr + stride_qh * head_id + stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :]
    K_blk_ptrs = K_ptr + stride_kh * head_id + stride_kn * N_blk_idxs[None, :] + stride_kd * D_range[:, None]
    V_blk_ptrs = V_ptr + stride_vh * head_id + stride_vn * N_blk_idxs[:, None] + stride_vd * D_range[None, :]
    DO_blk_ptrs = DO_ptr + stride_doh * head_id + stride_dom * M_blk_idxs[:, None] + stride_dod * D_range[None, :]
    DQ_blk_ptrs = DQ_ptr + stride_dqh * head_id + stride_dqm * M_blk_idxs[:, None] + stride_dqd * D_range[None, :]
    DR_blk_ptrs = DR_ptr + stride_drh * head_id + stride_drm * M_blk_idxs
    M_blk_ptrs = M_ptr + stride_mh * head_id + stride_mi * M_start_idx + stride_mm * M_range
    DM_blk_ptrs = DM_ptr + stride_mh * head_id + stride_mi * M_start_idx + stride_mm * M_range

    # L_blk_ptrs = L_ptr + stride_lm * M_blk_idxs[:, None] + stride_ln * N_blk_idxs[None, :] # TODO dev

    # --- Load band vectors ---
    q = tl.load(Q_blk_ptrs, mask=M_mask[:, None], other=0.0)
    do = tl.load(DO_blk_ptrs, mask=M_mask[:, None], other=0.0)
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)
    dr = tl.load(DR_blk_ptrs, mask=M_mask, other=0.0)
    grad_prev_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    # --- End band vectors ---

    # Iterate all the way to first block
    # iters = ((M_block_id + 1) * BLOCK_M) // BLOCK_N
    # Iterate only up to start of sequence
    min_start_idxs = tl.min(start_idxs)
    is_same_start = min_start_idxs == tl.max(start_idxs)
    iters = last_N_block_id - first_N_block_id  # tl.cdiv(tl.max(end_m - start_idxs), BLOCK_N)
    N_block_id = first_N_block_id
    for i in range(iters):
        neg_log_acc = tl.load(M_blk_ptrs)
        on_band = (iters - i - 1) < BLOCK_M // BLOCK_N
        is_last_block = i == 0
        on_N_edge = on_band and i == iters - 1
        neg_log, p, k, v = compute_block(
            q,
            neg_log_acc,
            min_start_idxs,
            start_idxs,
            token_size,
            cm,
            qk_scale,
            K_blk_ptrs,
            V_blk_ptrs,
            M_blk_idxs,
            N_blk_idxs,
            is_last_block,
            is_same_start,
            on_N_edge,
            on_band,
        )
        tl.store(DM_blk_ptrs, grad_prev_acc)
        # --- End compute attn stuff ---
        # --- Do gradient stuff ---
        dA = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:, None]
        att_dA = (p * dA).to(cm.dtype)
        cumul_att_dA = tl.dot(att_dA, tl.trans(cm), allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
        dqk = (att_dA - (1 - tl.math.exp2(neg_log.to(tl.float32))) * cumul_att_dA).to(k.dtype)
        dq += tl.dot(dqk, tl.trans(k), allow_tf32=ALLOW_TF32)
        grad_prev_acc += tl.sum(att_dA, axis=1)
        # --- End gradient stuff ---

        N_block_id += 1
        N_blk_idxs += BLOCK_N
        K_blk_ptrs += BLOCK_N * stride_kn
        V_blk_ptrs += BLOCK_N * stride_vn
        M_blk_ptrs += stride_mi
        DM_blk_ptrs += stride_mi

    tl.store(DQ_blk_ptrs, (logit_scale * dq).to(DQ_ptr.type.element_ty), mask=M_mask[:, None])


@triton.jit
def _backward_dkdv(
    DO_ptr,
    stride_doh,
    stride_dom,
    stride_dod,
    DR_ptr,
    stride_drh,
    stride_drm,
    Q_ptr,
    stride_qh,
    stride_qm,
    stride_qd,
    K_ptr,
    stride_kh,
    stride_kn,
    stride_kd,
    V_ptr,
    stride_vh,
    stride_vn,
    stride_vd,
    M_ptr,
    stride_mh,
    stride_mi,
    stride_mm,
    DM_ptr,
    CRB_ptr,
    FRB_ptr,
    DK_ptr,
    stride_dkh,
    stride_dkm,
    stride_dkd,
    DV_ptr,
    stride_dvh,
    stride_dvm,
    stride_dvd,
    batch_ptr,
    CSL_ptr,
    # MI_ptr, stride_mim, stride_min,
    # L_ptr, stride_lm, stride_ln,
    logit_scale,
    batch_size,
    token_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    inv_log2: tl.constexpr = inv_log2,
):

    head_id = tl.program_id(0)
    N_block_id = tl.program_id(1)
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    # Start by calculating the output block
    N_blk_idxs = N_block_id * BLOCK_N + N_range
    N_mask = N_blk_idxs < token_size
    # Load all sequence boundaries
    batch_ids = tl.load(batch_ptr + N_blk_idxs, mask=N_mask, other=batch_size).to(tl.int32)
    # Loading important thread information
    end_idxs = tl.load(CSL_ptr + batch_ids, mask=N_mask, other=token_size).to(tl.int32)
    first_idx = N_block_id * BLOCK_N
    last_idx = tl.max(end_idxs) - 1
    first_M_block_id = first_idx // BLOCK_M
    last_M_block_id = last_idx // BLOCK_M
    M_blk_idxs = first_M_block_id * BLOCK_M + M_range

    # Init block pointers
    DO_blk_ptrs = DO_ptr + stride_doh * head_id + stride_dom * M_blk_idxs[:, None] + stride_dod * D_range[None, :]
    Q_blk_ptrs = Q_ptr + stride_qh * head_id + stride_qm * M_blk_idxs[:, None] + stride_qd * D_range[None, :]
    K_blk_ptrs = K_ptr + stride_kh * head_id + stride_kn * N_blk_idxs[None, :] + stride_kd * D_range[:, None]
    V_blk_ptrs = V_ptr + stride_vh * head_id + stride_vn * N_blk_idxs[None, :] + stride_vd * D_range[:, None]
    DK_blk_ptrs = DK_ptr + stride_dkh * head_id + stride_dkm * N_blk_idxs[:, None] + stride_dkd * D_range[None, :]
    DV_blk_ptrs = DV_ptr + stride_dvh * head_id + stride_dvm * N_blk_idxs[:, None] + stride_dvd * D_range[None, :]
    DR_blk_ptrs = DR_ptr + stride_drh * head_id + stride_drm * M_blk_idxs
    # L_blk_ptrs = L_ptr + stride_lm * M_blk_idxs[:, None] + stride_ln * N_blk_idxs[None, :] # TODO dev

    # --- Load band vectors ---
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.type.element_ty)
    k = tl.load(K_blk_ptrs, mask=N_mask[None, :], other=0.0)
    vT = tl.load(V_blk_ptrs, mask=N_mask[None, :], other=0.0)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    # --- End band vectors ---
    # Iterate only up to start of sequence
    iters = last_M_block_id - first_M_block_id + 1
    M_block_id = first_M_block_id
    is_same_end = tl.min(end_idxs) == tl.max(end_idxs)
    for i in range(iters):
        M_mask = M_blk_idxs < token_size
        q = tl.load(Q_blk_ptrs, mask=M_mask[:, None], other=0.0)
        dr = tl.load(DR_blk_ptrs, mask=M_mask, other=0.0)
        do = tl.load(DO_blk_ptrs, mask=M_mask[:, None])

        N_first_idx = tl.load(FRB_ptr + M_block_id)
        if M_block_id == 0:
            M_start_idx = 0
        else:
            M_start_idx = tl.load(CRB_ptr + M_block_id - 1).to(tl.int32)
        # if head_id == 0:
        #     tl.store(MI_ptr + stride_mim * M_block_id + stride_min * N_block_id,
        #              (M_start_idx + (N_block_id - N_first_idx)))

        M_idxs = stride_mh * head_id + stride_mi * (M_start_idx + (N_block_id - N_first_idx)) + stride_mm * M_range
        # --- Do compute attn stuff ---
        # Load intermediate values
        neg_log_acc = tl.load(M_ptr + M_idxs)
        (M_block_id + 1) * BLOCK_M - 1 < token_size
        on_band = i == 0
        is_last_block = i == iters - 1
        needs_mask = on_band or not (is_same_end and (not is_last_block))
        if needs_mask:  # On band
            mask = M_blk_idxs[:, None] < end_idxs[None, :]  # sequence boundary
            if on_band:
                mask &= M_blk_idxs[:, None] > N_blk_idxs[None, :]  # diagonal boundary
                # if on_M_edge:
                #     mask &= (M_blk_idxs < token_size)[None, :]  # K-side block_boundary
            neg_log, p = compute_attn_weights(q, k, cm, neg_log_acc, qk_scale, mask, MASK=True)
        else:
            neg_log, p = compute_attn_weights(q, k, cm, neg_log_acc, qk_scale, None, MASK=False)

        """
        TODO: old.
        mask = M_blk_idxs[:, None] < end_idxs[None, :]    # sequence boundary
        mask &= M_blk_idxs[:, None] > N_blk_idxs[None, :] # diagonal boundary
        mask &= N_mask[None, :]                           # K-side block_boundary
        neg_log, p = compute_attn_weights(q, k, cm, neg_log_acc, qk_scale, mask, MASK=True)
        """
        # --- End compute attn stuff ---
        # --- Do gradient stuff ---
        grad_prev_acc = tl.load(DM_ptr + M_idxs)
        dA = tl.dot(do, vT, allow_tf32=ALLOW_TF32) - dr[:, None]
        att_dA = (p * dA).to(cm.dtype)
        cumul_att_dA = tl.dot(att_dA, tl.trans(cm), allow_tf32=ALLOW_TF32) + grad_prev_acc[:, None]
        dqk = (att_dA - (1 - tl.math.exp2(neg_log.to(tl.float32))) * cumul_att_dA).to(k.dtype)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do, allow_tf32=ALLOW_TF32)
        dk += tl.dot(tl.trans(dqk.to(q.dtype)), q, allow_tf32=ALLOW_TF32)
        # --- End gradient stuff ---
        # block = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32) + 1
        # tl.store(L_blk_ptrs, -(block * 0.5 + N_block_id * 0.25), mask=M_mask[:, None] & N_mask[None, :])
        # tl.store(L_blk_ptrs, block * 2, mask=mask)
        # tl.store(L_blk_ptrs, p, mask=mask)
        # L_blk_ptrs += BLOCK_M * stride_lm # TODO dev
        M_block_id += 1
        M_blk_idxs += BLOCK_M
        Q_blk_ptrs += BLOCK_M * stride_qm
        DO_blk_ptrs += BLOCK_M * stride_dom
        DR_blk_ptrs += BLOCK_M * stride_drm
    tl.store(DK_blk_ptrs, (dk * logit_scale).to(DK_ptr.type.element_ty), mask=N_mask[:, None])
    tl.store(DV_blk_ptrs, dv.to(DV_ptr.type.element_ty), mask=N_mask[:, None])
    # tl.store(R_blk_ptrs, neg_log_acc)


def sb_bwd(do, dr, q, k, v, cu_seqlens, M, sequence_ids, cu_row_blocks, first_row_block, logit_scale=None):
    with torch.cuda.device(q.device):
        batch_size = cu_seqlens.size(0)
        num_heads = q.size(0)
        token_size = q.size(1)
        dim_size = q.size(-1)
        if logit_scale is None:
            logit_scale = 1 / math.sqrt(dim_size)
        BLOCK_D = triton.next_power_of_2(dim_size)
        # BLOCK_BATCH = triton.next_power_of_2(batch_size)
        M_count = triton.cdiv(token_size, BLOCK_M)
        N_count = triton.cdiv(token_size, BLOCK_N)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        DM = torch.zeros_like(M)
        # print(DM.numel())
        # att = torch.zeros((token_size, token_size), device=out.device, dtype=torch.int32)
        # att_dq = torch.zeros((token_size, token_size), device=q.device, dtype=q.dtype)
        _backward_dq[num_heads, M_count](
            # DO_ptr, stride_dom, stride_dod,
            do,
            do.stride(0),
            do.stride(1),
            do.stride(2),
            # DR_ptr, stride_drm,
            dr,
            dr.stride(0),
            dr.stride(1),
            # Q_ptr, stride_qm, stride_qd,
            q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            # K_ptr, stride_kn, stride_kd,
            k,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            # V_ptr, stride_vn, stride_vd,
            v,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            # DQ_ptr, stride_dqm, stride_dqd,
            dq,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            # M_ptr, stride_mi, stride_mm, CRB_ptr,
            M,
            M.stride(0),
            M.stride(1),
            M.stride(2),
            DM,
            cu_row_blocks,
            # batch_ptr, CSL_ptr,
            sequence_ids,
            cu_seqlens,
            # L_ptr, stride_lm, stride_ln,
            # att_dq, att_dq.stride(0), att_dq.stride(1),
            # logit_scale, batch_size, token_size,
            logit_scale=logit_scale,
            batch_size=batch_size,
            token_size=token_size,
            # BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        # M_idxs = torch.zeros((M_count, N_count), dtype=torch.int32, device=do.device) - 1
        _backward_dkdv[num_heads, N_count](
            do,
            do.stride(0),
            do.stride(1),
            do.stride(2),
            dr,
            dr.stride(0),
            dr.stride(1),
            q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            M,
            M.stride(0),
            M.stride(1),
            M.stride(2),
            DM,
            cu_row_blocks,
            first_row_block,
            dk,
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv,
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            sequence_ids,
            cu_seqlens,
            # M_idxs, M_idxs.stride(0), M_idxs.stride(1),
            # att_dkdv, att_dkdv.stride(0), att_dkdv.stride(1),
            logit_scale=logit_scale,
            batch_size=batch_size,
            token_size=token_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        # print(M_idxs)
        # print(M.size())
        # unique_idxs = torch.sort(M_idxs[M_idxs != -1])[0]
        # correct_idxs = torch.arange(M.size(1), dtype=torch.int32, device=do.device)
        # assert all(unique_idxs == correct_idxs)
        # assert M.size(1) - 1 == M_idxs[-1, -1], (M.size(1), M_idxs[-1, -1])
        return dq, dk, dv


"""
| Tasks  |Version|Filter|n-shot|    Metric     |   | Value |   |Stderr|
|--------|------:|------|-----:|---------------|---|------:|---|------|
|pile_10k|      1|none  |     0|bits_per_byte  |↓  | 0.7010|±  |   N/A|
|        |       |none  |     0|byte_perplexity|↓  | 1.6256|±  |   N/A|
|        |       |none  |     0|word_perplexity|↓  |25.9136|±  |   N/A|
|--------|------:|------|-----:|---------------|---|------:|---|------|
|pile_10k|      1|none  |     0|bits_per_byte  |↓  | 0.7010|±  |   N/A|
|        |       |none  |     0|byte_perplexity|↓  | 1.6256|±  |   N/A|
|        |       |none  |     0|word_perplexity|↓  |25.9136|±  |   N/A|
"""


class StickBreakingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, cu_row_blocks, first_row_block, sequence_ids, inv_temp):
        no_grad = not ctx.needs_input_grad[0]
        logit_scale = inv_temp
        o, rem, M = sb_fwd(q, k, v, cu_seqlens, sequence_ids, cu_row_blocks, logit_scale=inv_temp, no_grad=no_grad)
        if no_grad:
            # del M
            M = None
            # gc.collect()
            # torch.cuda.empty_cache()
        ctx.save_for_backward(q, k, v, M, cu_seqlens, sequence_ids, cu_row_blocks, first_row_block)
        ctx.logit_scale = logit_scale
        return o, rem

    @staticmethod
    def backward(ctx, do, drem):
        logit_scale = ctx.logit_scale
        q, k, v, M, cu_seqlens, sequence_ids, cu_row_blocks, first_row_block = ctx.saved_tensors
        dq, dk, dv = sb_bwd(
            do, drem, q, k, v, cu_seqlens, M, sequence_ids, cu_row_blocks, first_row_block, logit_scale
        )
        return dq, dk, dv, None, None, None, None, None


def sb_flash_attn_varlen(q, k, v, cu_seqlens, inv_temp=None, zero_start=True):
    if zero_start:
        assert cu_seqlens[0] == 0
        cu_seqlens = cu_seqlens[1:]
    if inv_temp is None:
        inv_temp = 1 / math.sqrt(q.size(-1))
    with torch.no_grad():
        cu_row_blocks, first_row_block, sequence_ids = row_block_counts_and_sequence_ids(cu_seqlens, BLOCK_M, BLOCK_N)
    return sb_attn_varlen_(q, k, v, inv_temp, cu_seqlens, first_row_block, cu_row_blocks, sequence_ids)


def sb_attn_varlen_(q, k, v, inv_temp, cu_seqlens, first_row_block, cu_row_blocks, sequence_ids):
    return StickBreakingAttention.apply(q, k, v, cu_seqlens, cu_row_blocks, first_row_block, sequence_ids, inv_temp)
