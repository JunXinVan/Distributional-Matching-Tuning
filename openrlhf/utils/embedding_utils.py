import torch
import torch.nn.functional as F
import numpy as np
import ray
import math
from collections import Counter


_RFF_CACHE = {}
_CF_FREQ_CACHE = {}


def prepare_tensors_for_embedding(prompts_list, full_sequences_list, prompt_length, stride, num_blocks, n_samples_per_prompt, context_length, gen_len, return_inputs=False):
    """
    Returns a list of decoded strings for semantic reward calculation, across all rollout_batch_size * n_samples_per_prompt examples.
    """
    prompts_tensor = torch.stack(prompts_list)  # (rollout_batch_size // micro_rollout_batch_size, micro_rollout_batch_size, prompt_len)
    prompts_tensor = prompts_tensor.reshape(prompts_tensor.shape[0], prompts_tensor.shape[1] // n_samples_per_prompt, n_samples_per_prompt, prompts_tensor.shape[2])

    full_tensor = torch.stack(full_sequences_list)
    full_tensor = full_tensor.reshape(full_tensor.shape[0], full_tensor.shape[1] // n_samples_per_prompt, n_samples_per_prompt, full_tensor.shape[2])

    starting_idx = context_length
    gt_tensor = prompts_tensor[:,:,:,starting_idx:].unfold(3, gen_len, stride)

    gen_tensor = full_tensor[:,:,:,prompt_length:]
    gen_tensor = gen_tensor.reshape(gen_tensor.shape[0], gen_tensor.shape[1], gen_tensor.shape[2], gen_len, num_blocks)
    gen_tensor = gen_tensor.transpose(-1, -2)

    if return_inputs:
        ct_tensor = prompts_tensor[:,:,:,:-gen_len].unfold(3, context_length, stride)
        return gen_tensor, gt_tensor, ct_tensor
    return gen_tensor, gt_tensor

def temp_embed_one_hot(input_sequences, gt_sequences, vocab_size, dtype=torch.float32):
    # one_hot requires Long dtype with values in [0, vocab_size-1]
    input_sequences = input_sequences.to(torch.long, non_blocking=True)
    gt_sequences    = gt_sequences.to(torch.long, non_blocking=True)

    input_oh = F.one_hot(input_sequences, num_classes=vocab_size).to(dtype)
    gt_oh    = F.one_hot(gt_sequences,    num_classes=vocab_size).to(dtype)

    return input_oh, gt_oh

def decode_tensor(input_tensor, tokenizer):
    return tokenizer.batch_decode(input_tensor, skip_special_tokens=True)

def prepare_tensors_for_reward_model(input_sequences, gt_sequences, tokenizer, ct_sequences=None):
    if ct_sequences is not None:
        # flatten (rollout_batch_size * n_samples_per_prompt // micro_rollout_batch_size, micro_rollout_batch_size // n_samples_per_prompt, n_samples_per_prompt, num_blocks, partial_ct_len + gen_len)
        return decode_tensor(input_sequences.reshape(-1, input_sequences.shape[-1]), tokenizer), decode_tensor(gt_sequences.reshape(-1, gt_sequences.shape[-1]), tokenizer), decode_tensor(ct_sequences.reshape(-1, ct_sequences.shape[-1]), tokenizer)
    # flatten (rollout_batch_size * n_samples_per_prompt // micro_rollout_batch_size, micro_rollout_batch_size // n_samples_per_prompt, n_samples_per_prompt, num_blocks, partial_ct_len + gen_len)
    return decode_tensor(input_sequences.reshape(-1, input_sequences.shape[-1]), tokenizer), decode_tensor(gt_sequences.reshape(-1, gt_sequences.shape[-1]), tokenizer)


def whiten_embeddings_batched(
    Phi: torch.Tensor,
    Phi_gt: torch.Tensor,
    whiten_tol: float = 1e-5,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Batch whitening across the *sample axis* (N).

    This function expects the sample axis to be dimension 2 and the embedding dimension
    to be the last dimension.

    Supported input shapes:
    - 5D: (d1, d2, N, d3, D)
    - 7D: (d1, d2, N, d3, d4, d5, D)

    For each fixed index in the non-(N,D) dimensions, we take X ∈ R^{N×D} from Phi
    and compute:
        W = (X X^T)^(-1/2)   (pseudo-inverse with tolerance)
        Xw    = W X
        Xw_gt = W X_gt

    So Xw has (approximately) orthonormal rows when rank(X)=N; when rank-deficient,
    this becomes a projection onto the row-space.
    """
    if Phi.shape != Phi_gt.shape:
        raise ValueError(f"Phi and Phi_gt must have the same shape, got {Phi.shape} vs {Phi_gt.shape}")
    if Phi.ndim not in (5, 7):
        raise ValueError(f"Expected Phi.ndim in {{5,7}} with sample axis at dim=2, got {Phi.shape}")

    # Permute to move sample axis N next-to-last, keep embedding dim last.
    # For nd=5:  [0,1,3,2,4]
    # For nd=7:  [0,1,3,4,5,2,6]
    nd = Phi.ndim
    perm = [0, 1] + list(range(3, nd - 1)) + [2, nd - 1]
    inv_perm = [0] * nd
    for i, p in enumerate(perm):
        inv_perm[p] = i

    Phi_perm = Phi.permute(*perm).contiguous()
    Phi_gt_perm = Phi_gt.permute(*perm).contiguous()

    # Flatten all "batch" dims into one for batched linear algebra.
    *batch_dims, N, D = Phi_perm.shape
    B = 1
    for x in batch_dims:
        B *= int(x)

    Phi_flat = Phi_perm.reshape(B, N, D).float()
    Phi_gt_flat = Phi_gt_perm.reshape(B, N, D).float()

    # Batched SVD on (B, N, D) where typically N << D.
    # Use robust SVD with fallback for ill-conditioned matrices.
    try:
        U, S, _ = torch.linalg.svd(Phi_flat, full_matrices=False)  # U: (B,N,N), S: (B,N)
    except torch._C._LinAlgError:
        # Fallback 1: Add small noise to break degeneracy and retry
        noise_scale = 1e-6 * Phi_flat.abs().mean()
        Phi_flat_noisy = Phi_flat + noise_scale * torch.randn_like(Phi_flat)
        try:
            U, S, _ = torch.linalg.svd(Phi_flat_noisy, full_matrices=False)
        except torch._C._LinAlgError:
            # Fallback 2: Return original embeddings without whitening
            if normalize:
                Phi_out = F.normalize(Phi, p=2, dim=-1)
                Phi_gt_out = F.normalize(Phi_gt, p=2, dim=-1)
                return Phi_out, Phi_gt_out
            return Phi, Phi_gt

    # Safe inverse: zero out tiny singular values (per-batch).
    Smax = S.max(dim=-1, keepdim=True).values
    inv_S = torch.where(S > whiten_tol * Smax, 1.0 / (S + 1e-12), torch.zeros_like(S))  # (B,N)

    # W = U diag(inv_S) U^T
    W = (U * inv_S.unsqueeze(-2)) @ U.transpose(-1, -2)  # (B,N,N)
    Xw = W @ Phi_flat
    Xw_gt = W @ Phi_gt_flat

    # Cast back and reshape/unpermute to original.
    Xw = Xw.to(dtype=Phi.dtype)
    Xw_gt = Xw_gt.to(dtype=Phi_gt.dtype)

    Phi_tilde = Xw.reshape(*batch_dims, N, D).permute(*inv_perm).contiguous()
    Phi_gt_tilde = Xw_gt.reshape(*batch_dims, N, D).permute(*inv_perm).contiguous()

    if normalize:
        Phi_tilde = F.normalize(Phi_tilde, p=2, dim=-1)
        Phi_gt_tilde = F.normalize(Phi_gt_tilde, p=2, dim=-1)

    return Phi_tilde, Phi_gt_tilde


def _get_fixed_rff_params(input_dim: int, output_dim: int, sigma: float, seed: int, device: torch.device):
    key = (int(input_dim), int(output_dim), float(sigma), int(seed), str(device))
    if key not in _RFF_CACHE:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        weight = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
        weight = weight / max(float(sigma), 1e-6)
        phase = 2 * math.pi * torch.rand(output_dim, generator=generator, dtype=torch.float32)
        _RFF_CACHE[key] = (weight.to(device=device), phase.to(device=device))
    return _RFF_CACHE[key]


def apply_feature_map(embedding: torch.Tensor, feature_map_type: str = "identity", rff_num_features: int = 128, rff_sigma: float = 1.0, rff_seed: int = 43) -> torch.Tensor:
    """Apply an optional fixed feature map on the last embedding dimension.

    The default `identity` path is a no-op and preserves vanilla EBFT behavior.
    The `rff` path applies a deterministic random Fourier feature map with a
    fixed Gaussian projection and random phase.
    """
    if feature_map_type == "identity":
        return embedding
    if feature_map_type != "rff":
        raise ValueError(f"Unknown feature_map_type: {feature_map_type}")

    orig_shape = embedding.shape
    input_dim = orig_shape[-1]
    flat = embedding.reshape(-1, input_dim).float()
    weight, phase = _get_fixed_rff_params(
        input_dim=input_dim,
        output_dim=int(rff_num_features),
        sigma=float(rff_sigma),
        seed=int(rff_seed),
        device=flat.device,
    )
    mapped = math.sqrt(2.0 / float(rff_num_features)) * torch.cos(flat @ weight + phase)
    return mapped.to(dtype=embedding.dtype).reshape(*orig_shape[:-1], int(rff_num_features))


def _get_fixed_cf_frequencies(input_dim: int, num_freqs: int, sigma: float, seed: int, device: torch.device):
    key = (int(input_dim), int(num_freqs), float(sigma), int(seed), str(device))
    if key not in _CF_FREQ_CACHE:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        freqs = torch.randn(int(num_freqs), int(input_dim), generator=generator, dtype=torch.float32)
        freqs = freqs / max(float(sigma), 1e-6)
        _CF_FREQ_CACHE[key] = freqs.to(device=device)
    return _CF_FREQ_CACHE[key]


def _compute_cf_loss_terms(target_real, target_imag, gen_real, gen_imag, alpha: float, beta: float):
    target_norm = torch.sqrt(target_real * target_real + target_imag * target_imag)
    gen_norm = torch.sqrt(gen_real * gen_real + gen_imag * gen_imag)

    amp_diff = target_norm - gen_norm
    loss_amp = amp_diff * amp_diff

    loss_pha = 2 * (
        target_norm * gen_norm
        - gen_real * target_real
        - gen_imag * target_imag
    )
    loss_pha = loss_pha.clamp(min=1e-12)
    return torch.sqrt(float(alpha) * loss_amp + float(beta) * loss_pha)


def _build_cf_target_embedding(
    gt_embedding: torch.Tensor,
    cf_target_mode: str,
    cf_target_num_refs: int,
    cf_target_std: float,
    cf_target_seed: int,
) -> torch.Tensor:
    """Build the target empirical measure used by the CF discrepancy.

    Modes:
    - single: keep the original EBFT-style single-reference target.
    - vicinal: create a small local target distribution around the GT feature
      by adding deterministic Gaussian perturbations in feature space.
    """
    target_embedding = gt_embedding[:, :, :1, :, :].float()

    if cf_target_mode == "single" or int(cf_target_num_refs) <= 1:
        return target_embedding
    if cf_target_mode != "vicinal":
        raise ValueError(f"Unknown cf_target_mode: {cf_target_mode}")

    num_refs = int(cf_target_num_refs)
    base = target_embedding
    flat = base.reshape(-1, 1, base.shape[-1])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cf_target_seed))
    noise = torch.randn(
        flat.shape[0],
        num_refs - 1,
        flat.shape[-1],
        generator=generator,
        dtype=torch.float32,
    ).to(device=flat.device)

    # Scale perturbations by the local feature RMS so the smoothing radius stays
    # meaningful across different feature maps / whitening settings.
    local_rms = flat.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    noisy = flat + float(cf_target_std) * local_rms * noise

    target_flat = torch.cat([flat, noisy], dim=1)
    target_embedding = target_flat.reshape(
        base.shape[0],
        base.shape[1],
        num_refs,
        base.shape[3],
        base.shape[4],
    )
    return target_embedding


def groom_last_token_embedding(hidden_states: torch.Tensor, qa_mask: torch.Tensor, qa_masking: bool = False) -> torch.Tensor:
    """Select the last answer token along the token axis.

    Expected shapes:
    - hidden_states: (..., T, NF, H)
    - qa_mask:       (..., T, NF, 1)
    """
    *prefix_dims, num_tokens, num_feat, hidden_dim = hidden_states.shape
    device = hidden_states.device

    if not qa_masking:
        qa_mask = torch.ones_like(qa_mask)

    mask = qa_mask.squeeze(-1).to(torch.bool)  # (..., T, NF)
    time_idx = torch.arange(num_tokens, device=device).view(*([1] * len(prefix_dims)), num_tokens, 1)
    time_idx = time_idx.expand(*prefix_dims, num_tokens, num_feat)
    last_idx = time_idx.masked_fill(~mask, -1).amax(dim=-2)  # (..., NF)

    safe_idx = last_idx.clamp_min(0).unsqueeze(-2).unsqueeze(-1).expand(*prefix_dims, 1, num_feat, hidden_dim)
    out = hidden_states.gather(dim=-3, index=safe_idx)  # (..., 1, NF, H)

    no_ans = last_idx.eq(-1).unsqueeze(-2).unsqueeze(-1).expand_as(out)
    out = out.masked_fill(no_ans, 0.0)
    return out


def prepare_distribution_embeddings_from_split_hidden_states(
    gt_hidden_states: torch.Tensor,
    gen_hidden_states: torch.Tensor,
    gt_qa_mask: torch.Tensor,
    gen_qa_mask: torch.Tensor,
    n_samples_per_prompt: int,
    embed_method: str,
    use_whitening: bool,
    feature_map_type: str = "identity",
    rff_num_features: int = 128,
    rff_sigma: float = 1.0,
    rff_seed: int = 43,
    qa_masking: bool = False,
):
    """Turn critic split hidden states into the same embedding tensors used by EBFT rewards.

    Input shapes:
    - gt_hidden_states / gen_hidden_states: (B, K, T, NF, H)
    - gt_qa_mask / gen_qa_mask:             (B, K, T, NF, 1)

    Output shapes:
    - non-token modes: (1, G, N, K, D)
    - token mode:      (1, G, N, K, T, D)
    """
    if gt_hidden_states.shape != gen_hidden_states.shape:
        raise ValueError(f"gt_hidden_states and gen_hidden_states must match, got {gt_hidden_states.shape} vs {gen_hidden_states.shape}")

    batch_size, num_blocks, generate_length, num_feat, hidden_dim = gt_hidden_states.shape
    if batch_size % int(n_samples_per_prompt) != 0:
        raise ValueError(
            f"batch_size {batch_size} must be divisible by n_samples_per_prompt {n_samples_per_prompt}"
        )

    num_groups = batch_size // int(n_samples_per_prompt)
    gt_embedding = gt_hidden_states.reshape(1, num_groups, int(n_samples_per_prompt), num_blocks, generate_length, num_feat, hidden_dim)
    gen_embedding = gen_hidden_states.reshape(1, num_groups, int(n_samples_per_prompt), num_blocks, generate_length, num_feat, hidden_dim)
    gt_mask = gt_qa_mask.reshape(1, num_groups, int(n_samples_per_prompt), num_blocks, generate_length, num_feat, 1)
    gen_mask = gen_qa_mask.reshape(1, num_groups, int(n_samples_per_prompt), num_blocks, generate_length, num_feat, 1)

    if embed_method == "mean_pooling":
        gt_embedding = torch.mean(gt_embedding, dim=-3, keepdim=True)
        gen_embedding = torch.mean(gen_embedding, dim=-3, keepdim=True)
    elif embed_method == "last_token":
        gt_embedding = groom_last_token_embedding(gt_embedding, gt_mask, qa_masking=qa_masking)
        gen_embedding = groom_last_token_embedding(gen_embedding, gen_mask, qa_masking=qa_masking)
    elif embed_method == "concat":
        gt_embedding = gt_embedding.transpose(-2, -3).reshape(
            1, num_groups, int(n_samples_per_prompt), num_blocks, 1, num_feat, generate_length * hidden_dim
        )
        gen_embedding = gen_embedding.transpose(-2, -3).reshape(
            1, num_groups, int(n_samples_per_prompt), num_blocks, 1, num_feat, generate_length * hidden_dim
        )
    elif embed_method == "token":
        pass
    else:
        raise ValueError(f"Unknown embed_method: {embed_method}")

    if use_whitening:
        gen_embedding, gt_embedding = whiten_embeddings_batched(gen_embedding, gt_embedding, whiten_tol=1e-5, normalize=False)

    gt_embedding = gt_embedding.reshape(
        gt_embedding.shape[0], gt_embedding.shape[1], gt_embedding.shape[2], gt_embedding.shape[3], gt_embedding.shape[4], gt_embedding.shape[5] * gt_embedding.shape[6]
    )
    gen_embedding = gen_embedding.reshape(
        gen_embedding.shape[0], gen_embedding.shape[1], gen_embedding.shape[2], gen_embedding.shape[3], gen_embedding.shape[4], gen_embedding.shape[5] * gen_embedding.shape[6]
    )

    if embed_method != "token":
        gt_embedding = gt_embedding.squeeze(-2)
        gen_embedding = gen_embedding.squeeze(-2)

    gen_embedding = apply_feature_map(
        gen_embedding,
        feature_map_type=feature_map_type,
        rff_num_features=rff_num_features,
        rff_sigma=rff_sigma,
        rff_seed=rff_seed,
    )
    gt_embedding = apply_feature_map(
        gt_embedding,
        feature_map_type=feature_map_type,
        rff_num_features=rff_num_features,
        rff_sigma=rff_sigma,
        rff_seed=rff_seed,
    )
    return gt_embedding, gen_embedding


@torch.no_grad()
def get_cf_l1oo_rewards(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    cf_num_freqs: int = 128,
    cf_sigma: float = 1.0,
    cf_seed: int = 43,
    cf_alpha: float = 0.5,
    cf_beta: float = 0.5,
    cf_reward_scale: float = 1.0,
    cf_target_mode: str = "single",
    cf_target_num_refs: int = 1,
    cf_target_std: float = 0.05,
    cf_target_seed: int = 43,
) -> torch.Tensor:
    """NCFM-style empirical CF reward with leave-one-out sample attribution.

    The returned reward for sample j is the marginal gain
    `D(X\\{j removed}, Y) - D(X, Y)`, where lower discrepancy is better.
    Positive reward means sample j helps reduce the group-level CF discrepancy.

    Expected shapes:
    - gen_embedding: (num_micro_batches, num_groups, n_samples, num_blocks, num_features)
    - gt_embedding:  same shape, with the sample axis containing repeated GT entries
    """
    if gen_embedding.ndim != 5 or gt_embedding.ndim != 5:
        raise ValueError(
            "get_cf_l1oo_rewards currently expects non-token embeddings with shape "
            f"(B, G, N, K, D); got {gen_embedding.shape} and {gt_embedding.shape}"
        )

    if gen_embedding.shape != gt_embedding.shape:
        raise ValueError(
            f"gen_embedding and gt_embedding must have identical shape, got "
            f"{gen_embedding.shape} vs {gt_embedding.shape}"
        )

    batch_size, num_groups, n_samples, num_blocks, feat_dim = gen_embedding.shape
    target_embedding = _build_cf_target_embedding(
        gt_embedding,
        cf_target_mode=cf_target_mode,
        cf_target_num_refs=cf_target_num_refs,
        cf_target_std=cf_target_std,
        cf_target_seed=cf_target_seed,
    )

    gen_flat = gen_embedding.permute(0, 1, 3, 2, 4).reshape(-1, n_samples, feat_dim).float()
    target_flat = target_embedding.permute(0, 1, 3, 2, 4).reshape(-1, target_embedding.shape[2], feat_dim).float()

    freqs = _get_fixed_cf_frequencies(
        input_dim=feat_dim,
        num_freqs=int(cf_num_freqs),
        sigma=float(cf_sigma),
        seed=int(cf_seed),
        device=gen_flat.device,
    )

    gen_proj = torch.einsum("fd,bnd->bfn", freqs, gen_flat)
    gen_real_vals = torch.cos(gen_proj)
    gen_imag_vals = torch.sin(gen_proj)
    gen_real = gen_real_vals.mean(dim=-1)
    gen_imag = gen_imag_vals.mean(dim=-1)

    target_proj = torch.einsum("fd,bnd->bfn", freqs, target_flat)
    target_real = torch.cos(target_proj).mean(dim=-1)
    target_imag = torch.sin(target_proj).mean(dim=-1)

    full_loss = _compute_cf_loss_terms(
        target_real, target_imag, gen_real, gen_imag, cf_alpha, cf_beta
    ).mean(dim=-1)

    if n_samples == 1:
        rewards = -full_loss.unsqueeze(-1)
    else:
        loo_real = (gen_real_vals.sum(dim=-1, keepdim=True) - gen_real_vals) / float(n_samples - 1)
        loo_imag = (gen_imag_vals.sum(dim=-1, keepdim=True) - gen_imag_vals) / float(n_samples - 1)
        loo_loss = _compute_cf_loss_terms(
            target_real.unsqueeze(-1),
            target_imag.unsqueeze(-1),
            loo_real,
            loo_imag,
            cf_alpha,
            cf_beta,
        ).mean(dim=1)
        rewards = loo_loss - full_loss.unsqueeze(-1)

    rewards = rewards.reshape(batch_size, num_groups, num_blocks, n_samples).permute(0, 1, 3, 2).contiguous()
    return rewards.to(dtype=gen_embedding.dtype) * float(cf_reward_scale)


def compute_cf_discrepancy_loss(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    cf_num_freqs: int = 128,
    cf_sigma: float = 1.0,
    cf_seed: int = 43,
    cf_alpha: float = 0.5,
    cf_beta: float = 0.5,
    cf_target_mode: str = "single",
    cf_target_num_refs: int = 1,
    cf_target_std: float = 0.05,
    cf_target_seed: int = 43,
) -> torch.Tensor:
    """Differentiable group-level CF discrepancy for direct geometry learning.

    Expected shapes:
    - gen_embedding: (B, G, N, K, D)
    - gt_embedding:  (B, G, N, K, D)
    """
    if gen_embedding.ndim != 5 or gt_embedding.ndim != 5:
        raise ValueError(
            "compute_cf_discrepancy_loss currently expects non-token embeddings with shape "
            f"(B, G, N, K, D); got {gen_embedding.shape} and {gt_embedding.shape}"
        )
    if gen_embedding.shape != gt_embedding.shape:
        raise ValueError(
            f"gen_embedding and gt_embedding must have identical shape, got {gen_embedding.shape} vs {gt_embedding.shape}"
        )

    _, _, n_samples, _, feat_dim = gen_embedding.shape
    target_embedding = _build_cf_target_embedding(
        gt_embedding,
        cf_target_mode=cf_target_mode,
        cf_target_num_refs=cf_target_num_refs,
        cf_target_std=cf_target_std,
        cf_target_seed=cf_target_seed,
    )

    gen_flat = gen_embedding.permute(0, 1, 3, 2, 4).reshape(-1, n_samples, feat_dim).float()
    target_flat = target_embedding.permute(0, 1, 3, 2, 4).reshape(-1, target_embedding.shape[2], feat_dim).float()

    freqs = _get_fixed_cf_frequencies(
        input_dim=feat_dim,
        num_freqs=int(cf_num_freqs),
        sigma=float(cf_sigma),
        seed=int(cf_seed),
        device=gen_flat.device,
    )

    gen_proj = torch.einsum("fd,bnd->bfn", freqs, gen_flat)
    gen_real = torch.cos(gen_proj).mean(dim=-1)
    gen_imag = torch.sin(gen_proj).mean(dim=-1)

    target_proj = torch.einsum("fd,bnd->bfn", freqs, target_flat)
    target_real = torch.cos(target_proj).mean(dim=-1)
    target_imag = torch.sin(target_proj).mean(dim=-1)

    full_loss = _compute_cf_loss_terms(
        target_real, target_imag, gen_real, gen_imag, cf_alpha, cf_beta
    )
    return full_loss.mean()


@torch.no_grad()
def get_cf_tokencloud_l1oo_rewards(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    cf_num_freqs: int = 128,
    cf_sigma: float = 1.0,
    cf_seed: int = 43,
    cf_alpha: float = 0.5,
    cf_beta: float = 0.5,
    cf_reward_scale: float = 1.0,
) -> torch.Tensor:
    """NCFM-style CF reward using token clouds as the empirical target measure.

    Expected shapes:
    - gen_embedding: (B, G, N, K, T, D)
    - gt_embedding:  (B, G, N, K, T, D)

    For each `(b, g, k)`, the generated-side empirical distribution is the cloud
    of all generated token features from all `N` samples in the block.
    The target-side empirical distribution is the cloud of GT token features from
    the first reference sequence in that block.

    Sample reward uses leave-one-out attribution at the *sample* level:
    removing sample `j` means removing its whole token cloud from the generated
    empirical measure.
    """
    if gen_embedding.ndim != 6 or gt_embedding.ndim != 6:
        raise ValueError(
            "get_cf_tokencloud_l1oo_rewards expects token embeddings with shape "
            f"(B, G, N, K, T, D); got {gen_embedding.shape} and {gt_embedding.shape}"
        )
    if gen_embedding.shape != gt_embedding.shape:
        raise ValueError(
            f"gen_embedding and gt_embedding must have identical shape, got "
            f"{gen_embedding.shape} vs {gt_embedding.shape}"
        )

    batch_size, num_groups, n_samples, num_blocks, num_tokens, feat_dim = gen_embedding.shape
    if n_samples <= 1:
        raise ValueError("get_cf_tokencloud_l1oo_rewards requires n_samples > 1")

    # Generated empirical measure: all tokens from all samples within a block.
    # Target empirical measure: GT token cloud from the first reference sequence.
    gen_flat = gen_embedding.permute(0, 1, 3, 2, 4, 5).reshape(-1, n_samples, num_tokens, feat_dim).float()
    target_flat = gt_embedding[:, :, :1, :, :, :].permute(0, 1, 3, 2, 4, 5).reshape(-1, 1, num_tokens, feat_dim).float()

    freqs = _get_fixed_cf_frequencies(
        input_dim=feat_dim,
        num_freqs=int(cf_num_freqs),
        sigma=float(cf_sigma),
        seed=int(cf_seed),
        device=gen_flat.device,
    )

    gen_proj = torch.einsum("fd,bntd->bfnt", freqs, gen_flat)
    gen_real_vals = torch.cos(gen_proj)
    gen_imag_vals = torch.sin(gen_proj)

    # Full generated empirical CF over N * T token features.
    gen_real = gen_real_vals.mean(dim=(-1, -2))
    gen_imag = gen_imag_vals.mean(dim=(-1, -2))

    target_proj = torch.einsum("fd,bmtd->bfmt", freqs, target_flat)
    target_real = torch.cos(target_proj).mean(dim=(-1, -2))
    target_imag = torch.sin(target_proj).mean(dim=(-1, -2))

    full_loss = _compute_cf_loss_terms(
        target_real, target_imag, gen_real, gen_imag, cf_alpha, cf_beta
    ).mean(dim=-1)

    full_real_sum = gen_real_vals.sum(dim=(-1, -2))
    full_imag_sum = gen_imag_vals.sum(dim=(-1, -2))
    sample_real_sum = gen_real_vals.sum(dim=-1)  # (B*, F, N)
    sample_imag_sum = gen_imag_vals.sum(dim=-1)
    denom = float((n_samples - 1) * num_tokens)
    loo_real = (full_real_sum.unsqueeze(-1) - sample_real_sum) / denom
    loo_imag = (full_imag_sum.unsqueeze(-1) - sample_imag_sum) / denom

    loo_loss = _compute_cf_loss_terms(
        target_real.unsqueeze(-1),
        target_imag.unsqueeze(-1),
        loo_real,
        loo_imag,
        cf_alpha,
        cf_beta,
    ).mean(dim=1)
    rewards = loo_loss - full_loss.unsqueeze(-1)

    rewards = rewards.reshape(batch_size, num_groups, num_blocks, n_samples).permute(0, 1, 3, 2).contiguous()
    return rewards.to(dtype=gen_embedding.dtype) * float(cf_reward_scale)

@torch.no_grad()
def call_rm_model(input_sequences, gt_sequences, n_samples, num_blocks, rm_actors, args, training=True, eval_dataloader_len=None):
    all_sequences = input_sequences + gt_sequences
    all_sequences = [(i // args.micro_reward_batch_size, all_sequences[i:i + args.micro_reward_batch_size]) for i in range(0, len(all_sequences), args.micro_reward_batch_size)]
    if not rm_actors:
        raise RuntimeError("No actors available in reward_model_group.")

    inflight_refs = []  # List[Tuple[int, ObjectRef, List[int]]]
    for k, (bid, batch_2d) in enumerate(all_sequences):
        a = rm_actors[k % len(rm_actors)]
        # IMPORTANT: pass a single 2D batch, not a list of batches
        ref = a.forward.remote(
            input_sequences=batch_2d,
        )
        inflight_refs.append((bid, ref))
    
    id_by_ref = {ref: bid for (bid, ref) in inflight_refs}
    pending = [ref for (_, ref) in inflight_refs]
    results_by_bid = {}  # bid -> batch_out

    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        r = ready[0]
        bid = id_by_ref[r]
        batch_out = ray.get(r)
        results_by_bid[bid] = batch_out

    # ---------- 5) Reassemble in submission order; flatten ----------
    ordered = [results_by_bid[i] for i in range(len(results_by_bid))] # list of length rollout_batch * (1+n_samples) * n_blocks // reward_micro_batch of tensors of size (reward_micro_batch, embd_dim)
    ordered_tensor = torch.cat(ordered, dim=0) # (rollout_batch * (1+n_samples) * n_blocks, embd_dim)
    gen_embeddings, gt_embeddings = ordered_tensor[:-len(gt_sequences),:], ordered_tensor[-len(gt_sequences):,:]  # (rollout_batch * n_samples * n_blocks, embd_dim) and (rollout_batch * 1 * n_blocks, embd_dim)

    # what shape do we want to end up with? take all the samples (rollout_batch*n_samples_pp). first split into batches of micro_rollout_batch size. each mrb can contain multiple prompts
    # then split each mrb into groups of size n_samples_pp. each item in this group is of size num_blocks, embed_dim. 
    # final shape is rollout_batch_size * n_samples_pp / micro rbs, micro rbs/ n_samples pp, n samples pp, num blocks, embed dim

    if training:
        gen_embeddings_reshaping_shape = (args.rollout_batch_size * n_samples // args.micro_rollout_batch_size, args.micro_rollout_batch_size // n_samples, n_samples, num_blocks, gen_embeddings.shape[-1])
        gt_embeddings_reshaping_shape = (args.rollout_batch_size * n_samples // args.micro_rollout_batch_size, args.micro_rollout_batch_size // n_samples, n_samples, num_blocks, gt_embeddings.shape[-1])
    else:
        gen_embeddings_reshaping_shape = (eval_dataloader_len, 1, n_samples, num_blocks, gen_embeddings.shape[-1])
        gt_embeddings_reshaping_shape = (eval_dataloader_len, 1, n_samples, num_blocks, gt_embeddings.shape[-1])
    gen_embeddings = gen_embeddings.reshape(gen_embeddings_reshaping_shape)
    gt_embeddings = gt_embeddings.reshape(gt_embeddings_reshaping_shape)
    return gen_embeddings, gt_embeddings

@torch.no_grad()
def compute_ngram_similarity(seq_y, seq_t, n):
    """
    Compute normalized n-gram overlap between sequences.
    This corresponds to equation (46) in the document.
    """
    # Ensure sequences are lists of ints
    if hasattr(seq_y, 'tolist'):
        seq_y = seq_y.tolist()
    if hasattr(seq_t, 'tolist'):
        seq_t = seq_t.tolist()

    # Get n-grams for both sequences
    ngrams_y = Counter([tuple(seq_y[i:i+n]) 
                        for i in range(len(seq_y) - n + 1)])
    ngrams_t = Counter([tuple(seq_t[i:i+n]) 
                        for i in range(len(seq_t) - n + 1)])
    
    # Compute numerator: sum of products for common n-grams
    common_ngrams = set(ngrams_y.keys()) & set(ngrams_t.keys())

     
    numerator = sum(ngrams_y[g] * ngrams_t[g] for g in common_ngrams)
    # Compute denominators
    norm_y = np.sqrt(sum(count**2 for count in ngrams_y.values()))
    norm_t = np.sqrt(sum(count**2 for count in ngrams_t.values()))
    
    # Return normalized similarity
    if norm_y > 0 and norm_t > 0:
        return numerator / (norm_y * norm_t)
    return 0.0

@torch.no_grad()
def get_mean_ngram_similarities(seq_y, seq_t, bleu_max_n, mean_mode):
    similarities = []
    for i in range(bleu_max_n):
        similarity = compute_ngram_similarity(seq_y, seq_t, i+1)
        similarities.append(similarity)
    similarities = torch.tensor(similarities)
    if mean_mode == "geometric":
        similarities = torch.log(similarities+1e-6)
    similarity = similarities.mean(dim=0)
    if mean_mode == "geometric":
        similarity = torch.exp(similarity)
    return similarity


@torch.no_grad()
def get_alignment_rewards(gen_embedding, gt_embedding):
    # Alignment reward: cosine similarity so the actor optimizes directional
    # alignment in embedding space (not raw vector magnitude).
    gt_rewards_tensor = F.cosine_similarity(gen_embedding, gt_embedding, dim=-1)
    return gt_rewards_tensor


@torch.no_grad()
def get_diversity_rewards(gen_embedding, per_token=False):
    if gen_embedding.shape[2] > 1:
        if per_token:
            #rollout_batch_size * n_samples_pp / micro rbs, micro rbs/ n_samples pp, n samples pp, num blocks, embed dim
            reorg = gen_embedding.permute(0,1,3,2,4,5) # num micro batches, num groups per micro batch, num blocks, n samples pp, embed dim
            n_samples_per_prompt = gen_embedding.shape[2]
            gen_embedding_unsqueeze_2 = reorg.unsqueeze(3).repeat(1,1,1,n_samples_per_prompt,1,1,1)
            gen_embedding_unsqueeze_3 = reorg.unsqueeze(4).repeat(1,1,1,1,n_samples_per_prompt,1,1)
            full_sims = torch.sum(gen_embedding_unsqueeze_2 * gen_embedding_unsqueeze_3, dim=-1) # num micro batches, num groups per micro batch, num blocks, num samples per group, num_samples_per_group
            # must zero out sim with itself. First create 2d diagonal mask
            no_jvms = torch.eye(full_sims.shape[-2], device=full_sims.device, dtype=torch.bool)
            # reshape diagonal mask to correct shape. fill full sims along this diag with zeros
            sims = full_sims.masked_fill(no_jvms.view(1,1,1,full_sims.shape[-2],full_sims.shape[-2],1), 0.0)
            # average across samples to get diversity reward for each sample
            diversity_rewards = sims.sum(dim=-2) / (full_sims.shape[-2] - 1)
            # reshape into original format
            # num micro batches, num groups per micro batch, num samples per group, num blocks
            diversity_rewards_tensor = diversity_rewards.permute(0,1,3,2,4)
        else:
            #rollout_batch_size * n_samples_pp / micro rbs, micro rbs/ n_samples pp, n samples pp, num blocks, num features, embed dim
            reorg = gen_embedding.permute(0,1,3,2,4) # num micro batches, num groups per micro batch, num blocks, n samples pp, num features, embed dim
            n_samples_per_prompt = gen_embedding.shape[2]
            gen_embedding_unsqueeze_2 = reorg.unsqueeze(3).repeat(1,1,1,n_samples_per_prompt,1,1)
            gen_embedding_unsqueeze_3 = reorg.unsqueeze(4).repeat(1,1,1,1,n_samples_per_prompt,1)
            full_sims = torch.sum(gen_embedding_unsqueeze_2 * gen_embedding_unsqueeze_3, dim=-1) # num micro batches, num groups per micro batch, num blocks, num samples per group, num_samples_per_group
            # must zero out sim with itself. First create 2d diagonal mask
            no_jvms = torch.eye(full_sims.shape[-1], device=full_sims.device, dtype=torch.bool)
            # reshape diagonal mask to correct shape. fill full sims along this diag with zeros
            sims = full_sims.masked_fill(no_jvms.view(1,1,1,full_sims.shape[-1],full_sims.shape[-1]), 0.0)
            # average across samples to get diversity reward for each sample
            diversity_rewards = sims.sum(dim=-1) / (full_sims.shape[-1] - 1)
            # reshape into original format
            # num micro batches, num groups per micro batch, num samples per group, num blocks
            diversity_rewards_tensor = diversity_rewards.permute(0,1,3,2)
    else:
        # num micro batches, num groups per micro batch, num samples per group, num blocks
        diversity_rewards_tensor = torch.zeros(gen_embedding.shape[0], gen_embedding.shape[1], gen_embedding.shape[2], gen_embedding.shape[3], device=gen_embedding.device)
    return diversity_rewards_tensor
