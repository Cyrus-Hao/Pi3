from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_SEQ = "01"  # 直接在这里改默认 sequence，例如 "00"/"05"/"10"
DEFAULT_POOL_FRAMES = 500
DEFAULT_NUM_SAMPLES = 100
DEFAULT_SAMPLING_MODE = "uniform"  # manual / uniform / stride / contiguous
DEFAULT_SAMPLING_STRIDE = 5
DEFAULT_USE_GT_POSE_PRIOR = False  # True: 使用GT pose prior, False: 不使用pose prior
DEFAULT_CHUNK_TOTAL_FRAMES = 500

# 可按 sequence 覆盖默认采样参数；不在表中的 sequence 会走全局默认值。
# 例如想测 05 序列，且希望前 800 帧均匀采样 160 帧，可加：
# "05": {"pool_frames": 800, "num_samples": 160}
SEQUENCE_SAMPLE_PRESETS: dict[str, dict[str, int]] = {
    "02": {"pool_frames": 500, "num_samples": 100},
}

ATTN_REGION_KEYS = (
    "register_register",
    "register_patch",
    "patch_patch",
    "patch_register",
)


def default_pose_prior_mode() -> str:
    return "gt" if DEFAULT_USE_GT_POSE_PRIOR else "none"


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi3.utils.geometry import depth_edge
from tools.eval_kitti_odometry import (
    align_predicted_poses_to_gt,
    align_predicted_poses_umeyama,
    compute_pose_metrics,
    find_sequence_assets,
    get_autocast_dtype,
    list_frame_paths,
    load_calibration_matrix,
    load_model,
    load_pose_file,
    load_sequence_tensors,
    resolve_kitti_root,
)
from pi3.models.layers.attention import get_attn_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Pi3X predicted poses against GT on KITTI odometry."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/Pi3/dataset/kitti-od",
        help="Path to KITTI odometry root.",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=DEFAULT_SEQ,
        help="KITTI sequence ID.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help=(
            "Sampling pool size (legacy-compatible alias): use the first N frames as the pool. "
            "If unset, use `SEQUENCE_SAMPLE_PRESETS[seq].pool_frames` or default pool size."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help=(
            "Number of sampled frames when sampling_mode is `uniform` or `contiguous`. "
            "If unset, use preset/default value."
        ),
    )
    parser.add_argument(
        "--frame_indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit 1-based frame numbers to evaluate, e.g. `1 10 20 30 40`. "
            "When provided, this enables manual mode and overrides all automatic sampling options."
        ),
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default=DEFAULT_SAMPLING_MODE,
        choices=["manual", "uniform", "stride", "contiguous"],
        help=(
            "Sampling strategy when `--frame_indices` is not provided: "
            "`uniform` (uniformly sample num_samples from pool), "
            "`stride` (sample every `sampling_stride`), "
            "`contiguous` (take first num_samples in pool)."
        ),
    )
    parser.add_argument(
        "--sampling_stride",
        type=int,
        default=DEFAULT_SAMPLING_STRIDE,
        help="Stride used when sampling_mode is `stride`.",
    )
    parser.add_argument(
        "--pool_start",
        type=int,
        default=1,
        help="1-based start frame index of sampling pool (inclusive).",
    )
    parser.add_argument(
        "--pool_end",
        type=int,
        default=None,
        help=(
            "1-based end frame index of sampling pool (inclusive). "
            "If unset, use `--num_frames` (legacy behavior) or preset/default pool size."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_outputs",
        help="Directory to save debug outputs.",
    )
    parser.add_argument(
        "--prior_scale_aug_mode",
        type=str,
        default="random",
        choices=["random"],
        help="Pi3X prior scale augmentation mode. Currently only `random` is supported.",
    )
    parser.add_argument(
        "--pose_prior_mode",
        type=str,
        default=default_pose_prior_mode(),
        choices=["gt", "none"],
        help=(
            "`gt` feeds GT poses as pose prior. "
            "`none` disables pose prior and uses only GT intrinsics."
        ),
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="full",
        choices=["full", "chunk"],
        help="`full`: infer all sampled frames in one pass. `chunk`: overlap chunk inference + Sim3 alignment.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Chunk length when inference_mode=chunk.",
    )
    parser.add_argument(
        "--chunk_total_frames",
        type=int,
        default=DEFAULT_CHUNK_TOTAL_FRAMES,
        help="In chunk mode, directly use the first N frames as chunk input pool.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=20,
        help=(
            "Overlap length between adjacent chunks when inference_mode=chunk. "
            "Default 20 gives overlap-based alignment between adjacent chunks."
        ),
    )
    parser.add_argument(
        "--chunk_conf_thre",
        type=float,
        default=0.05,
        help="Confidence threshold used for overlap alignment masks in chunk mode.",
    )
    parser.add_argument(
        "--chunk_inject_condition",
        type=str,
        nargs="+",
        default=["pose", "depth", "ray"],
        help="Chunk prior injection conditions, e.g. `pose depth ray`.",
    )
    parser.add_argument(
        "--chunk_disable_depth_ray_injection",
        action="store_true",
        help="Temporarily disable depth/ray injection in chunk mode (keep pose injection only).",
    )
    parser.add_argument(
        "--chunk_gt_pose_strategy",
        type=str,
        default="direct_gt",
        choices=["bootstrap_then_pred", "direct_gt"],
        help=(
            "When pose_prior_mode=gt in chunk mode: "
            "`bootstrap_then_pred` uses GT pose only for the first chunk, then injects previous predicted poses; "
            "`direct_gt` keeps feeding GT poses to every chunk."
        ),
    )
    parser.add_argument(
        "--compare_pose_prior_chamfer",
        action="store_true",
        help=(
            "Run chunk inference for both pose prior modes (`gt` and `none`) "
            "and compute point-cloud Chamfer distance."
        ),
    )
    parser.add_argument(
        "--chamfer_align_mode",
        type=str,
        default="sim3",
        choices=["none", "sim3"],
        help="Alignment mode before Chamfer computation between `gt` and `none` point clouds.",
    )
    parser.add_argument(
        "--chamfer_max_points",
        type=int,
        default=30000,
        help="Maximum points sampled from each cloud for Chamfer computation.",
    )
    parser.add_argument(
        "--chamfer_nn_block_size",
        type=int,
        default=2048,
        help="Block size used in nearest-neighbor distance computation for Chamfer.",
    )
    parser.add_argument(
        "--chamfer_vis_max_points",
        type=int,
        default=12000,
        help="Maximum points used for Chamfer visualization plots.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_runtime_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("Requested `cuda` but CUDA is unavailable, falling back to `cpu`.")
        return torch.device("cpu")
    return torch.device(device_arg)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_output_dir(
    base_dir: str,
    seq: str,
    num_frames: int,
    pose_prior_mode: str,
    inference_mode: str,
) -> Path:
    stem = f"pose_compare_seq{seq}_n{num_frames}_mode_{inference_mode}"
    if pose_prior_mode != "gt":
        stem = f"{stem}_poseprior_{pose_prior_mode}"
    return Path(base_dir) / stem


def normalize_sequence_id(seq: str) -> str:
    seq_str = str(seq).strip()
    if not seq_str:
        raise ValueError("Sequence ID cannot be empty.")
    return seq_str.zfill(2) if seq_str.isdigit() else seq_str


def resolve_sequence_sampling_preset(seq: str) -> tuple[int, int]:
    preset = SEQUENCE_SAMPLE_PRESETS.get(seq, {})
    pool_frames = int(preset.get("pool_frames", DEFAULT_POOL_FRAMES))
    num_samples = int(preset.get("num_samples", DEFAULT_NUM_SAMPLES))
    return pool_frames, num_samples


def select_frame_indices(
    total_frames: int,
    frame_indices_1based: list[int] | None,
    sampling_mode: str,
    pool_start_1based: int,
    pool_end_1based: int | None,
    num_samples: int,
    sampling_stride: int,
) -> tuple[list[int], dict[str, object]]:
    if total_frames <= 0:
        raise RuntimeError("No usable frames found in sequence.")

    if frame_indices_1based:
        selected_indices = [idx - 1 for idx in frame_indices_1based]
        if any(idx < 0 for idx in selected_indices):
            raise RuntimeError("`--frame_indices` must be 1-based positive integers.")
        max_valid_index = total_frames - 1
        invalid_indices = [idx + 1 for idx in selected_indices if idx > max_valid_index]
        if invalid_indices:
            raise RuntimeError(
                f"Requested frame numbers out of range: {invalid_indices}. "
                f"Valid frame numbers are 1 to {max_valid_index + 1}."
            )
        return selected_indices, {
            "sampling_mode": "manual",
            "pool_start_1based": min(frame_indices_1based),
            "pool_end_1based": max(frame_indices_1based),
            "pool_size": len(selected_indices),
            "num_samples_requested": len(selected_indices),
            "num_samples_used": len(selected_indices),
        }

    if sampling_mode == "manual":
        raise RuntimeError("`sampling_mode=manual` requires `--frame_indices`.")
    if pool_start_1based <= 0:
        raise RuntimeError("`--pool_start` must be >= 1.")
    if num_samples <= 0:
        raise RuntimeError("`--num_samples` must be positive.")
    if sampling_stride <= 0:
        raise RuntimeError("`--sampling_stride` must be positive.")

    effective_pool_end_1based = total_frames if pool_end_1based is None else pool_end_1based
    if effective_pool_end_1based <= 0:
        effective_pool_end_1based = total_frames
    effective_pool_end_1based = min(effective_pool_end_1based, total_frames)
    if pool_start_1based > effective_pool_end_1based:
        raise RuntimeError(
            f"Invalid sampling pool: pool_start={pool_start_1based} > pool_end={effective_pool_end_1based}."
        )

    pool_indices = list(range(pool_start_1based - 1, effective_pool_end_1based))
    if not pool_indices:
        raise RuntimeError("Sampling pool is empty.")

    if sampling_mode == "uniform":
        if num_samples >= len(pool_indices):
            selected_indices = pool_indices
        else:
            positions = np.linspace(0, len(pool_indices) - 1, num=num_samples, dtype=np.int64)
            selected_indices = [pool_indices[int(pos)] for pos in positions.tolist()]
    elif sampling_mode == "stride":
        selected_indices = pool_indices[::sampling_stride]
    elif sampling_mode == "contiguous":
        selected_indices = pool_indices[: min(num_samples, len(pool_indices))]
    else:
        raise RuntimeError(f"Unsupported sampling_mode: {sampling_mode}")

    if not selected_indices:
        raise RuntimeError(
            "No frame selected. Please check sampling parameters (pool range / num_samples / stride)."
        )
    return selected_indices, {
        "sampling_mode": sampling_mode,
        "pool_start_1based": pool_start_1based,
        "pool_end_1based": effective_pool_end_1based,
        "pool_size": len(pool_indices),
        "num_samples_requested": int(num_samples),
        "num_samples_used": len(selected_indices),
        "sampling_stride": int(sampling_stride),
    }


def to_first_frame_relative(poses_np: np.ndarray) -> np.ndarray:
    first_inv = np.linalg.inv(poses_np[0])
    return np.einsum("ij,njk->nik", first_inv, poses_np)


def summarize_translations(poses_np: np.ndarray) -> list[list[float]]:
    return poses_np[:, :3, 3].astype(np.float64).tolist()


def format_metric_value(value: float | int | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6f}"


def print_metric_block(name: str, metrics: dict[str, object]) -> None:
    ape_t = metrics["ape_translation_m"]
    ape_r = metrics["ape_rotation_deg"]
    rpe_t = metrics["rpe_translation_m"]
    rpe_r = metrics["rpe_rotation_deg"]
    assert isinstance(ape_t, dict)
    assert isinstance(ape_r, dict)
    assert isinstance(rpe_t, dict)
    assert isinstance(rpe_r, dict)

    print(f"[{name}] APE_t_rmse(m): {format_metric_value(ape_t['rmse'])}")
    print(f"[{name}] APE_r_rmse(deg): {format_metric_value(ape_r['rmse'])}")
    print(f"[{name}] RPE_t_rmse(m): {format_metric_value(rpe_t['rmse'])}")
    print(f"[{name}] RPE_r_rmse(deg): {format_metric_value(rpe_r['rmse'])}")


def save_pose_comparison_plot(
    sequence: str,
    gt_poses: np.ndarray,
    pred_poses: np.ndarray,
    mapped_pred_origin: np.ndarray,
    save_path: Path,
    title_suffix: str,
    pred_label: str,
    mapped_pred_origin_label: str,
) -> None:
    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz = pred_poses[:, :3, 3]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], color="green", linewidth=2.0, label="Ground Truth")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 2], color="red", linewidth=2.0, label=pred_label)
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 2], color="black", s=45, marker="o", label="Start", zorder=5)
    ax.scatter(0.0, 0.0, color="blue", s=140, marker="+", linewidths=2.0, label="World Origin (0,0,0)", zorder=6)
    ax.annotate(
        "World Origin",
        xy=(0.0, 0.0),
        xytext=(8, 8),
        textcoords="offset points",
        color="blue",
        fontsize=10,
    )
    ax.scatter(
        mapped_pred_origin[0],
        mapped_pred_origin[2],
        color="#c218d4",
        s=90,
        marker="x",
        linewidths=2.0,
        label=mapped_pred_origin_label,
        zorder=6,
    )
    ax.annotate(
        "Mapped Pred Origin",
        xy=(mapped_pred_origin[0], mapped_pred_origin[2]),
        xytext=(8, -14),
        textcoords="offset points",
        color="#c218d4",
        fontsize=10,
    )

    ax.set_title(f"Sequence {sequence} — Bird-Eye View ({title_suffix})", fontsize=15)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Z (m)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, color="#d0d0d0", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def get_frame_attention_matrix_from_block(
    blk_class: torch.nn.Module,
    x: torch.Tensor,
    frame_num: int,
    token_length: int,
    xpos: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Equivalent to `get_attn_score()` before the final sum over key-frames.
    This keeps the frame-to-frame matrix while avoiding token-token NxN materialization.
    """
    x = blk_class.norm1(x)

    batch_size, num_tokens, channels = x.shape
    qkv = blk_class.attn.qkv(x).reshape(
        batch_size,
        num_tokens,
        3,
        blk_class.attn.num_heads,
        channels // blk_class.attn.num_heads,
    )
    qkv = qkv.transpose(1, 3)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    q, k = blk_class.attn.q_norm(q).to(v.dtype), blk_class.attn.k_norm(k).to(v.dtype)

    if blk_class.attn.rope is not None:
        q = blk_class.attn.rope(q, xpos)
        k = blk_class.attn.rope(k, xpos)

    q = q.transpose(1, 2).reshape(batch_size, frame_num, token_length, blk_class.attn.num_heads, -1)
    k = k.transpose(1, 2).reshape(batch_size, frame_num, token_length, blk_class.attn.num_heads, -1)
    q = q.permute(0, 3, 1, 2, 4) # [B, H, F, N, D]
    k = k.permute(0, 3, 1, 2, 4)    

    q_mean = q.mean(dim=3) # [B, H, F, D]
    k_mean = k.mean(dim=3) # [B, H, F, D]
    score = torch.einsum("bhfd,bhgd->bhfg", q_mean * blk_class.attn.scale, k_mean).sum(dim=1)
    return score


def get_attention_region_scores_from_block(
    blk_class: torch.nn.Module,
    x: torch.Tensor,
    frame_num: int,
    token_length: int,
    patch_start_idx: int,
    xpos: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute mean attention logits between register/patch token regions
    without materializing token-token full attention matrices.
    Returns batch-wise scores for four directional regions.
    """
    if patch_start_idx <= 0 or patch_start_idx >= token_length:
        raise ValueError(
            f"Invalid patch_start_idx={patch_start_idx} for token_length={token_length}."
        )

    x = blk_class.norm1(x)
    batch_size, num_tokens, channels = x.shape
    qkv = blk_class.attn.qkv(x).reshape(
        batch_size,
        num_tokens,
        3,
        blk_class.attn.num_heads,
        channels // blk_class.attn.num_heads,
    )
    qkv = qkv.transpose(1, 3)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    q, k = blk_class.attn.q_norm(q).to(v.dtype), blk_class.attn.k_norm(k).to(v.dtype)

    if blk_class.attn.rope is not None:
        q = blk_class.attn.rope(q, xpos)
        k = blk_class.attn.rope(k, xpos)

    q = q.transpose(1, 2).reshape(batch_size, frame_num, token_length, blk_class.attn.num_heads, -1)
    k = k.transpose(1, 2).reshape(batch_size, frame_num, token_length, blk_class.attn.num_heads, -1)
    q = q.permute(0, 3, 1, 2, 4)  # [B, H, F, N, D]
    k = k.permute(0, 3, 1, 2, 4)

    q_register = q[:, :, :, :patch_start_idx, :].mean(dim=3)
    q_patch = q[:, :, :, patch_start_idx:, :].mean(dim=3)
    k_register = k[:, :, :, :patch_start_idx, :].mean(dim=3)
    k_patch = k[:, :, :, patch_start_idx:, :].mean(dim=3)

    def pair_score(q_group: torch.Tensor, k_group: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bhfd,bhgd->bhfg", q_group * blk_class.attn.scale, k_group)
        return logits.mean(dim=(1, 2, 3))

    return {
        "register_register": pair_score(q_register, k_register),
        "register_patch": pair_score(q_register, k_patch),
        "patch_patch": pair_score(q_patch, k_patch),
        "patch_register": pair_score(q_patch, k_register),
    }


def collect_global_decoder_attention(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    poses: torch.Tensor,
    use_pose_mask: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    reference_token_limit: int = 4096,
) -> tuple[torch.Tensor, list[int], dict[str, object], torch.Tensor]:
    """
    Replays Pi3X.decode() and captures frame-to-frame attention scores
    for each global-attention layer (odd decoder layers).
    """
    device = hidden.device
    if hidden.ndim == 4:
        batch_size, num_frames_from_hidden, tokens_per_frame, _ = hidden.shape
        if num_frames_from_hidden != num_frames:
            raise ValueError(
                f"Hidden frame count mismatch: hidden has {num_frames_from_hidden}, expected {num_frames}."
            )
    else:
        batch_size = hidden.shape[0] // num_frames
        tokens_per_frame = hidden.shape[1]

    hidden = hidden.reshape(batch_size * num_frames, tokens_per_frame, -1)
    register_token = model.register_token.repeat(batch_size, num_frames, 1, 1).reshape(
        batch_size * num_frames, *model.register_token.shape[-2:]
    )
    hidden = torch.cat([register_token, hidden], dim=1)
    token_length = hidden.shape[1]

    pos = model.position_getter(batch_size * num_frames, height // model.patch_size, width // model.patch_size, device)
    if model.patch_start_idx > 0:
        pos_patch = pos + 1
        pos_special = torch.zeros(batch_size * num_frames, model.patch_start_idx, 2, device=device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos_patch], dim=1)

    pose_inject_blk_idx = 0
    if model.use_multimodal:
        num_pose_frames = int(use_pose_mask.sum().item())
        if num_pose_frames == 0 or num_pose_frames == batch_size * num_frames:
            pose_inject_mask = None
        else:
            view_interaction_mask = use_pose_mask.unsqueeze(2) & use_pose_mask.unsqueeze(1)
            token_interaction_mask = view_interaction_mask.repeat_interleave(
                token_length - model.patch_start_idx, dim=1
            )
            token_interaction_mask = token_interaction_mask.repeat_interleave(
                token_length - model.patch_start_idx, dim=2
            )
            pose_inject_mask = token_interaction_mask[:, None]
    else:
        pose_inject_mask = None

    global_layer_indices: list[int] = []
    global_frame_scores: list[torch.Tensor] = []
    global_region_scores: list[torch.Tensor] = []
    reference_validation_errors: list[float] = []

    for layer_idx, blk in enumerate(model.decoder):
        if layer_idx % 2 == 0:
            pos = pos.reshape(batch_size * num_frames, token_length, -1)
            hidden = hidden.reshape(batch_size * num_frames, token_length, -1)
        else:
            pos = pos.reshape(batch_size, num_frames * token_length, -1)
            hidden = hidden.reshape(batch_size, num_frames * token_length, -1)

            frame_score = get_frame_attention_matrix_from_block(
                blk,
                hidden,
                frame_num=num_frames,
                token_length=token_length,
                xpos=pos,
            )
            region_score_dict = get_attention_region_scores_from_block(
                blk,
                hidden,
                frame_num=num_frames,
                token_length=token_length,
                patch_start_idx=model.patch_start_idx,
                xpos=pos,
            )
            region_vector = torch.stack(
                [region_score_dict[key] for key in ATTN_REGION_KEYS],
                dim=-1,
            ).mean(dim=0)
            global_layer_indices.append(layer_idx)
            global_frame_scores.append(frame_score.detach().float().cpu())
            global_region_scores.append(region_vector.detach().float().cpu())

            total_tokens = num_frames * token_length
            if total_tokens <= reference_token_limit:
                reference_score = get_attn_score(
                    blk,
                    hidden,
                    frame_num=num_frames,
                    token_length=token_length,
                    xpos=pos,
                )
                reference_error = (reference_score - frame_score.sum(dim=-1)).abs().max().item()
                reference_validation_errors.append(reference_error)

        hidden = blk(hidden, xpos=pos)

        if model.use_multimodal:
            if layer_idx in [1, 9, 17, 25, 33] and use_pose_mask.sum() > 0:
                hidden = hidden.reshape(batch_size, num_frames, -1, model.dec_embed_dim)
                poses_feat = model.pose_inject_blk[pose_inject_blk_idx](
                    hidden[..., model.patch_start_idx:, :].reshape(
                        batch_size, num_frames * (token_length - model.patch_start_idx), -1
                    ),
                    poses,
                    height,
                    width,
                    height // model.patch_size,
                    width // model.patch_size,
                    attn_mask=pose_inject_mask,
                ).reshape(batch_size, num_frames, -1, model.dec_embed_dim)
                hidden[..., model.patch_start_idx:, :] += poses_feat * use_pose_mask.view(batch_size, num_frames, 1, 1)
                hidden = hidden.reshape(batch_size, num_frames * token_length, -1)
                pose_inject_blk_idx += 1

    if not global_frame_scores:
        raise RuntimeError("No global decoder layers were collected for attention diagnostics.")

    metadata = {
        "token_length_per_frame": token_length,
        "patch_tokens_per_frame": token_length - model.patch_start_idx,
        "num_global_layers": len(global_layer_indices),
        "region_keys": list(ATTN_REGION_KEYS),
        "reference_token_limit": reference_token_limit,
        "reference_validation_ran": bool(reference_validation_errors),
        "reference_validation_max_abs_error": (
            float(max(reference_validation_errors)) if reference_validation_errors else None
        ),
    }
    return (
        torch.stack(global_frame_scores, dim=0),
        global_layer_indices,
        metadata,
        torch.stack(global_region_scores, dim=0),
    )


def compute_attention_decay(attn_matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_layers, num_frames, _ = attn_matrices.shape
    distances = np.arange(num_frames, dtype=np.int64)
    row_ids, col_ids = np.indices((num_frames, num_frames))
    frame_distance = np.abs(row_ids - col_ids)

    per_layer_decay = np.zeros((num_layers, num_frames), dtype=np.float64)
    for layer_idx in range(num_layers):
        layer_scores = attn_matrices[layer_idx]
        for distance in distances:
            mask = frame_distance == distance
            per_layer_decay[layer_idx, distance] = float(layer_scores[mask].mean())

    base = per_layer_decay[:, :1]
    safe_base = np.where(np.abs(base) < 1e-8, 1.0, base)
    per_layer_decay_normalized = per_layer_decay / safe_base
    return distances, per_layer_decay, per_layer_decay_normalized


def softmax_np(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = values - max_values
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=axis, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return exp_values / denom


def save_attention_heatmap(
    matrix: np.ndarray,
    save_path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Key Frame Index", fontsize=11)
    ax.set_ylabel("Query Frame Index", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(colorbar_label, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_attention_layer_grid(
    attn_matrices: np.ndarray,
    layer_indices: list[int],
    save_path: Path,
    title: str,
) -> None:
    num_layers = attn_matrices.shape[0]
    cols = 3
    rows = math.ceil(num_layers / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.8 * cols, 4.0 * rows),
        dpi=150,
        squeeze=False,
        constrained_layout=True,
    )
    vmin = float(attn_matrices.min())
    vmax = float(attn_matrices.max())

    for plot_idx, ax in enumerate(axes.flat):
        if plot_idx >= num_layers:
            ax.axis("off")
            continue
        im = ax.imshow(attn_matrices[plot_idx], cmap="magma", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"Global Layer {layer_indices[plot_idx]}", fontsize=10)
        ax.set_xlabel("Key Frame", fontsize=9)
        ax.set_ylabel("Query Frame", fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, pad=0.01, label="Mean attention logit score")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_attention_decay_plot(
    distances: np.ndarray,
    per_layer_decay: np.ndarray,
    save_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for layer_idx, layer_decay in enumerate(per_layer_decay):
        ax.plot(distances, layer_decay, color="#9ecae1", alpha=0.45, linewidth=1.0)

    mean_decay = per_layer_decay.mean(axis=0)
    ax.plot(distances, mean_decay, color="#d62728", linewidth=2.5, label="Mean Across Global Layers")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Absolute Frame Distance", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_attention_region_scores_plot(
    region_scores: np.ndarray,
    save_path: Path,
    title: str,
    ylabel: str = "Mean Attention Score",
) -> None:
    if region_scores.ndim != 2 or region_scores.shape[1] != len(ATTN_REGION_KEYS):
        raise ValueError(
            f"Expected region_scores shape [num_layers, {len(ATTN_REGION_KEYS)}], got {region_scores.shape}."
        )

    x_index = np.arange(region_scores.shape[0], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    style = [
        ("Register-Register", "#ff3b30"),
        ("Register-Patch", "#2e7d32"),
        ("Patch-Patch", "#0047ff"),
        ("Patch-Register", "#f5a623"),
    ]
    for column_idx, (label, color) in enumerate(style):
        ax.plot(
            x_index,
            region_scores[:, column_idx],
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            color=color,
            label=label,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Global Attention Block Index", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=10)
    if x_index.size > 0:
        tick_step = max(1, int(math.ceil(x_index.size / 20)))
        ax.set_xticks(x_index[::tick_step])
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def compute_point_mask_from_outputs(
    outputs: dict[str, torch.Tensor],
    conf_threshold: float,
) -> torch.Tensor:
    conf = torch.sigmoid(outputs["conf"][..., 0])
    non_edge = ~depth_edge(outputs["local_points"][..., 2], rtol=0.03)
    return torch.logical_and(conf > conf_threshold, non_edge)


def compute_sim3_umeyama_masked(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = src_points.shape[0]
    device = src_points.device
    src = src_points.reshape(batch_size, -1, 3)
    tgt = tgt_points.reshape(batch_size, -1, 3)

    mask = (src_mask.reshape(batch_size, -1) & tgt_mask.reshape(batch_size, -1)).float().unsqueeze(-1)
    valid_cnt = mask.sum(dim=1).squeeze(-1)
    eps = 1e-6
    bad_mask = valid_cnt < 10
    if bad_mask.all():
        return torch.eye(4, device=device).repeat(batch_size, 1, 1)

    src_mean = (src * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(batch_size, 1, 1) + eps)
    tgt_mean = (tgt * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(batch_size, 1, 1) + eps)
    src_centered = (src - src_mean) * mask
    tgt_centered = (tgt - tgt_mean) * mask

    cov = torch.bmm(src_centered.transpose(1, 2), tgt_centered)
    u, s, v = torch.svd(cov)
    r = torch.bmm(v, u.transpose(1, 2))

    det = torch.det(r)
    diag = torch.ones(batch_size, 3, device=device)
    diag[:, 2] = torch.sign(det)
    r = torch.bmm(torch.bmm(v, torch.diag_embed(diag)), u.transpose(1, 2))

    src_var = (src_centered ** 2).sum(dim=2) * mask.squeeze(-1)
    src_var = src_var.sum(dim=1) / (valid_cnt + eps)
    corrected_s = s.clone()
    corrected_s[:, 2] *= diag[:, 2]
    trace_s = corrected_s.sum(dim=1)
    scale = trace_s / (src_var * valid_cnt + eps)
    scale = scale.view(batch_size, 1, 1)

    t = tgt_mean.transpose(1, 2) - scale * torch.bmm(r, src_mean.transpose(1, 2))
    sim3 = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    sim3[:, :3, :3] = scale * r
    sim3[:, :3, 3] = t.squeeze(2)
    if bad_mask.any():
        sim3[bad_mask] = torch.eye(4, device=device).repeat(batch_size, 1, 1)[bad_mask]
    return sim3


def apply_sim3_to_points(points: torch.Tensor, sim3: torch.Tensor) -> torch.Tensor:
    batch_size = points.shape[0]
    flat_pts = points.reshape(batch_size, -1, 3)
    r_s = sim3[:, :3, :3]
    t = sim3[:, :3, 3].unsqueeze(1)
    out_pts = torch.bmm(flat_pts, r_s.transpose(1, 2)) + t
    return out_pts.reshape_as(points)


def apply_sim3_to_poses(poses: torch.Tensor, sim3: torch.Tensor) -> torch.Tensor:
    return torch.matmul(sim3.unsqueeze(1), poses)


def sim3_scale_values(sim3: torch.Tensor) -> list[float]:
    rot_scale = sim3[:, :3, :3]
    det_val = torch.det(rot_scale).detach().float().cpu().numpy()
    return (np.cbrt(np.abs(det_val)) * np.sign(det_val)).tolist()


def run_chunked_inference(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    intrinsics_t: torch.Tensor,
    gt_poses_t: torch.Tensor,
    use_pose_prior: bool,
    chunk_size: int,
    chunk_overlap: int,
    chunk_conf_thre: float,
    inject_conditions: list[str],
    gt_pose_strategy: str,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if chunk_size <= 1:
        raise RuntimeError("`--chunk_size` must be > 1 in chunk mode.")
    if chunk_overlap < 0:
        raise RuntimeError("`--chunk_overlap` must be >= 0.")
    if chunk_overlap >= chunk_size:
        raise RuntimeError("`--chunk_overlap` must be smaller than `--chunk_size`.")

    batch_size, total_frames, _, height, width = imgs.shape
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise RuntimeError("Invalid chunk step. Please set `chunk_size > chunk_overlap`.")

    merged_points: list[torch.Tensor] = []
    merged_local_points: list[torch.Tensor] = []
    merged_conf: list[torch.Tensor] = []
    merged_camera_poses: list[torch.Tensor] = []
    merged_metric: list[torch.Tensor] = []
    merged_rays: list[torch.Tensor] = []

    prev_global_pts_overlap: torch.Tensor | None = None
    prev_global_mask_overlap: torch.Tensor | None = None
    prev_aligned_poses_overlap: torch.Tensor | None = None
    prev_local_depth_overlap: torch.Tensor | None = None
    prev_local_conf_overlap: torch.Tensor | None = None
    prev_rays_overlap: torch.Tensor | None = None

    chunk_ranges: list[list[int]] = []
    overlap_scales: list[float] = []
    overlap_rmse_m: list[float | None] = []
    overlap_valid_points: list[int] = []

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    for start_idx in range(0, total_frames, step):
        end_idx = min(start_idx + chunk_size, total_frames)
        current_len = end_idx - start_idx
        if current_len <= chunk_overlap and start_idx > 0:
            break

        chunk_ranges.append([int(start_idx), int(end_idx)])
        chunk_imgs = imgs[:, start_idx:end_idx]
        model_kwargs: dict[str, object] = {"with_prior": False}

        if start_idx > 0 and chunk_overlap > 0:
            overlap_len = min(chunk_overlap, current_len)

            if "pose" in inject_conditions and prev_aligned_poses_overlap is not None:
                prior_poses = torch.eye(4, device=device).repeat(batch_size, current_len, 1, 1)
                prior_poses[:, :overlap_len] = prev_aligned_poses_overlap[:, :overlap_len]
                mask_pose = torch.zeros((batch_size, current_len), dtype=torch.bool, device=device)
                mask_pose[:, :overlap_len] = True
                model_kwargs["poses"] = prior_poses
                model_kwargs["mask_add_pose"] = mask_pose
                model_kwargs["with_prior"] = True

            if "depth" in inject_conditions and prev_local_depth_overlap is not None:
                prior_depths = torch.zeros((batch_size, current_len, height, width), device=device)
                prior_depths[:, :overlap_len] = prev_local_depth_overlap[:, :overlap_len]
                mask_depth = torch.zeros((batch_size, current_len), dtype=torch.bool, device=device)
                mask_depth[:, :overlap_len] = True
                if prev_local_conf_overlap is not None:
                    valid_depth_mask = prev_local_conf_overlap[:, :overlap_len] > chunk_conf_thre
                    prior_depths[:, :overlap_len][~valid_depth_mask] = 0.0
                model_kwargs["depths"] = prior_depths
                model_kwargs["mask_add_depth"] = mask_depth
                model_kwargs["with_prior"] = True

            if ("ray" in inject_conditions or "intrinsic" in inject_conditions) and prev_rays_overlap is not None:
                prior_rays = torch.zeros((batch_size, current_len, height, width, 3), device=device)
                prior_rays[:, :overlap_len] = prev_rays_overlap[:, :overlap_len]
                mask_ray = torch.zeros((batch_size, current_len), dtype=torch.bool, device=device)
                mask_ray[:, :overlap_len] = True
                model_kwargs["rays"] = prior_rays
                model_kwargs["mask_add_ray"] = mask_ray
                model_kwargs["with_prior"] = True

        if use_pose_prior and (
            gt_pose_strategy == "direct_gt"
            or start_idx == 0
        ):
            chunk_gt_poses = gt_poses_t[:, start_idx:end_idx]
            mask_add_pose = torch.ones((batch_size, current_len), dtype=torch.bool, device=device)
            model_kwargs["poses"] = chunk_gt_poses
            model_kwargs["mask_add_pose"] = mask_add_pose
            model_kwargs["with_prior"] = True

        # Keep GT intrinsics path available when ray priors are not injected.
        if "rays" not in model_kwargs:
            chunk_intrinsics = intrinsics_t[:, start_idx:end_idx]
            mask_add_ray = torch.ones((batch_size, current_len), dtype=torch.bool, device=device)
            model_kwargs["intrinsics"] = chunk_intrinsics
            model_kwargs["mask_add_ray"] = mask_add_ray

        with torch.inference_mode():
            with autocast_ctx:
                pred = model(chunk_imgs, **model_kwargs)

        curr_pts = pred["points"]
        curr_local_pts = pred["local_points"]
        curr_conf = pred["conf"]
        curr_poses = pred["camera_poses"]
        curr_metric = pred["metric"]
        curr_rays = pred["rays"]
        curr_mask = compute_point_mask_from_outputs(pred, conf_threshold=chunk_conf_thre)
        if int(curr_mask.sum().item()) < 10:
            flat_conf = torch.sigmoid(curr_conf[..., 0]).view(batch_size, current_len, -1)
            k = max(1, int(flat_conf.shape[-1] * 0.1))
            topk_vals, _ = torch.topk(flat_conf, k, dim=-1)
            min_vals = topk_vals[..., -1].unsqueeze(-1).unsqueeze(-1)
            curr_mask = torch.sigmoid(curr_conf[..., 0]) >= min_vals

        if start_idx == 0:
            aligned_pts = curr_pts
            aligned_poses = curr_poses
            overlap_scales.append(1.0)
            overlap_rmse_m.append(None)
            overlap_valid_points.append(0)
        else:
            assert prev_global_pts_overlap is not None and prev_global_mask_overlap is not None
            src_pts = curr_pts[:, :chunk_overlap]
            src_mask = curr_mask[:, :chunk_overlap]
            tgt_pts = prev_global_pts_overlap
            tgt_mask = prev_global_mask_overlap
            sim3 = compute_sim3_umeyama_masked(src_pts, tgt_pts, src_mask, tgt_mask)

            aligned_pts = apply_sim3_to_points(curr_pts, sim3)
            aligned_poses = apply_sim3_to_poses(curr_poses, sim3)
            overlap_scales.extend(sim3_scale_values(sim3))

            overlap_mask = src_mask & tgt_mask
            valid_count = int(overlap_mask.sum().item())
            overlap_valid_points.append(valid_count)
            if valid_count > 0:
                aligned_overlap = aligned_pts[:, :chunk_overlap]
                diff = aligned_overlap - tgt_pts
                rmse = torch.sqrt((diff[overlap_mask] ** 2).mean()).item()
                overlap_rmse_m.append(float(rmse))
            else:
                overlap_rmse_m.append(None)

        if start_idx == 0:
            merged_points.append(aligned_pts.detach().cpu())
            merged_local_points.append(curr_local_pts.detach().cpu())
            merged_conf.append(curr_conf.detach().cpu())
            merged_camera_poses.append(aligned_poses.detach().cpu())
            merged_metric.append(curr_metric.unsqueeze(1).detach().cpu())
            merged_rays.append(curr_rays.detach().cpu())
        else:
            merged_points.append(aligned_pts[:, chunk_overlap:].detach().cpu())
            merged_local_points.append(curr_local_pts[:, chunk_overlap:].detach().cpu())
            merged_conf.append(curr_conf[:, chunk_overlap:].detach().cpu())
            merged_camera_poses.append(aligned_poses[:, chunk_overlap:].detach().cpu())
            merged_metric.append(curr_metric.unsqueeze(1).detach().cpu())
            merged_rays.append(curr_rays[:, chunk_overlap:].detach().cpu())

        prev_global_pts_overlap = aligned_pts[:, -chunk_overlap:] if chunk_overlap > 0 else aligned_pts[:, :0]
        prev_global_mask_overlap = curr_mask[:, -chunk_overlap:] if chunk_overlap > 0 else curr_mask[:, :0]
        prev_aligned_poses_overlap = aligned_poses[:, -chunk_overlap:] if chunk_overlap > 0 else aligned_poses[:, :0]
        prev_local_depth_overlap = curr_local_pts[:, -chunk_overlap:, ..., 2] if chunk_overlap > 0 else curr_local_pts[:, :0, ..., 2]
        prev_local_conf_overlap = torch.sigmoid(curr_conf[:, -chunk_overlap:, ..., 0]) if chunk_overlap > 0 else curr_conf[:, :0, ..., 0]
        prev_rays_overlap = curr_rays[:, -chunk_overlap:] if chunk_overlap > 0 else curr_rays[:, :0]

        del pred, curr_pts, curr_local_pts, curr_conf, curr_poses, curr_metric, curr_rays, curr_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if end_idx == total_frames:
            break

    merged = {
        "points": torch.cat(merged_points, dim=1),
        "local_points": torch.cat(merged_local_points, dim=1),
        "conf": torch.cat(merged_conf, dim=1),
        "camera_poses": torch.cat(merged_camera_poses, dim=1),
        "metric": torch.cat(merged_metric, dim=1).mean(dim=1),
        "rays": torch.cat(merged_rays, dim=1),
    }
    diagnostics = {
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "chunk_step": int(step),
        "chunk_inject_condition": list(inject_conditions),
        "chunk_gt_pose_strategy": str(gt_pose_strategy),
        "num_chunks": len(chunk_ranges),
        "chunk_ranges_0based_half_open": chunk_ranges,
        "sim3_scales": overlap_scales,
        "overlap_alignment_rmse_m": overlap_rmse_m,
        "overlap_valid_points": overlap_valid_points,
    }
    return merged, diagnostics


def extract_masked_points(
    outputs: dict[str, torch.Tensor],
    conf_threshold: float,
) -> torch.Tensor:
    mask = compute_point_mask_from_outputs(outputs, conf_threshold=conf_threshold)
    points = outputs["points"][0][mask[0]]
    return points.detach().float()


def downsample_points(
    points: torch.Tensor,
    max_points: int,
    seed: int,
) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points
    generator = torch.Generator(device=points.device)
    generator.manual_seed(seed)
    perm = torch.randperm(points.shape[0], generator=generator, device=points.device)
    keep_idx = perm[:max_points]
    return points[keep_idx]


def pairwise_min_distances_blockwise(
    src_points: torch.Tensor,
    dst_points: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if src_points.shape[0] == 0 or dst_points.shape[0] == 0:
        return torch.empty((0,), device=src_points.device, dtype=src_points.dtype)
    if block_size <= 0:
        raise RuntimeError("`chamfer_nn_block_size` must be positive.")

    min_dists: list[torch.Tensor] = []
    for start in range(0, src_points.shape[0], block_size):
        src_block = src_points[start : start + block_size]
        block_min = None
        for dst_start in range(0, dst_points.shape[0], block_size):
            dst_block = dst_points[dst_start : dst_start + block_size]
            dists = torch.cdist(src_block, dst_block, p=2)
            current_min = dists.min(dim=1).values
            block_min = current_min if block_min is None else torch.minimum(block_min, current_min)
        assert block_min is not None
        min_dists.append(block_min)
    return torch.cat(min_dists, dim=0)


def compute_chamfer_metrics(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    block_size: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    dists_a_to_b = pairwise_min_distances_blockwise(points_a, points_b, block_size=block_size)
    dists_b_to_a = pairwise_min_distances_blockwise(points_b, points_a, block_size=block_size)

    if dists_a_to_b.numel() == 0 or dists_b_to_a.numel() == 0:
        raise RuntimeError("Empty point cloud encountered during Chamfer computation.")

    chamfer_l1_sym = float((dists_a_to_b.mean() + dists_b_to_a.mean()).item())
    chamfer_l2_sym = float(((dists_a_to_b ** 2).mean() + (dists_b_to_a ** 2).mean()).item())
    metrics = {
        "chamfer_l1_sym_m": chamfer_l1_sym,
        "chamfer_l2_sym_m2": chamfer_l2_sym,
        "a_to_b_mean_m": float(dists_a_to_b.mean().item()),
        "b_to_a_mean_m": float(dists_b_to_a.mean().item()),
    }
    return metrics, dists_a_to_b, dists_b_to_a


def align_points_sim3(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_center = src_points.mean(dim=0, keepdim=True)
    tgt_center = tgt_points.mean(dim=0, keepdim=True)
    src_zero = src_points - src_center
    tgt_zero = tgt_points - tgt_center

    cov = src_zero.transpose(0, 1) @ tgt_zero / max(src_points.shape[0], 1)
    u, s, v = torch.linalg.svd(cov)
    r = v.transpose(0, 1) @ u.transpose(0, 1)
    if torch.det(r) < 0:
        correction = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=src_points.device, dtype=src_points.dtype))
        r = v.transpose(0, 1) @ correction @ u.transpose(0, 1)

    src_var = (src_zero ** 2).sum() / max(src_points.shape[0], 1)
    scale = (s.sum() / (src_var + 1e-8)).clamp(min=1e-8)
    t = tgt_center.squeeze(0) - scale * (r @ src_center.squeeze(0))

    aligned = scale * (src_points @ r.transpose(0, 1)) + t
    sim3 = torch.eye(4, device=src_points.device, dtype=src_points.dtype)
    sim3[:3, :3] = scale * r
    sim3[:3, 3] = t
    return aligned, sim3


def save_chamfer_bev_plot(
    points_ref: np.ndarray,
    points_cmp: np.ndarray,
    save_path: Path,
    title: str,
    ref_label: str,
    cmp_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.scatter(points_ref[:, 0], points_ref[:, 2], s=0.8, c="#2e7d32", alpha=0.35, label=ref_label)
    ax.scatter(points_cmp[:, 0], points_cmp[:, 2], s=0.8, c="#c62828", alpha=0.35, label=cmp_label)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Z (m)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_chamfer_hist_plot(
    dists_a_to_b: np.ndarray,
    dists_b_to_a: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(dists_a_to_b, bins=80, alpha=0.6, color="#1f77b4", label="GT prior -> None prior")
    ax.hist(dists_b_to_a, bins=80, alpha=0.6, color="#ff7f0e", label="None prior -> GT prior")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Nearest-neighbor distance (m)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_runtime_device(args.device)
    set_seed(args.seed)
    seq = normalize_sequence_id(args.seq)

    data_root = resolve_kitti_root(args.data_root)
    preset_pool_frames, preset_num_samples = resolve_sequence_sampling_preset(seq)
    effective_num_samples = args.num_samples if args.num_samples is not None else preset_num_samples
    pool_end_1based = (
        args.pool_end
        if args.pool_end is not None
        else (args.num_frames if args.num_frames is not None else preset_pool_frames)
    )
    requested_num_frames = len(args.frame_indices) if args.frame_indices else int(effective_num_samples)
    output_dir = build_output_dir(
        base_dir=args.output_dir,
        seq=seq,
        num_frames=requested_num_frames,
        pose_prior_mode=args.pose_prior_mode,
        inference_mode=args.inference_mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    use_pose_prior = args.pose_prior_mode == "gt"

    print("=== Pi3X Pose Compare Config ===")
    print(f"data_root: {data_root}")
    print(f"seq: {seq}")
    print(f"sampling_mode: {'manual' if args.frame_indices else args.sampling_mode}")
    print(f"pool_start: {args.pool_start}")
    print(f"pool_end: {pool_end_1based}")
    print(f"num_samples: {effective_num_samples}")
    print(f"sampling_stride: {args.sampling_stride}")
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"pose prior: {args.pose_prior_mode}")
    print(f"inference_mode: {args.inference_mode}")
    chunk_inject_conditions = list(args.chunk_inject_condition)
    if args.chunk_disable_depth_ray_injection:
        chunk_inject_conditions = [
            condition for condition in chunk_inject_conditions if condition not in {"depth", "ray", "intrinsic"}
        ]
        if "pose" not in chunk_inject_conditions:
            chunk_inject_conditions.insert(0, "pose")
    if args.inference_mode == "chunk":
        print(f"chunk_total_frames: {args.chunk_total_frames}")
        print(f"chunk_size: {args.chunk_size}")
        print(f"chunk_overlap: {args.chunk_overlap}")
        print(f"chunk_conf_thre: {args.chunk_conf_thre}")
        print(f"chunk_inject_condition: {chunk_inject_conditions}")
        print(f"chunk_disable_depth_ray_injection: {args.chunk_disable_depth_ray_injection}")
        print(f"chunk_gt_pose_strategy: {args.chunk_gt_pose_strategy}")
    print(f"compare_pose_prior_chamfer: {args.compare_pose_prior_chamfer}")
    if args.compare_pose_prior_chamfer:
        print(f"chamfer_align_mode: {args.chamfer_align_mode}")
        print(f"chamfer_max_points: {args.chamfer_max_points}")
        print(f"chamfer_nn_block_size: {args.chamfer_nn_block_size}")
        print(f"chamfer_vis_max_points: {args.chamfer_vis_max_points}")
    print("intrinsics: GT")
    print(f"pose_only: {args.inference_mode == 'full'}")
    print(f"prior_scale_aug_mode: {args.prior_scale_aug_mode}")
    if not use_pose_prior:
        print("scale_aug: irrelevant because pose prior is disabled")
    else:
        print("scale_aug: 0.8 + torch.rand((B,), device=device) * 0.4")
    print()

    image_dir, pose_file, calib_file, camera_key = find_sequence_assets(data_root, seq)
    frame_paths_all = list_frame_paths(image_dir)
    gt_poses_all = load_pose_file(pose_file)
    intrinsic_np = load_calibration_matrix(calib_file, camera_key)
    total_frames = min(len(frame_paths_all), gt_poses_all.shape[0])
    if args.inference_mode == "chunk":
        chunk_total = min(args.chunk_total_frames, total_frames)
        if chunk_total <= 0:
            raise RuntimeError("No usable frames for chunk mode.")
        selected_indices = list(range(chunk_total))
        sampling_meta = {
            "sampling_mode": "chunk_first_n_contiguous",
            "pool_start_1based": 1,
            "pool_end_1based": int(chunk_total),
            "pool_size": int(chunk_total),
            "num_samples_requested": int(chunk_total),
            "num_samples_used": int(chunk_total),
            "sampling_stride": None,
        }
    else:
        selected_indices, sampling_meta = select_frame_indices(
            total_frames=total_frames,
            frame_indices_1based=args.frame_indices,
            sampling_mode=args.sampling_mode,
            pool_start_1based=args.pool_start,
            pool_end_1based=pool_end_1based,
            num_samples=effective_num_samples,
            sampling_stride=args.sampling_stride,
        )
    frame_paths = [frame_paths_all[idx] for idx in selected_indices]
    gt_poses_np = gt_poses_all[selected_indices]
    num_frames = len(selected_indices)
    imgs, gt_poses_t, intrinsics_t, tensor_metadata = load_sequence_tensors(
        frame_paths,
        gt_poses_np,
        intrinsic_np,
        device,
    )

    print("Loading Pi3X model...")
    model = load_model(device)
    model.eval()

    if args.compare_pose_prior_chamfer and args.inference_mode != "chunk":
        raise RuntimeError("`--compare_pose_prior_chamfer` currently requires `--inference_mode chunk`.")

    batch_size, num_frames_loaded, _, _, _ = imgs.shape
    autocast_dtype = get_autocast_dtype(device)
    global_attn_scores_np: np.ndarray | None = None
    global_attn_region_scores_np: np.ndarray | None = None
    global_attn_scores_softmax_np: np.ndarray | None = None
    global_attn_region_scores_softmax_np: np.ndarray | None = None
    attn_mean_matrix: np.ndarray | None = None
    global_layer_indices: list[int] = []
    attn_metadata: dict[str, object] = {}

    attn_mean_heatmap_path: Path | None = None
    attn_layer_grid_path: Path | None = None
    attn_layer_grid_softmax_path: Path | None = None
    attn_region_scores_plot_path: Path | None = None
    attn_region_scores_softmax_plot_path: Path | None = None

    chunk_diagnostics: dict[str, object] | None = None
    point_cloud_metrics: dict[str, object] | None = None
    pose_prior_chamfer_result: dict[str, object] | None = None
    chamfer_bev_plot_path: Path | None = None
    chamfer_hist_plot_path: Path | None = None

    if args.inference_mode == "full":
        mask_add_ray = torch.ones((batch_size, num_frames_loaded), dtype=torch.bool, device=device)
        mask_add_pose = torch.ones((batch_size, num_frames_loaded), dtype=torch.bool, device=device)
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                hidden, poses_for_decode, _, use_pose_mask, _ = model.encode(
                    imgs,
                    with_prior=True,
                    depths=None,
                    intrinsics=intrinsics_t,
                    poses=gt_poses_t if use_pose_prior else None,
                    rays=None,
                    mask_add_depth=None,
                    mask_add_ray=mask_add_ray,
                    mask_add_pose=mask_add_pose if use_pose_prior else None,
                )
                hidden = hidden.reshape(batch_size, num_frames_loaded, -1, model.dec_embed_dim)
                (
                    global_attn_scores_t,
                    global_layer_indices,
                    attn_metadata,
                    global_attn_region_scores_t,
                ) = collect_global_decoder_attention(
                    model=model,
                    hidden=hidden,
                    poses=poses_for_decode,
                    use_pose_mask=use_pose_mask,
                    num_frames=num_frames_loaded,
                    height=imgs.shape[-2],
                    width=imgs.shape[-1],
                )
                decoded_hidden, pos = model.decode(
                    hidden,
                    num_frames_loaded,
                    imgs.shape[-2],
                    imgs.shape[-1],
                    poses_for_decode,
                    use_pose_mask,
                )
                outputs = model.forward_head(
                    decoded_hidden,
                    pos,
                    batch_size,
                    num_frames_loaded,
                    imgs.shape[-2],
                    imgs.shape[-1],
                    imgs.shape[-2] // model.patch_size,
                    imgs.shape[-1] // model.patch_size,
                    pose_only=True,
                )

        global_attn_scores_np = global_attn_scores_t[:, 0].numpy().astype(np.float64)
        global_attn_region_scores_np = global_attn_region_scores_t.numpy().astype(np.float64)
        global_attn_scores_softmax_np = softmax_np(global_attn_scores_np, axis=-1)
        global_attn_region_scores_softmax_np = softmax_np(global_attn_region_scores_np, axis=-1)
        attn_mean_matrix = global_attn_scores_np.mean(axis=0)

        attn_mean_heatmap_path = output_dir / "decoder_global_frame_attn_mean_heatmap.png"
        attn_layer_grid_path = output_dir / "decoder_global_frame_attn_layers.png"
        attn_layer_grid_softmax_path = output_dir / "decoder_global_frame_attn_layers_softmax.png"
        attn_region_scores_plot_path = output_dir / "decoder_global_region_attn_scores.png"
        attn_region_scores_softmax_plot_path = output_dir / "decoder_global_region_attn_scores_softmax.png"
        save_attention_heatmap(
            attn_mean_matrix,
            attn_mean_heatmap_path,
            title=f"Pi3X Global Decoder Frame Attention Mean (seq {seq})",
            colorbar_label="Mean attention logit score",
        )
        save_attention_layer_grid(
            global_attn_scores_np,
            global_layer_indices,
            attn_layer_grid_path,
            title=f"Pi3X Global Decoder Frame Attention by Layer (seq {seq})",
        )
        save_attention_layer_grid(
            global_attn_scores_softmax_np,
            global_layer_indices,
            attn_layer_grid_softmax_path,
            title=f"Pi3X Global Decoder Frame Attention by Layer (softmax, seq {seq})",
        )
        save_attention_region_scores_plot(
            region_scores=global_attn_region_scores_np,
            save_path=attn_region_scores_plot_path,
            title=f"Pi3X Mean Attention Scores by Region (seq {seq})",
        )
        save_attention_region_scores_plot(
            region_scores=global_attn_region_scores_softmax_np,
            save_path=attn_region_scores_softmax_plot_path,
            title=f"Pi3X Mean Attention Scores by Region (softmax over regions, seq {seq})",
            ylabel="Mean Attention Probability",
        )
    else:
        outputs, chunk_diagnostics = run_chunked_inference(
            model=model,
            imgs=imgs,
            intrinsics_t=intrinsics_t,
            gt_poses_t=gt_poses_t,
            use_pose_prior=use_pose_prior,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunk_conf_thre=args.chunk_conf_thre,
            inject_conditions=chunk_inject_conditions,
            gt_pose_strategy=args.chunk_gt_pose_strategy,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        point_mask = compute_point_mask_from_outputs(outputs, conf_threshold=args.chunk_conf_thre)[0]
        valid_points = int(point_mask.sum().item())
        total_points = int(point_mask.numel())
        overlap_rmse = [
            float(value) for value in (chunk_diagnostics or {}).get("overlap_alignment_rmse_m", []) if value is not None
        ]
        overlap_scales = [float(v) for v in (chunk_diagnostics or {}).get("sim3_scales", [])[1:]]
        point_cloud_metrics = {
            "valid_points": valid_points,
            "total_points": total_points,
            "valid_ratio": float(valid_points / max(total_points, 1)),
            "overlap_alignment_rmse_mean_m": (float(np.mean(overlap_rmse)) if overlap_rmse else None),
            "overlap_alignment_rmse_max_m": (float(np.max(overlap_rmse)) if overlap_rmse else None),
            "overlap_sim3_scale_mean": (float(np.mean(overlap_scales)) if overlap_scales else None),
            "overlap_sim3_scale_std": (float(np.std(overlap_scales)) if overlap_scales else None),
            "overlap_sim3_scale_abs_error_mean": (
                float(np.mean(np.abs(np.asarray(overlap_scales) - 1.0))) if overlap_scales else None
            ),
        }
        if args.compare_pose_prior_chamfer:
            gt_outputs = outputs if use_pose_prior else None
            none_outputs = outputs if not use_pose_prior else None

            if gt_outputs is None:
                gt_outputs, _ = run_chunked_inference(
                    model=model,
                    imgs=imgs,
                    intrinsics_t=intrinsics_t,
                    gt_poses_t=gt_poses_t,
                    use_pose_prior=True,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    chunk_conf_thre=args.chunk_conf_thre,
                    inject_conditions=chunk_inject_conditions,
                    gt_pose_strategy=args.chunk_gt_pose_strategy,
                    device=device,
                    autocast_dtype=autocast_dtype,
                )
            if none_outputs is None:
                none_outputs, _ = run_chunked_inference(
                    model=model,
                    imgs=imgs,
                    intrinsics_t=intrinsics_t,
                    gt_poses_t=gt_poses_t,
                    use_pose_prior=False,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    chunk_conf_thre=args.chunk_conf_thre,
                    inject_conditions=chunk_inject_conditions,
                    gt_pose_strategy=args.chunk_gt_pose_strategy,
                    device=device,
                    autocast_dtype=autocast_dtype,
                )

            points_gt = extract_masked_points(gt_outputs, conf_threshold=args.chunk_conf_thre)
            points_none = extract_masked_points(none_outputs, conf_threshold=args.chunk_conf_thre)
            points_gt = downsample_points(points_gt, max_points=args.chamfer_max_points, seed=args.seed + 11)
            points_none = downsample_points(points_none, max_points=args.chamfer_max_points, seed=args.seed + 29)

            if args.chamfer_align_mode == "sim3":
                points_none_aligned, none_to_gt_sim3 = align_points_sim3(points_none, points_gt)
            else:
                points_none_aligned = points_none
                none_to_gt_sim3 = torch.eye(4, device=points_none.device, dtype=points_none.dtype)

            chamfer_metrics, dists_gt_to_none, dists_none_to_gt = compute_chamfer_metrics(
                points_gt,
                points_none_aligned,
                block_size=args.chamfer_nn_block_size,
            )

            points_gt_vis = downsample_points(
                points_gt,
                max_points=args.chamfer_vis_max_points,
                seed=args.seed + 101,
            ).detach().cpu().numpy()
            points_none_vis = downsample_points(
                points_none_aligned,
                max_points=args.chamfer_vis_max_points,
                seed=args.seed + 131,
            ).detach().cpu().numpy()

            chamfer_bev_plot_path = output_dir / "chunk_poseprior_chamfer_bev.png"
            chamfer_hist_plot_path = output_dir / "chunk_poseprior_chamfer_hist.png"
            save_chamfer_bev_plot(
                points_ref=points_gt_vis,
                points_cmp=points_none_vis,
                save_path=chamfer_bev_plot_path,
                title=f"Chunk Point Cloud BEV (GT prior vs None prior, seq {seq})",
                ref_label="GT prior",
                cmp_label="None prior (aligned)",
            )
            save_chamfer_hist_plot(
                dists_gt_to_none.detach().cpu().numpy(),
                dists_none_to_gt.detach().cpu().numpy(),
                save_path=chamfer_hist_plot_path,
                title=f"Chunk Chamfer NN Distance Histogram (seq {seq})",
            )

            pose_prior_chamfer_result = {
                "align_mode": args.chamfer_align_mode,
                "chamfer_metrics": chamfer_metrics,
                "num_points_gt": int(points_gt.shape[0]),
                "num_points_none": int(points_none.shape[0]),
                "none_to_gt_sim3": none_to_gt_sim3.detach().cpu().numpy().tolist(),
                "visualizations": {
                    "bev_overlay": str(chamfer_bev_plot_path),
                    "nn_distance_histogram": str(chamfer_hist_plot_path),
                },
            }

    pred_poses_np = outputs["camera_poses"][0].detach().float().cpu().numpy().astype(np.float64)
    gt_poses_eval_np = gt_poses_t[0].detach().float().cpu().numpy().astype(np.float64)
    pred_metric = outputs["metric"].detach().float().cpu().numpy().astype(np.float64)

    pred_poses_origin_np, mapped_pred_origin = align_predicted_poses_to_gt(pred_poses_np, gt_poses_eval_np)
    pred_poses_umeyama_np, mapped_pred_origin_umeyama = align_predicted_poses_umeyama(
        pred_poses_np,
        gt_poses_eval_np,
    )
    gt_poses_rel_np = to_first_frame_relative(gt_poses_eval_np)
    pred_poses_rel_np = to_first_frame_relative(pred_poses_np)

    metrics_raw = compute_pose_metrics(pred_poses_np, gt_poses_eval_np)
    metrics_origin = compute_pose_metrics(pred_poses_origin_np, gt_poses_eval_np)
    metrics_umeyama = compute_pose_metrics(pred_poses_umeyama_np, gt_poses_eval_np)
    metrics_first_frame_relative = compute_pose_metrics(pred_poses_rel_np, gt_poses_rel_np)
    visualization_raw_path = output_dir / "trajectory_raw_world.png"
    visualization_origin_path = output_dir / "trajectory_origin_world.png"
    visualization_umeyama_path = output_dir / "trajectory_umeyama_world.png"
    save_pose_comparison_plot(
        sequence=seq,
        gt_poses=gt_poses_eval_np,
        pred_poses=pred_poses_np,
        mapped_pred_origin=mapped_pred_origin,
        save_path=visualization_raw_path,
        title_suffix="World / Raw",
        pred_label="Pi3 Predicted (Raw)",
        mapped_pred_origin_label="Mapped Pred Origin (start-align transform)",
    )
    save_pose_comparison_plot(
        sequence=seq,
        gt_poses=gt_poses_eval_np,
        pred_poses=pred_poses_origin_np,
        mapped_pred_origin=mapped_pred_origin,
        save_path=visualization_origin_path,
        title_suffix="World / Origin-Aligned",
        pred_label="Pi3 Predicted (Origin-Aligned)",
        mapped_pred_origin_label="Mapped Pred Origin (before-align frame)",
    )
    save_pose_comparison_plot(
        sequence=seq,
        gt_poses=gt_poses_eval_np,
        pred_poses=pred_poses_umeyama_np,
        mapped_pred_origin=mapped_pred_origin_umeyama,
        save_path=visualization_umeyama_path,
        title_suffix="World / Umeyama-Aligned",
        pred_label="Pi3 Predicted (Umeyama-Aligned)",
        mapped_pred_origin_label="Mapped Pred Origin (umeyama transform)",
    )

    print("=== Pose Metrics ===")
    print_metric_block("raw", metrics_raw)
    print_metric_block("origin_aligned", metrics_origin)
    print_metric_block("umeyama_aligned", metrics_umeyama)
    print_metric_block("first_frame_relative", metrics_first_frame_relative)
    print()

    print("=== Sample Translations (XYZ, meters) ===")
    print(f"gt frame0: {gt_poses_eval_np[0, :3, 3].tolist()}")
    print(f"pred_raw frame0: {pred_poses_np[0, :3, 3].tolist()}")
    print(f"pred_origin frame0: {pred_poses_origin_np[0, :3, 3].tolist()}")
    print(f"gt_rel frame0: {gt_poses_rel_np[0, :3, 3].tolist()}")
    print(f"pred_rel frame0: {pred_poses_rel_np[0, :3, 3].tolist()}")
    print(f"gt frame{num_frames - 1}: {gt_poses_eval_np[-1, :3, 3].tolist()}")
    print(f"pred_raw frame{num_frames - 1}: {pred_poses_np[-1, :3, 3].tolist()}")
    print(f"pred_origin frame{num_frames - 1}: {pred_poses_origin_np[-1, :3, 3].tolist()}")
    print(f"gt_rel frame{num_frames - 1}: {gt_poses_rel_np[-1, :3, 3].tolist()}")
    print(f"pred_rel frame{num_frames - 1}: {pred_poses_rel_np[-1, :3, 3].tolist()}")
    print()

    if args.inference_mode == "full":
        print("=== Decoder Attention Diagnostics ===")
        print(f"global decoder layers: {global_layer_indices}")
        print(f"tokens per frame (with register tokens): {attn_metadata['token_length_per_frame']}")
        print(f"reference validation ran: {attn_metadata['reference_validation_ran']}")
        print(
            "reference validation max abs error: "
            f"{format_metric_value(attn_metadata['reference_validation_max_abs_error'])}"
        )
        print()
    else:
        print("=== Chunk Diagnostics ===")
        assert chunk_diagnostics is not None
        assert point_cloud_metrics is not None
        print(f"num_chunks: {chunk_diagnostics['num_chunks']}")
        print(f"chunk_size: {chunk_diagnostics['chunk_size']}")
        print(f"chunk_overlap: {chunk_diagnostics['chunk_overlap']}")
        print(f"valid_point_ratio: {point_cloud_metrics['valid_ratio']:.6f}")
        print(
            "overlap_rmse_mean(m): "
            f"{format_metric_value(point_cloud_metrics['overlap_alignment_rmse_mean_m'])}"
        )
        print(
            "overlap_scale_abs_err_mean: "
            f"{format_metric_value(point_cloud_metrics['overlap_sim3_scale_abs_error_mean'])}"
        )
        if pose_prior_chamfer_result is not None:
            chamfer_metrics = pose_prior_chamfer_result["chamfer_metrics"]
            assert isinstance(chamfer_metrics, dict)
            print("pose prior chamfer comparison:")
            print(f"  align_mode: {pose_prior_chamfer_result['align_mode']}")
            print(f"  chamfer_l1_sym(m): {format_metric_value(chamfer_metrics.get('chamfer_l1_sym_m'))}")
            print(f"  chamfer_l2_sym(m^2): {format_metric_value(chamfer_metrics.get('chamfer_l2_sym_m2'))}")
        print()

    summary = {
        "config": {
            "data_root": str(data_root),
            "seq": seq,
            "num_frames_requested": int(sampling_meta["num_samples_requested"]),
            "num_frames_used": int(num_frames),
            "sampling_mode": sampling_meta["sampling_mode"],
            "pool_start_1based": int(sampling_meta["pool_start_1based"]),
            "pool_end_1based": int(sampling_meta["pool_end_1based"]),
            "pool_size": int(sampling_meta["pool_size"]),
            "sampling_stride": int(sampling_meta.get("sampling_stride", args.sampling_stride)),
            "selected_frame_indices_1based": [idx + 1 for idx in selected_indices],
            "device": str(device),
            "seed": int(args.seed),
            "pose_prior_mode": args.pose_prior_mode,
            "intrinsics": "gt",
            "pose_only": args.inference_mode == "full",
            "inference_mode": args.inference_mode,
            "chunk_size": int(args.chunk_size) if args.inference_mode == "chunk" else None,
            "chunk_total_frames": int(args.chunk_total_frames) if args.inference_mode == "chunk" else None,
            "chunk_overlap": int(args.chunk_overlap) if args.inference_mode == "chunk" else None,
            "chunk_conf_thre": float(args.chunk_conf_thre) if args.inference_mode == "chunk" else None,
            "chunk_inject_condition": list(chunk_inject_conditions) if args.inference_mode == "chunk" else None,
            "chunk_disable_depth_ray_injection": (
                bool(args.chunk_disable_depth_ray_injection) if args.inference_mode == "chunk" else None
            ),
            "chunk_gt_pose_strategy": args.chunk_gt_pose_strategy if args.inference_mode == "chunk" else None,
            "compare_pose_prior_chamfer": bool(args.compare_pose_prior_chamfer),
            "chamfer_align_mode": args.chamfer_align_mode if args.compare_pose_prior_chamfer else None,
            "chamfer_max_points": int(args.chamfer_max_points) if args.compare_pose_prior_chamfer else None,
            "chamfer_nn_block_size": int(args.chamfer_nn_block_size) if args.compare_pose_prior_chamfer else None,
            "chamfer_vis_max_points": int(args.chamfer_vis_max_points) if args.compare_pose_prior_chamfer else None,
            "prior_scale_aug_mode": args.prior_scale_aug_mode,
            "frame_names": [path.name for path in frame_paths],
            "camera_key": camera_key,
            "input_image_size": {
                "width": tensor_metadata["width_orig"],
                "height": tensor_metadata["height_orig"],
            },
            "model_image_size": {
                "width": tensor_metadata["target_width"],
                "height": tensor_metadata["target_height"],
            },
            "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
        },
        "metrics": {
            "raw": metrics_raw,
            "origin_aligned": metrics_origin,
            "umeyama_aligned": metrics_umeyama,
            "first_frame_relative": metrics_first_frame_relative,
        },
        "visualizations": {
            "trajectory_raw_world": str(visualization_raw_path),
            "trajectory_origin_world": str(visualization_origin_path),
            "trajectory_umeyama_world": str(visualization_umeyama_path),
            **(
                {
                    "decoder_global_frame_attn_mean_heatmap": str(attn_mean_heatmap_path),
                    "decoder_global_frame_attn_layers": str(attn_layer_grid_path),
                    "decoder_global_frame_attn_layers_softmax": str(attn_layer_grid_softmax_path),
                    "decoder_global_region_attn_scores": str(attn_region_scores_plot_path),
                    "decoder_global_region_attn_scores_softmax": str(attn_region_scores_softmax_plot_path),
                }
                if args.inference_mode == "full"
                else {}
            ),
            **(
                {
                    "chunk_poseprior_chamfer_bev": str(chamfer_bev_plot_path),
                    "chunk_poseprior_chamfer_hist": str(chamfer_hist_plot_path),
                }
                if pose_prior_chamfer_result is not None
                else {}
            ),
        },
        "samples": {
            "gt_translations_xyz": summarize_translations(gt_poses_eval_np),
            "pred_raw_translations_xyz": summarize_translations(pred_poses_np),
            "pred_origin_aligned_translations_xyz": summarize_translations(pred_poses_origin_np),
            "pred_umeyama_aligned_translations_xyz": summarize_translations(pred_poses_umeyama_np),
            "gt_first_frame_relative_translations_xyz": summarize_translations(gt_poses_rel_np),
            "pred_first_frame_relative_translations_xyz": summarize_translations(pred_poses_rel_np),
            "pred_metric": pred_metric.reshape(-1).tolist(),
            "gt_pose_frame0": gt_poses_eval_np[0].tolist(),
            "pred_pose_raw_frame0": pred_poses_np[0].tolist(),
            "pred_pose_origin_aligned_frame0": pred_poses_origin_np[0].tolist(),
            "pred_pose_umeyama_aligned_frame0": pred_poses_umeyama_np[0].tolist(),
            "gt_pose_first_frame_relative_frame0": gt_poses_rel_np[0].tolist(),
            "pred_pose_first_frame_relative_frame0": pred_poses_rel_np[0].tolist(),
        },
        "attention_diagnostics": (
            {
                "global_layer_indices": global_layer_indices,
                "token_length_per_frame": int(attn_metadata["token_length_per_frame"]),
                "patch_tokens_per_frame": int(attn_metadata["patch_tokens_per_frame"]),
                "num_global_layers": int(attn_metadata["num_global_layers"]),
                "global_region_keys": list(attn_metadata["region_keys"]),
                "global_region_scores": {
                    "register_register": global_attn_region_scores_np[:, 0].tolist(),
                    "register_patch": global_attn_region_scores_np[:, 1].tolist(),
                    "patch_patch": global_attn_region_scores_np[:, 2].tolist(),
                    "patch_register": global_attn_region_scores_np[:, 3].tolist(),
                },
                "global_region_scores_softmax_over_regions": {
                    "register_register": global_attn_region_scores_softmax_np[:, 0].tolist(),
                    "register_patch": global_attn_region_scores_softmax_np[:, 1].tolist(),
                    "patch_patch": global_attn_region_scores_softmax_np[:, 2].tolist(),
                    "patch_register": global_attn_region_scores_softmax_np[:, 3].tolist(),
                },
                "frame_attention_softmax_axis": "key_frame",
                "reference_token_limit": int(attn_metadata["reference_token_limit"]),
                "reference_validation_ran": bool(attn_metadata["reference_validation_ran"]),
                "reference_validation_max_abs_error": attn_metadata["reference_validation_max_abs_error"],
            }
            if args.inference_mode == "full"
            else {}
        ),
        "chunk_diagnostics": chunk_diagnostics if args.inference_mode == "chunk" else None,
        "point_cloud_metrics": point_cloud_metrics if args.inference_mode == "chunk" else None,
        "pose_prior_chamfer_comparison": pose_prior_chamfer_result,
        "notes": [
            (
                f"This run uses {num_frames} sampled frames from KITTI sequence {seq} "
                "with GT intrinsics, and either GT pose prior or no pose prior depending on "
                "`pose_prior_mode`."
            ),
            *(
                [
                    (
                        "Decoder attention diagnostics use the same q/k path as `get_attn_score()` but "
                        "keep the frame-to-frame matrix instead of collapsing it to a per-frame scalar."
                    ),
                    (
                        "This run saves both pre-softmax logits and softmax-normalized diagnostics. "
                        "For frame attention matrices, softmax is applied over key frames."
                    ),
                    "For region curves, the softmax plot normalizes over the 4 region channels per layer.",
                ]
                if args.inference_mode == "full"
                else [
                    (
                        "Chunk mode uses overlap-based Sim3 alignment (similar to example_vo) "
                        "to merge local chunk predictions into a global trajectory/point cloud."
                    ),
                    (
                        "Compare runs with `pose_prior_mode=gt` vs `pose_prior_mode=none` "
                        "to inspect GT pose prior impact on overlap RMSE and Sim3 scale drift."
                    ),
                ]
            ),
            *(
                [
                    (
                        "Chamfer comparison reruns chunk inference for both pose prior settings "
                        "and computes symmetric nearest-neighbor distance between masked point clouds."
                    ),
                    (
                        "When `chamfer_align_mode=sim3`, the `none` cloud is globally aligned to the "
                        "`gt` cloud before Chamfer to isolate structural differences."
                    ),
                ]
                if pose_prior_chamfer_result is not None
                else []
            ),
            "Raw metrics test whether the model output numerically matches GT directly.",
            "Origin-aligned metrics test trajectory agreement after forcing the predicted first pose to match GT.",
            "Umeyama-aligned metrics additionally allow a global similarity alignment.",
            "First-frame-relative metrics compare inv(T0) @ Ti for both GT and prediction explicitly.",
            (
                "When `prior_scale_aug_mode=random`, Pi3X keeps its original inference-time "
                "prior scale augmentation path."
            ),
            "When `pose_prior_mode=none`, Pi3X receives only GT intrinsics and no pose prior.",
        ],
    }

    write_json(output_dir / "pose_compare_summary.json", summary)
    write_json(output_dir / "metrics_raw.json", metrics_raw)
    write_json(output_dir / "metrics_origin_aligned.json", metrics_origin)
    write_json(output_dir / "metrics_umeyama_aligned.json", metrics_umeyama)
    write_json(output_dir / "metrics_first_frame_relative.json", metrics_first_frame_relative)

    np.save(output_dir / "gt_poses.npy", gt_poses_eval_np)
    np.save(output_dir / "pred_poses_raw.npy", pred_poses_np)
    np.save(output_dir / "pred_poses_origin_aligned.npy", pred_poses_origin_np)
    np.save(output_dir / "pred_poses_umeyama_aligned.npy", pred_poses_umeyama_np)
    np.save(output_dir / "gt_poses_first_frame_relative.npy", gt_poses_rel_np)
    np.save(output_dir / "pred_poses_first_frame_relative.npy", pred_poses_rel_np)
    np.save(output_dir / "mapped_pred_origin.npy", mapped_pred_origin)
    np.save(output_dir / "mapped_pred_origin_umeyama.npy", mapped_pred_origin_umeyama)
    if args.inference_mode == "full":
        np.save(output_dir / "decoder_global_frame_attn.npy", global_attn_scores_np)
        np.save(output_dir / "decoder_global_frame_attn_softmax.npy", global_attn_scores_softmax_np)
        np.save(output_dir / "decoder_global_frame_attn_mean.npy", attn_mean_matrix)
        np.save(output_dir / "decoder_global_region_attn_scores.npy", global_attn_region_scores_np)
        np.save(output_dir / "decoder_global_region_attn_scores_softmax.npy", global_attn_region_scores_softmax_np)

    print("=== Saved Outputs ===")
    print(output_dir / "pose_compare_summary.json")
    print(output_dir / "metrics_raw.json")
    print(output_dir / "metrics_origin_aligned.json")
    print(output_dir / "metrics_umeyama_aligned.json")
    print(output_dir / "metrics_first_frame_relative.json")
    print(output_dir / "gt_poses.npy")
    print(output_dir / "pred_poses_raw.npy")
    print(output_dir / "pred_poses_origin_aligned.npy")
    print(output_dir / "pred_poses_umeyama_aligned.npy")
    print(output_dir / "gt_poses_first_frame_relative.npy")
    print(output_dir / "pred_poses_first_frame_relative.npy")
    if args.inference_mode == "full":
        print(output_dir / "decoder_global_frame_attn.npy")
        print(output_dir / "decoder_global_frame_attn_softmax.npy")
        print(output_dir / "decoder_global_frame_attn_mean.npy")
        print(output_dir / "decoder_global_region_attn_scores.npy")
        print(output_dir / "decoder_global_region_attn_scores_softmax.npy")
    print(visualization_raw_path)
    print(visualization_origin_path)
    print(visualization_umeyama_path)
    if args.inference_mode == "full":
        print(attn_mean_heatmap_path)
        print(attn_layer_grid_path)
        print(attn_layer_grid_softmax_path)
        print(attn_region_scores_plot_path)
        print(attn_region_scores_softmax_plot_path)
    if pose_prior_chamfer_result is not None:
        print(chamfer_bev_plot_path)
        print(chamfer_hist_plot_path)
    print()
    print("Conclusion:")
    if use_pose_prior:
        print(
            "If raw pose errors are near zero, Pi3X is numerically reproducing the GT pose prior. "
            "If only aligned errors are small, then the GT prior helps the trajectory shape but is not copied directly."
        )
    else:
        print(
            "This run disables pose priors, so the metrics measure how well Pi3X recovers poses "
            "from images plus GT intrinsics alone."
        )


if __name__ == "__main__":
    main()
