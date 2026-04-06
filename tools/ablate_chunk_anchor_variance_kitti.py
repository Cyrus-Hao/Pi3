from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools.ablate_chunk_stitch_drift_kitti import (
    _import_pyplot,
    apply_sim3_to_poses,
    build_gt_rays,
    canonical_online_chunk_pass,
    center_alignment_rmse,
    compute_center_alignment_transform,
    compute_pose_metrics,
    find_sequence_assets,
    get_autocast_dtype,
    get_runtime_device,
    list_frame_paths,
    load_calibration_matrix,
    load_json,
    load_model,
    load_pose_file,
    load_sequence_tensors,
    normalize_pose_rotations,
    resolve_kitti_root,
    save_json,
    sim3_scale_values,
    to_first_frame_relative,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate long-sequence pose uncertainty by treating each chunk as an "
            "independent anchor, aligning that anchor chunk to GT, propagating all "
            "other chunks through overlap-pose Sim3, and visualizing cross-anchor variance."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/Pi3/dataset/kitti-od",
        help="Path to KITTI odometry root.",
    )
    parser.add_argument("--seq", type=str, default="00", help="KITTI sequence id.")
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="0-based start frame within the full KITTI sequence.",
    )
    parser.add_argument("--num_frames", type=int, default=500, help="Use the first N contiguous frames.")
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size for inference.")
    parser.add_argument("--overlap", type=int, default=20, help="Overlap length between chunks.")
    parser.add_argument("--conf_thre", type=float, default=0.05, help="Confidence threshold for overlap masks.")
    parser.add_argument(
        "--inject_condition",
        type=str,
        nargs="+",
        default=["pose", "depth", "ray"],
        help="Overlap prior types used by the canonical online chunk inference.",
    )
    parser.add_argument(
        "--disable_gt_pose_prior",
        action="store_true",
        help="Do not feed GT poses to each chunk. Canonical online pass then falls back to overlap priors.",
    )
    parser.add_argument(
        "--disable_gt_ray_prior",
        action="store_true",
        help="Do not feed GT rays to each chunk. The model will use GT intrinsics instead.",
    )
    parser.add_argument(
        "--disable_prior_scale_aug_for_inference",
        action="store_true",
        help="Disable Pi3X inference-time prior scale augmentation.",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument(
        "--anchor_se3",
        dest="anchor_with_scale",
        action="store_false",
        help="Use SE3 instead of Sim3 when aligning the selected anchor chunk to GT.",
    )
    parser.add_argument(
        "--boundary_se3",
        dest="boundary_with_scale",
        action="store_false",
        help="Use SE3 instead of Sim3 when propagating other chunks through overlap poses.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_outputs/chunk_anchor_variance",
        help="Directory to save json/npy outputs.",
    )
    parser.add_argument(
        "--visualize_only_summary",
        type=str,
        default=None,
        help="Only render visualization figures from an existing chunk_anchor_variance_summary.json.",
    )
    parser.set_defaults(anchor_with_scale=True, boundary_with_scale=True)
    return parser.parse_args()


def compute_chunk_overlap_slices(
    source_range: tuple[int, int],
    target_range: tuple[int, int],
) -> tuple[slice, slice, tuple[int, int]]:
    overlap_start = max(source_range[0], target_range[0])
    overlap_end = min(source_range[1], target_range[1])
    if overlap_end <= overlap_start:
        raise ValueError(f"Chunks do not overlap: {source_range} vs {target_range}")
    source_slice = slice(overlap_start - source_range[0], overlap_end - source_range[0])
    target_slice = slice(overlap_start - target_range[0], overlap_end - target_range[0])
    return source_slice, target_slice, (int(overlap_start), int(overlap_end))


def align_pose_chunk_by_centers(
    source_poses: torch.Tensor,
    target_poses: torch.Tensor,
    with_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    sim3 = compute_center_alignment_transform(
        src_centers=source_poses[:, :, :3, 3],
        tgt_centers=target_poses[:, :, :3, 3],
        with_scale=with_scale,
    )
    aligned = apply_sim3_to_poses(source_poses, sim3)
    return aligned, sim3


def get_chunk_seed_poses(
    record: dict[str, object],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, str]:
    pose_key = "canonical_aligned_poses" if "canonical_aligned_poses" in record else "raw_poses"
    poses = record[pose_key]
    if not isinstance(poses, torch.Tensor):
        raise TypeError(f"Chunk record field `{pose_key}` is not a torch.Tensor.")
    return poses.to(device=device, dtype=dtype), pose_key


def compute_pairwise_chunk_alignment(
    source_record: dict[str, object],
    target_record: dict[str, object],
    target_aligned_poses: torch.Tensor,
    with_scale: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    source_range = (int(source_record["start_idx"]), int(source_record["end_idx"]))
    target_range = (int(target_record["start_idx"]), int(target_record["end_idx"]))
    source_slice, target_slice, overlap_range = compute_chunk_overlap_slices(source_range, target_range)

    source_seed, source_pose_key = get_chunk_seed_poses(source_record, device=device, dtype=dtype)
    source_overlap = source_seed[:, source_slice]
    target_overlap = target_aligned_poses[:, target_slice]

    _, sim3 = align_pose_chunk_by_centers(source_overlap, target_overlap, with_scale=with_scale)
    aligned_source = apply_sim3_to_poses(source_seed, sim3)
    overlap_len = overlap_range[1] - overlap_range[0]
    overlap_rmse = center_alignment_rmse(aligned_source[:, source_slice], target_overlap[:, :, :3, 3], overlap_len)

    return aligned_source, sim3, {
        "source_chunk_idx": int(source_record["chunk_idx"]),
        "target_chunk_idx": int(target_record["chunk_idx"]),
        "source_pose_key": source_pose_key,
        "overlap_range_0based_half_open": [int(overlap_range[0]), int(overlap_range[1])],
        "overlap_len": int(overlap_len),
        "with_scale": bool(with_scale),
        "scale": float(sim3_scale_values(sim3)[0]),
        "overlap_center_rmse_m": overlap_rmse,
    }


def merge_aligned_chunks(
    aligned_chunks: list[torch.Tensor],
    chunk_records: list[dict[str, object]],
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    merged = np.zeros((total_frames, 4, 4), dtype=np.float64)
    owner_chunk_idx = np.full((total_frames,), -1, dtype=np.int64)

    for record, aligned_chunk in zip(chunk_records, aligned_chunks):
        start_idx = int(record["start_idx"])
        end_idx = int(record["end_idx"])
        local_np = aligned_chunk[0].detach().cpu().numpy().astype(np.float64)
        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            if owner_chunk_idx[global_idx] >= 0:
                continue
            merged[global_idx] = local_np[local_idx]
            owner_chunk_idx[global_idx] = int(record["chunk_idx"])

    if np.any(owner_chunk_idx < 0):
        missing = np.where(owner_chunk_idx < 0)[0].tolist()
        raise RuntimeError(f"Failed to merge all frames, missing indices: {missing}")
    return merged, owner_chunk_idx


def summarize_anchor_trajectory(pred_poses_np: np.ndarray, gt_poses_np: np.ndarray) -> dict[str, object]:
    pred_norm = normalize_pose_rotations(pred_poses_np)
    gt_norm = normalize_pose_rotations(gt_poses_np)
    pred_rel = to_first_frame_relative(pred_norm)
    gt_rel = to_first_frame_relative(gt_norm)
    metrics_raw = compute_pose_metrics(pred_norm, gt_norm)
    metrics_rel = compute_pose_metrics(pred_rel, gt_rel)
    return {
        "metrics": {
            "raw_vs_gt_world": metrics_raw,
            "first_frame_relative_vs_gt_world": metrics_rel,
        },
        "error_scalars": {
            "raw_ape_translation_rmse_m": float(metrics_raw["ape_translation_m"]["rmse"]),
            "raw_ape_rotation_rmse_deg": float(metrics_raw["ape_rotation_deg"]["rmse"]),
            "first_frame_relative_ape_translation_rmse_m": float(metrics_rel["ape_translation_m"]["rmse"]),
            "first_frame_relative_ape_rotation_rmse_deg": float(metrics_rel["ape_rotation_deg"]["rmse"]),
            "last_frame_translation_error_m": float(metrics_rel["per_frame"]["ape_translation_m"][-1]),
            "last_frame_rotation_error_deg": float(metrics_rel["per_frame"]["ape_rotation_deg"][-1]),
        },
    }


def reexpress_poses_in_world_frame(poses_np: np.ndarray, world_from_reference_np: np.ndarray) -> np.ndarray:
    reference_inv = np.linalg.inv(world_from_reference_np)
    return np.einsum("ij,njk->nik", reference_inv, poses_np)


def build_full_sequence_from_anchor(
    chunk_records: list[dict[str, object]],
    gt_poses_t: torch.Tensor,
    anchor_chunk_idx: int,
    anchor_with_scale: bool,
    boundary_with_scale: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    if not chunk_records:
        raise RuntimeError("No chunk records found.")
    if anchor_chunk_idx < 0 or anchor_chunk_idx >= len(chunk_records):
        raise IndexError(f"Invalid anchor chunk index: {anchor_chunk_idx}")

    device = gt_poses_t.device
    dtype = gt_poses_t.dtype
    aligned_chunks: list[torch.Tensor | None] = [None for _ in chunk_records]
    boundary_steps: list[dict[str, object]] = []

    anchor_record = chunk_records[anchor_chunk_idx]
    anchor_start = int(anchor_record["start_idx"])
    anchor_end = int(anchor_record["end_idx"])
    anchor_seed, anchor_pose_key = get_chunk_seed_poses(anchor_record, device=device, dtype=dtype)
    anchor_gt = gt_poses_t[:, anchor_start:anchor_end]
    anchor_aligned, anchor_sim3 = align_pose_chunk_by_centers(anchor_seed, anchor_gt, with_scale=anchor_with_scale)
    aligned_chunks[anchor_chunk_idx] = anchor_aligned

    for idx in range(anchor_chunk_idx + 1, len(chunk_records)):
        current_record = chunk_records[idx]
        target_record = chunk_records[idx - 1]
        target_aligned = aligned_chunks[idx - 1]
        assert target_aligned is not None
        aligned_current, _, step_diag = compute_pairwise_chunk_alignment(
            source_record=current_record,
            target_record=target_record,
            target_aligned_poses=target_aligned,
            with_scale=boundary_with_scale,
            device=device,
            dtype=dtype,
        )
        aligned_chunks[idx] = aligned_current
        step_diag["direction"] = "right"
        boundary_steps.append(step_diag)

    for idx in range(anchor_chunk_idx - 1, -1, -1):
        current_record = chunk_records[idx]
        target_record = chunk_records[idx + 1]
        target_aligned = aligned_chunks[idx + 1]
        assert target_aligned is not None
        aligned_current, _, step_diag = compute_pairwise_chunk_alignment(
            source_record=current_record,
            target_record=target_record,
            target_aligned_poses=target_aligned,
            with_scale=boundary_with_scale,
            device=device,
            dtype=dtype,
        )
        aligned_chunks[idx] = aligned_current
        step_diag["direction"] = "left"
        boundary_steps.append(step_diag)

    finalized_chunks = [chunk for chunk in aligned_chunks if chunk is not None]
    if len(finalized_chunks) != len(chunk_records):
        raise RuntimeError("Some chunks were not aligned during anchor propagation.")

    merged_poses_np, owner_chunk_idx = merge_aligned_chunks(
        aligned_chunks=[chunk for chunk in aligned_chunks if chunk is not None],
        chunk_records=chunk_records,
        total_frames=int(gt_poses_t.shape[1]),
    )
    gt_poses_np = gt_poses_t[0].detach().cpu().numpy().astype(np.float64)
    anchor_summary = summarize_anchor_trajectory(merged_poses_np, gt_poses_np)

    return merged_poses_np, {
        "anchor_chunk_idx": int(anchor_chunk_idx),
        "anchor_range_0based_half_open": [anchor_start, anchor_end],
        "anchor_pose_key": anchor_pose_key,
        "anchor_with_scale": bool(anchor_with_scale),
        "boundary_with_scale": bool(boundary_with_scale),
        "anchor_gt_alignment_scale": float(sim3_scale_values(anchor_sim3)[0]),
        "anchor_gt_alignment_rmse_m": center_alignment_rmse(anchor_aligned, anchor_gt[:, :, :3, 3], anchor_end - anchor_start),
        "anchor_gt_alignment_sim3": anchor_sim3[0].detach().cpu().numpy().astype(np.float64).tolist(),
        "boundary_steps": sorted(boundary_steps, key=lambda item: (item["source_chunk_idx"], item["target_chunk_idx"])),
        "merged_owner_chunk_idx_per_frame": owner_chunk_idx.tolist(),
        "trajectory_summary": anchor_summary,
    }


def build_cross_anchor_statistics(
    anchor_trajectories_np: np.ndarray,
    chunk_records: list[dict[str, object]],
    gt_poses_np: np.ndarray,
) -> dict[str, object]:
    translations = anchor_trajectories_np[:, :, :3, 3]
    mean_translation = np.mean(translations, axis=0)
    translation_var = np.var(translations, axis=0)
    translation_std = np.sqrt(translation_var)
    total_std = np.sqrt(np.sum(translation_var, axis=1))
    total_var = np.sum(translation_var, axis=1)

    mean_pose = gt_poses_np.copy()
    mean_pose[:, :3, 3] = mean_translation

    gt_translation = gt_poses_np[:, :3, 3]
    per_anchor_deviation = np.linalg.norm(translations - mean_translation[None], axis=-1)
    per_anchor_gt_error = np.linalg.norm(translations - gt_translation[None], axis=-1)
    per_chunk_stats = []
    per_anchor_chunk_mean_deviation = []
    per_anchor_chunk_mean_squared_deviation = []
    per_anchor_chunk_mean_gt_error = []
    per_anchor_chunk_rmse_gt_error = []
    for anchor_idx in range(per_anchor_deviation.shape[0]):
        anchor_mean_dev_row = []
        anchor_mean_sq_dev_row = []
        anchor_mean_gt_error_row = []
        anchor_rmse_gt_error_row = []
        for record in chunk_records:
            start_idx = int(record["start_idx"])
            end_idx = int(record["end_idx"])
            chunk_anchor_dev = per_anchor_deviation[anchor_idx, start_idx:end_idx]
            chunk_anchor_gt_error = per_anchor_gt_error[anchor_idx, start_idx:end_idx]
            anchor_mean_dev_row.append(float(np.mean(chunk_anchor_dev)))
            anchor_mean_sq_dev_row.append(float(np.mean(np.square(chunk_anchor_dev))))
            anchor_mean_gt_error_row.append(float(np.mean(chunk_anchor_gt_error)))
            anchor_rmse_gt_error_row.append(float(np.sqrt(np.mean(np.square(chunk_anchor_gt_error)))))
        per_anchor_chunk_mean_deviation.append(anchor_mean_dev_row)
        per_anchor_chunk_mean_squared_deviation.append(anchor_mean_sq_dev_row)
        per_anchor_chunk_mean_gt_error.append(anchor_mean_gt_error_row)
        per_anchor_chunk_rmse_gt_error.append(anchor_rmse_gt_error_row)

    for record in chunk_records:
        start_idx = int(record["start_idx"])
        end_idx = int(record["end_idx"])
        chunk_total_std = total_std[start_idx:end_idx]
        per_chunk_stats.append(
            {
                "chunk_idx": int(record["chunk_idx"]),
                "range_0based_half_open": [start_idx, end_idx],
                "mean_translation_std_total_m": float(np.mean(chunk_total_std)),
                "max_translation_std_total_m": float(np.max(chunk_total_std)),
            }
        )

    return {
        "num_anchor_trajectories": int(anchor_trajectories_np.shape[0]),
        "num_frames": int(anchor_trajectories_np.shape[1]),
        "translation_mean_xyz_m": mean_translation.tolist(),
        "translation_axis_var_m2": {
            "x": translation_var[:, 0].tolist(),
            "y": translation_var[:, 1].tolist(),
            "z": translation_var[:, 2].tolist(),
        },
        "translation_axis_std_m": {
            "x": translation_std[:, 0].tolist(),
            "y": translation_std[:, 1].tolist(),
            "z": translation_std[:, 2].tolist(),
        },
        "translation_var_trace_m2": total_var.tolist(),
        "translation_std_total_m": total_std.tolist(),
        "mean_translation_std_total_m": float(np.mean(total_std)),
        "median_translation_std_total_m": float(np.median(total_std)),
        "max_translation_std_total_m": float(np.max(total_std)),
        "per_anchor_translation_deviation_to_mean_m": per_anchor_deviation.tolist(),
        "per_anchor_chunk_mean_translation_deviation_to_mean_m": per_anchor_chunk_mean_deviation,
        "per_anchor_chunk_mean_squared_translation_deviation_to_mean_m2": per_anchor_chunk_mean_squared_deviation,
        "per_anchor_translation_error_to_gt_m": per_anchor_gt_error.tolist(),
        "per_anchor_chunk_mean_translation_error_to_gt_m": per_anchor_chunk_mean_gt_error,
        "per_anchor_chunk_rmse_translation_error_to_gt_m": per_anchor_chunk_rmse_gt_error,
        "per_chunk_stats": per_chunk_stats,
        "mean_trajectory_summary": summarize_anchor_trajectory(mean_pose, gt_poses_np),
    }


def render_visualizations(summary: dict[str, object], output_dir: Path) -> list[Path]:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("viz_*.png"):
        stale_path.unlink(missing_ok=True)
    saved_paths: list[Path] = []

    aggregate = summary["aggregate"]
    gt_poses_np = np.load(output_dir / "gt_poses_world.npy").astype(np.float64)
    anchor_trajectories_np = np.load(output_dir / "anchor_trajectories_world.npy").astype(np.float64)
    sequence_start_frame = int(summary["config"].get("start_frame", 0))
    chunk_ranges = [item["range_0based_half_open"] for item in aggregate["per_chunk_stats"]]
    chunk_labels = [
        f"{sequence_start_frame + int(start)}-{sequence_start_frame + int(end)}" for start, end in chunk_ranges
    ]

    anchor_chunk_mean_gt_error = aggregate.get("per_anchor_chunk_mean_translation_error_to_gt_m")
    anchor_chunk_rmse_gt_error = aggregate.get("per_anchor_chunk_rmse_translation_error_to_gt_m")
    if anchor_chunk_mean_gt_error is None or anchor_chunk_rmse_gt_error is None:
        gt_translation = gt_poses_np[:, :3, 3]
        per_anchor_gt_error = np.linalg.norm(anchor_trajectories_np[:, :, :3, 3] - gt_translation[None], axis=-1)
        anchor_chunk_mean_gt_error_rows = []
        anchor_chunk_rmse_gt_error_rows = []
        for anchor_idx in range(per_anchor_gt_error.shape[0]):
            anchor_mean_gt_error_row = []
            anchor_rmse_gt_error_row = []
            for start_idx, end_idx in chunk_ranges:
                chunk_anchor_gt_error = per_anchor_gt_error[anchor_idx, int(start_idx):int(end_idx)]
                anchor_mean_gt_error_row.append(float(np.mean(chunk_anchor_gt_error)))
                anchor_rmse_gt_error_row.append(float(np.sqrt(np.mean(np.square(chunk_anchor_gt_error)))))
            anchor_chunk_mean_gt_error_rows.append(anchor_mean_gt_error_row)
            anchor_chunk_rmse_gt_error_rows.append(anchor_rmse_gt_error_row)
        anchor_chunk_mean_gt_error = np.asarray(anchor_chunk_mean_gt_error_rows, dtype=np.float64)
        anchor_chunk_rmse_gt_error = np.asarray(anchor_chunk_rmse_gt_error_rows, dtype=np.float64)
    else:
        anchor_chunk_mean_gt_error = np.asarray(anchor_chunk_mean_gt_error, dtype=np.float64)
        anchor_chunk_rmse_gt_error = np.asarray(anchor_chunk_rmse_gt_error, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(anchor_chunk_rmse_gt_error, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("Anchor chunk -> target chunk GT error")
    ax.set_ylabel("Anchor chunk index")
    ax.set_xlabel("Target chunk range")
    ax.set_xticks(np.arange(len(chunk_labels), dtype=np.int32))
    ax.set_xticklabels(chunk_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(anchor_chunk_rmse_gt_error.shape[0], dtype=np.int32))
    fig.colorbar(im, ax=ax, label="Translation RMSE to GT (m)")
    max_rmse = float(np.max(anchor_chunk_rmse_gt_error)) if anchor_chunk_rmse_gt_error.size > 0 else 0.0
    text_threshold = max_rmse * 0.5 if max_rmse > 0.0 else 0.0
    for row_idx in range(anchor_chunk_rmse_gt_error.shape[0]):
        for col_idx in range(anchor_chunk_rmse_gt_error.shape[1]):
            text_color = "white" if anchor_chunk_rmse_gt_error[row_idx, col_idx] > text_threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{anchor_chunk_mean_gt_error[row_idx, col_idx]:.1f}m",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    fig.tight_layout()
    path = output_dir / "viz_anchor_to_gt_chunk_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)

    seq = str(summary["config"]["seq"])
    anchors = summary["anchors"]
    for anchor_idx, anchor_summary in enumerate(anchors):
        anchor_poses_viz = reexpress_poses_in_world_frame(
            anchor_trajectories_np[anchor_idx],
            anchor_trajectories_np[anchor_idx, 0],
        )
        gt_poses_viz = reexpress_poses_in_world_frame(
            gt_poses_np,
            anchor_trajectories_np[anchor_idx, 0],
        )
        anchor_xyz = anchor_poses_viz[:, :3, 3]
        gt_xyz = gt_poses_viz[:, :3, 3]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], color="green", linewidth=2.0, label="Ground Truth", zorder=2)
        ax.plot(
            anchor_xyz[:, 0],
            anchor_xyz[:, 2],
            color="red",
            linewidth=1.6,
            label="Pi3 Predicted (pred-frame-0 world)",
            zorder=3,
        )
        ax.scatter(
            [float(gt_xyz[0, 0])],
            [float(gt_xyz[0, 2])],
            c="black",
            s=45,
            zorder=6,
            label="GT Start",
        )
        ax.plot(
            [0.0],
            [0.0],
            marker="+",
            color="tab:blue",
            markersize=12,
            markeredgewidth=2.0,
            linestyle="none",
            label="Pred Frame 0 / Viz Origin",
            zorder=5,
        )
        anchor_sim3 = np.asarray(anchor_summary.get("anchor_gt_alignment_sim3"), dtype=np.float64)
        if anchor_sim3.shape == (4, 4):
            mapped_origin = np.array([anchor_sim3[0, 3], anchor_sim3[1, 3], anchor_sim3[2, 3], 1.0], dtype=np.float64)
            mapped_origin_viz = np.linalg.inv(anchor_trajectories_np[anchor_idx, 0]) @ mapped_origin
            mapped_origin_x = float(mapped_origin_viz[0])
            mapped_origin_z = float(mapped_origin_viz[2])
            ax.plot(
                [mapped_origin_x],
                [mapped_origin_z],
                marker="x",
                color="#d000ff",
                markersize=12,
                markeredgewidth=2.4,
                linestyle="none",
                label="Anchor Local Origin",
                zorder=5,
            )
            ax.annotate(
                "Anchor Local Origin",
                (mapped_origin_x, mapped_origin_z),
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=9,
                color="#d000ff",
            )
        anchor_range = anchor_summary["anchor_range_0based_half_open"]
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="-", alpha=0.35)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(
            f"Sequence {seq} — Anchor Chunk {anchor_idx} "
            f"[{sequence_start_frame + int(anchor_range[0])}, {sequence_start_frame + int(anchor_range[1])}) "
            f"(pred frame 0 world)"
        )
        ax.legend(loc="upper left", framealpha=0.92)
        fig.tight_layout()
        path = output_dir / f"viz_anchor_bev_chunk{anchor_idx:02d}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    notes = [
        (
            f"Sequence: {summary['config']['seq']} | Start frame: {sequence_start_frame} | "
            f"Frames: {summary['config']['num_frames']} | "
            f"Chunk: {summary['config']['chunk_size']} | Overlap: {summary['config']['overlap']}"
        ),
        f"Anchor-to-GT: {'Sim3' if summary['config']['anchor_with_scale'] else 'SE3'}",
        f"Boundary propagation: {'Sim3' if summary['config']['boundary_with_scale'] else 'SE3'}",
        "",
        "Files:",
    ]
    notes.extend([f"- {path.name}" for path in saved_paths])
    notes.extend(
        [
            "",
            "Interpretation:",
            "- viz_anchor_to_gt_chunk_heatmap.png: row=anchor chunk, col=target chunk span; color shows translation RMSE to GT, text shows mean translation error to GT in meters.",
            "- viz_anchor_bev_chunkXX.png: GT and one anchor-built full trajectory in BEV, both re-expressed in the aligned predicted frame-0 coordinate system; blue plus marks the visualization origin and magenta X marks the anchor local-frame origin before alignment.",
        ]
    )
    notes_path = output_dir / "viz_README.txt"
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    saved_paths.append(notes_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    if args.visualize_only_summary is not None:
        summary_path = Path(args.visualize_only_summary).expanduser().resolve()
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary json does not exist: {summary_path}")
        print(f"Rendering visualizations from existing summary: {summary_path}")
        summary = load_json(summary_path)
        saved_paths = render_visualizations(summary, summary_path.parent)
        print("Saved visualizations:")
        for path in saved_paths:
            print(path)
        return

    device = get_runtime_device(args.device)
    seq = str(args.seq).zfill(2)

    data_root = resolve_kitti_root(args.data_root)
    image_dir, pose_file, calib_file, camera_key = find_sequence_assets(data_root, seq)
    frame_paths_all = list_frame_paths(image_dir)
    gt_poses_all = load_pose_file(pose_file)
    intrinsic_np = load_calibration_matrix(calib_file, camera_key)

    total_frames = min(len(frame_paths_all), gt_poses_all.shape[0])
    start_frame = int(args.start_frame)
    if start_frame < 0:
        raise ValueError(f"--start_frame must be >= 0, got {start_frame}")
    if start_frame >= total_frames:
        raise ValueError(
            f"--start_frame={start_frame} is out of range for sequence {seq} with {total_frames} frames."
        )

    available_frames = total_frames - start_frame
    num_frames = min(args.num_frames, available_frames)
    end_frame = start_frame + num_frames
    frame_paths = frame_paths_all[start_frame:end_frame]
    gt_poses_np = gt_poses_all[start_frame:end_frame]

    output_dir = Path(args.output_dir) / (
        f"seq{seq}_start{start_frame}_n{num_frames}_chunk{args.chunk_size}_overlap{args.overlap}"
        f"_anchor{'sim3' if args.anchor_with_scale else 'se3'}"
        f"_boundary{'sim3' if args.boundary_with_scale else 'se3'}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading KITTI sequence {seq} frames [{start_frame}, {end_frame}) "
        f"with {num_frames} contiguous frames..."
    )
    imgs, gt_poses_t, intrinsics_t, tensor_metadata = load_sequence_tensors(
        frame_paths=frame_paths,
        poses_np=gt_poses_np,
        intrinsic_np=intrinsic_np,
        device=device,
    )
    gt_rays_t = build_gt_rays(
        intrinsics=intrinsics_t,
        height=int(tensor_metadata["target_height"]),
        width=int(tensor_metadata["target_width"]),
    )

    model = load_model(device, args.ckpt)
    model.disable_prior_scale_aug_for_inference = bool(args.disable_prior_scale_aug_for_inference)
    autocast_dtype = get_autocast_dtype(device)

    print("Running canonical online chunk inference once...")
    with torch.inference_mode():
        _, chunk_records, baseline_diag = canonical_online_chunk_pass(
            model=model,
            imgs=imgs,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            conf_thre=args.conf_thre,
            inject_condition=list(args.inject_condition),
            dtype=autocast_dtype,
            poses=None if args.disable_gt_pose_prior else gt_poses_t,
            rays=None if args.disable_gt_ray_prior else gt_rays_t,
            intrinsics=intrinsics_t if args.disable_gt_ray_prior else None,
        )

    print(f"Rebuilding full sequence from {len(chunk_records)} anchor chunks...")
    anchor_trajectories = []
    anchor_summaries = []
    for anchor_idx in range(len(chunk_records)):
        print(f"  > Anchor chunk {anchor_idx + 1}/{len(chunk_records)}")
        anchor_poses_np, anchor_summary = build_full_sequence_from_anchor(
            chunk_records=chunk_records,
            gt_poses_t=gt_poses_t,
            anchor_chunk_idx=anchor_idx,
            anchor_with_scale=bool(args.anchor_with_scale),
            boundary_with_scale=bool(args.boundary_with_scale),
        )
        anchor_trajectories.append(anchor_poses_np)
        anchor_summaries.append(anchor_summary)

    anchor_trajectories_np = np.stack(anchor_trajectories, axis=0)
    aggregate = build_cross_anchor_statistics(
        anchor_trajectories_np=anchor_trajectories_np,
        chunk_records=chunk_records,
        gt_poses_np=gt_poses_np,
    )
    mean_poses_np = gt_poses_np.copy()
    mean_poses_np[:, :3, 3] = np.asarray(aggregate["translation_mean_xyz_m"], dtype=np.float64)

    summary = {
        "config": {
            "data_root": str(data_root),
            "seq": seq,
            "start_frame": int(start_frame),
            "num_frames": int(num_frames),
            "end_frame_exclusive": int(end_frame),
            "sequence_frame_range_0based_half_open": [int(start_frame), int(end_frame)],
            "chunk_size": int(args.chunk_size),
            "overlap": int(args.overlap),
            "conf_thre": float(args.conf_thre),
            "inject_condition": list(args.inject_condition),
            "use_gt_pose_prior": not args.disable_gt_pose_prior,
            "use_gt_ray_prior": not args.disable_gt_ray_prior,
            "disable_prior_scale_aug_for_inference": bool(args.disable_prior_scale_aug_for_inference),
            "device": str(device),
            "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
            "camera_key": camera_key,
            "anchor_with_scale": bool(args.anchor_with_scale),
            "boundary_with_scale": bool(args.boundary_with_scale),
            "model_image_size": {
                "width": int(tensor_metadata["target_width"]),
                "height": int(tensor_metadata["target_height"]),
            },
        },
        "canonical_chunk_inference": {
            "diagnostics": baseline_diag,
            "num_chunks": len(chunk_records),
            "chunk_ranges_0based_half_open": [
                [int(record["start_idx"]), int(record["end_idx"])] for record in chunk_records
            ],
            "chunk_ranges_in_sequence_0based_half_open": [
                [int(start_frame + int(record["start_idx"])), int(start_frame + int(record["end_idx"]))]
                for record in chunk_records
            ],
        },
        "anchors": anchor_summaries,
        "aggregate": aggregate,
        "notes": [
            "The model is run only once with the canonical chunked online pipeline. All later analyses reuse the saved raw pose chunk predictions.",
            "Each anchor experiment aligns one selected chunk to GT world coordinates, then propagates every other chunk to that world through overlap-pose center Sim3/SE3 alignment.",
            "Cross-anchor variance is computed from the final full-sequence trajectories after all anchor-specific propagations are finished.",
            "Pose metrics project each pose 3x3 block to the nearest SO(3) rotation before evaluating rotation errors.",
        ],
    }

    save_json(output_dir / "chunk_anchor_variance_summary.json", summary)
    np.save(output_dir / "gt_poses_world.npy", gt_poses_np.astype(np.float64))
    np.save(output_dir / "anchor_trajectories_world.npy", anchor_trajectories_np.astype(np.float64))
    np.save(output_dir / "anchor_mean_poses_world.npy", mean_poses_np.astype(np.float64))
    saved_visualizations = render_visualizations(summary, output_dir)

    print("Saved outputs:")
    print(output_dir / "chunk_anchor_variance_summary.json")
    print(output_dir / "gt_poses_world.npy")
    print(output_dir / "anchor_trajectories_world.npy")
    print(output_dir / "anchor_mean_poses_world.npy")
    for path in saved_visualizations:
        print(path)


if __name__ == "__main__":
    main()
