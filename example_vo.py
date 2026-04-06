import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pi3.models.pi3x import Pi3X
from pi3.pipe.pi3x_vo import Pi3XVO
from pi3.utils.basic import write_ply
from pi3.utils.geometry import get_pixel
from tools.eval_kitti_odometry import (
    align_predicted_poses_to_gt,
    compute_pose_metrics,
    find_sequence_assets,
    list_frame_paths,
    load_calibration_matrix,
    load_pose_file,
    load_sequence_tensors,
    resolve_kitti_root,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Pi3X VO on KITTI with GT pose / GT ray priors and save chunk metrics."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/Pi3/dataset/kitti-od",
        help="Path to KITTI odometry root.",
    )
    parser.add_argument("--seq", type=str, default="00", help="KITTI sequence id.")
    parser.add_argument("--num_frames", type=int, default=350, help="Use the first N contiguous frames.")
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size for inference.")
    parser.add_argument("--overlap", type=int, default=20, help="Overlap length between chunks.")
    parser.add_argument("--conf_thre", type=float, default=0.05, help="Confidence threshold for point export.")
    parser.add_argument(
        "--inject_condition",
        type=str,
        nargs="+",
        default=["pose", "ray"],
        help="Fallback overlap prior types when full GT priors are disabled.",
    )
    parser.add_argument("--disable_gt_pose_prior", action="store_true", help="Do not feed GT poses to each chunk.")
    parser.add_argument("--disable_gt_ray_prior", action="store_true", help="Do not feed GT rays to each chunk.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the model checkpoint file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_outputs/example_vo_kitti",
        help="Directory to save PLY, metrics and plots.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional point cloud save path. Defaults to output_dir/vo_seqXX_nYYY.ply",
    )
    return parser.parse_args()


def get_runtime_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("Requested `cuda` but CUDA is unavailable, falling back to `cpu`.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(device: torch.device, ckpt: str | None) -> Pi3X:
    print("Loading model...")
    if ckpt is not None:
        model = Pi3X().to(device).eval()
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    return model


def build_gt_rays(intrinsics: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, num_frames = intrinsics.shape[:2]
    device = intrinsics.device
    dtype = intrinsics.dtype
    pixels = torch.from_numpy(get_pixel(height, width).T.reshape(height, width, 3)).to(device=device, dtype=dtype)
    pixels = pixels.view(1, 1, height, width, 3).expand(batch_size, num_frames, height, width, 3)
    inv_intrinsics = torch.linalg.inv(intrinsics)
    return torch.einsum("bnij,bnhwj->bnhwi", inv_intrinsics, pixels)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_chunk_metric_plot(chunk_records: list[dict], save_path: Path, seq: str) -> None:
    chunk_indices = np.asarray([record["chunk_idx"] for record in chunk_records], dtype=np.int64)
    chunk_end_frames = np.asarray([record["end_frame_1based"] for record in chunk_records], dtype=np.int64)
    metric_values = np.asarray([record["absolute_metric"] for record in chunk_records], dtype=np.float64)
    cumulative_mean = np.cumsum(metric_values) / np.arange(1, metric_values.shape[0] + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=160, sharex=True)

    axes[0].plot(chunk_indices, metric_values, marker="o", linewidth=2.0, color="#d62728")
    axes[0].set_ylabel("Absolute Metric", fontsize=11)
    axes[0].set_title(f"Chunk Absolute Metric (KITTI seq {seq})", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(chunk_indices, cumulative_mean, marker="o", linewidth=2.0, color="#1f77b4")
    axes[1].set_xlabel("Chunk Index", fontsize=11)
    axes[1].set_ylabel("Cumulative Mean", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    if chunk_indices.size > 0:
        tick_step = max(1, int(np.ceil(chunk_indices.size / 12)))
        tick_positions = chunk_indices[::tick_step]
        tick_labels = [f"{int(idx)}\n(end={int(frame)})" for idx, frame in zip(tick_positions, chunk_end_frames[::tick_step])]
        axes[1].set_xticks(tick_positions)
        axes[1].set_xticklabels(tick_labels)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = get_runtime_device(args.device)
    seq = str(args.seq).zfill(2)

    output_dir = Path(args.output_dir) / f"seq{seq}_n{args.num_frames}_chunk{args.chunk_size}_overlap{args.overlap}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_path) if args.save_path is not None else output_dir / f"vo_seq{seq}_n{args.num_frames}.ply"

    data_root = resolve_kitti_root(args.data_root)
    image_dir, pose_file, calib_file, camera_key = find_sequence_assets(data_root, seq)
    frame_paths_all = list_frame_paths(image_dir)
    gt_poses_all = load_pose_file(pose_file)
    intrinsic_np = load_calibration_matrix(calib_file, camera_key)

    total_frames = min(len(frame_paths_all), gt_poses_all.shape[0])
    num_frames = min(args.num_frames, total_frames)
    frame_paths = frame_paths_all[:num_frames]
    gt_poses_np = gt_poses_all[:num_frames]

    print(f"Loading KITTI sequence {seq} with {num_frames} contiguous frames...")
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
    pipe = Pi3XVO(model)

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = None

    print("Running chunked VO inference...")
    with torch.no_grad():
        res = pipe(
            imgs=imgs,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            conf_thre=args.conf_thre,
            inject_condition=args.inject_condition,
            dtype=dtype,
            poses=None if args.disable_gt_pose_prior else gt_poses_t,
            rays=None if args.disable_gt_ray_prior else gt_rays_t,
            intrinsics=intrinsics_t if args.disable_gt_ray_prior else None,
        )

    masks = res["conf"][0] > args.conf_thre
    print(f"Saving point cloud to: {save_path}")
    if save_path.parent:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(res["points"][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], str(save_path))

    pred_poses_np = res["camera_poses"][0].detach().float().cpu().numpy().astype(np.float64)
    gt_poses_eval_np = gt_poses_t[0].detach().float().cpu().numpy().astype(np.float64)
    pred_poses_origin_np, _ = align_predicted_poses_to_gt(pred_poses_np, gt_poses_eval_np)
    pose_metrics_raw = compute_pose_metrics(pred_poses_np, gt_poses_eval_np)
    pose_metrics_origin = compute_pose_metrics(pred_poses_origin_np, gt_poses_eval_np)

    diagnostics = res.get("diagnostics", {})
    chunk_ranges = diagnostics.get("chunk_ranges_0based_half_open", [])
    chunk_metric_values = res["chunk_metrics"][0].detach().float().cpu().numpy().astype(np.float64).tolist()
    chunk_ranges = chunk_ranges[: len(chunk_metric_values)]

    chunk_records = []
    for chunk_idx, (chunk_range, metric_value) in enumerate(zip(chunk_ranges, chunk_metric_values)):
        start_idx, end_idx = int(chunk_range[0]), int(chunk_range[1])
        chunk_records.append(
            {
                "chunk_idx": chunk_idx,
                "start_frame_0based": start_idx,
                "end_frame_0based_exclusive": end_idx,
                "start_frame_1based": start_idx + 1,
                "end_frame_1based": end_idx,
                "num_frames": end_idx - start_idx,
                "absolute_metric": float(metric_value),
            }
        )

    chunk_metric_plot_path = output_dir / "chunk_absolute_metric.png"
    chunk_metric_json_path = output_dir / "chunk_absolute_metric.json"
    chunk_metric_npy_path = output_dir / "chunk_absolute_metric.npy"
    save_chunk_metric_plot(chunk_records, chunk_metric_plot_path, seq=seq)
    np.save(chunk_metric_npy_path, np.asarray(chunk_metric_values, dtype=np.float64))

    summary = {
        "config": {
            "data_root": str(data_root),
            "seq": seq,
            "num_frames": int(num_frames),
            "chunk_size": int(args.chunk_size),
            "overlap": int(args.overlap),
            "conf_thre": float(args.conf_thre),
            "inject_condition": list(args.inject_condition),
            "use_gt_pose_prior": not args.disable_gt_pose_prior,
            "use_gt_ray_prior": not args.disable_gt_ray_prior,
            "camera_key": camera_key,
            "model_image_size": {
                "width": int(tensor_metadata["target_width"]),
                "height": int(tensor_metadata["target_height"]),
            },
            "point_cloud_path": str(save_path),
        },
        "chunk_metrics": {
            "num_chunks": len(chunk_records),
            "values": chunk_metric_values,
            "mean": float(np.mean(chunk_metric_values)) if chunk_metric_values else None,
            "std": float(np.std(chunk_metric_values)) if chunk_metric_values else None,
            "min": float(np.min(chunk_metric_values)) if chunk_metric_values else None,
            "max": float(np.max(chunk_metric_values)) if chunk_metric_values else None,
            "per_chunk": chunk_records,
            "plot_path": str(chunk_metric_plot_path),
            "npy_path": str(chunk_metric_npy_path),
        },
        "pose_metrics": {
            "raw": pose_metrics_raw,
            "origin_aligned": pose_metrics_origin,
        },
        "chunk_diagnostics": diagnostics,
    }
    save_json(chunk_metric_json_path, summary)

    np.save(output_dir / "pred_poses_raw.npy", pred_poses_np)
    np.save(output_dir / "pred_poses_origin_aligned.npy", pred_poses_origin_np)
    np.save(output_dir / "gt_poses.npy", gt_poses_eval_np)

    print("Saved outputs:")
    print(f"  point cloud: {save_path}")
    print(f"  chunk metric json: {chunk_metric_json_path}")
    print(f"  chunk metric npy: {chunk_metric_npy_path}")
    print(f"  chunk metric plot: {chunk_metric_plot_path}")
    print(f"  pred poses: {output_dir / 'pred_poses_raw.npy'}")
    print(f"  gt poses: {output_dir / 'gt_poses.npy'}")
    print("Done.")


if __name__ == '__main__':
    main()
