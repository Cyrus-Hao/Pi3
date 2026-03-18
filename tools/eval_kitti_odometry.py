from __future__ import annotations

"""
Evaluate Pi3X on KITTI Odometry sequences and save bird-eye-view plots.

References:
- KITTI Odometry benchmark: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- Pi3X pose format: `pi3/models/pi3x.py`, where `poses` are camera-to-world
  matrices in OpenCV convention (Right-Down-Forward).

This script preserves the codebase's existing pose interpretation and does not
flip axes. For bird-eye visualization it plots the translation X and Z directly.
"""

import argparse
import json
import math
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import huggingface_hub  # noqa: F401
except ModuleNotFoundError:
    hub_module = types.ModuleType("huggingface_hub")

    class PyTorchModelHubMixin:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ModuleNotFoundError(
                "`huggingface_hub` is required for from_pretrained(). "
                "Install it or place a local Pi3X checkpoint at ckpt/pi3x.safetensors."
            )

    hub_module.PyTorchModelHubMixin = PyTorchModelHubMixin
    sys.modules["huggingface_hub"] = hub_module

from pi3.models.pi3x import Pi3X


VALID_SEQUENCES = {f"{idx:02d}" for idx in range(11)}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
PIXEL_LIMIT = 255000
LOCAL_PI3X_CKPT = REPO_ROOT / "ckpt" / "pi3x.safetensors"
RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pi3X on KITTI Odometry.")
    parser.add_argument(
        "--kitti_root",
        type=str,
        required=True,
        help="Path to the KITTI odometry dataset root.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        required=True,
        help="Sequence IDs to evaluate. Only 00-10 are supported.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        nargs="+",
        required=True,
        help="Use only the first N frames from each sequence. Accepts one or more values.",
    )
    parser.add_argument(
        "--use_gt_pose_prior",
        action="store_true",
        help="Use ground-truth poses as Pi3X pose priors.",
    )
    args = parser.parse_args()

    invalid_sequences = [seq for seq in args.sequences if seq not in VALID_SEQUENCES]
    if invalid_sequences:
        parser.error(f"Invalid sequence IDs: {invalid_sequences}. Only 00-10 are supported.")
    invalid_num_samples = [num_samples for num_samples in args.num_samples if num_samples <= 0]
    if invalid_num_samples:
        parser.error(f"--num_samples must contain only positive integers, got: {invalid_num_samples}")

    return args


def resolve_kitti_root(kitti_root: str) -> Path:
    candidate = Path(kitti_root).expanduser().resolve()
    if candidate.exists():
        return candidate

    fallback = REPO_ROOT / "dataset" / "kitti-od"
    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError(
        f"KITTI root does not exist: {candidate}\n"
        f"Fallback root also not found: {fallback}"
    )


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_sequence_assets(kitti_root: Path, sequence: str) -> tuple[Path, Path, Path, str]:
    sequence_dirs = [
        kitti_root / "sequences" / sequence,
        kitti_root / sequence,
        kitti_root / "dataset" / "sequences" / sequence,
        kitti_root / "data_odometry_gray" / "sequences" / sequence,
        kitti_root / "data_odometry_gray" / "dataset" / "sequences" / sequence,
        kitti_root / "data_odometry_color" / "sequences" / sequence,
        kitti_root / "data_odometry_color" / "dataset" / "sequences" / sequence,
    ]
    existing_sequence_dirs = [path for path in sequence_dirs if path.exists()]
    if not existing_sequence_dirs:
        raise FileNotFoundError(f"Could not find KITTI sequence directory for sequence {sequence}.")

    image_dir = None
    camera_key = None
    for sequence_dir in existing_sequence_dirs:
        for image_name, proj_name in (("image_0", "P0"), ("image_2", "P2"), ("image_1", "P1"), ("image_3", "P3")):
            candidate = sequence_dir / image_name
            if candidate.is_dir():
                image_dir = candidate
                camera_key = proj_name
                break
        if image_dir is not None:
            break

    if image_dir is None or camera_key is None:
        raise FileNotFoundError(f"Could not find image directory for sequence {sequence}.")

    pose_file = first_existing_path(
        [
            kitti_root / "poses" / f"{sequence}.txt",
            kitti_root / "data_odometry_poses" / "poses" / f"{sequence}.txt",
            kitti_root / "data_odometry_poses" / "dataset" / "poses" / f"{sequence}.txt",
            image_dir.parent / "poses.txt",
            image_dir.parent / "pose.txt",
        ]
    )
    if pose_file is None:
        raise FileNotFoundError(f"Could not find pose file for sequence {sequence}.")

    calib_file = first_existing_path(
        [
            image_dir.parent / "calib.txt",
            kitti_root / "sequences" / sequence / "calib.txt",
            kitti_root / sequence / "calib.txt",
            kitti_root / "dataset" / "sequences" / sequence / "calib.txt",
            kitti_root / "data_odometry_calib" / "sequences" / sequence / "calib.txt",
            kitti_root / "data_odometry_calib" / "dataset" / "sequences" / sequence / "calib.txt",
        ]
    )
    if calib_file is None:
        raise FileNotFoundError(f"Could not find calibration file for sequence {sequence}.")

    return image_dir, pose_file, calib_file, camera_key


def list_frame_paths(image_dir: Path) -> list[Path]:
    frame_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not frame_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}.")
    return frame_paths


def load_pose_file(pose_file: Path) -> np.ndarray:
    poses = []
    with pose_file.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = np.fromstring(stripped, sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(f"Pose file {pose_file} has an invalid line {line_idx}: expected 12 values.")
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :] = values.reshape(3, 4)
            poses.append(pose)

    if not poses:
        raise ValueError(f"Pose file is empty: {pose_file}")

    return np.stack(poses, axis=0)


def load_calibration_matrix(calib_file: Path, camera_key: str) -> np.ndarray:
    projection_row = None
    with calib_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            name, values = line.split(":", maxsplit=1)
            if name.strip() == camera_key:
                projection_row = np.fromstring(values.strip(), sep=" ", dtype=np.float64)
                break

    if projection_row is None:
        raise KeyError(f"Camera key {camera_key} not found in calibration file {calib_file}.")
    if projection_row.size != 12:
        raise ValueError(f"Calibration entry {camera_key} in {calib_file} must contain 12 values.")

    projection = projection_row.reshape(3, 4)
    intrinsic = projection[:, :3].copy()
    intrinsic[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return intrinsic


def compute_target_size(width: int, height: int, pixel_limit: int = PIXEL_LIMIT) -> tuple[int, int]:
    scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1.0
    width_target = width * scale
    height_target = height * scale

    k = round(width_target / 14)
    m = round(height_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > width_target / height_target:
            k -= 1
        else:
            m -= 1

    target_w = max(1, k) * 14
    target_h = max(1, m) * 14
    return target_w, target_h


def load_sequence_tensors(
    frame_paths: list[Path],
    poses_np: np.ndarray,
    intrinsic_np: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    first_image = Image.open(frame_paths[0]).convert("RGB")
    width_orig, height_orig = first_image.size
    target_w, target_h = compute_target_size(width_orig, height_orig)
    scale_x = target_w / width_orig
    scale_y = target_h / height_orig

    to_tensor = transforms.ToTensor()
    image_tensors = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            image_rgb = image.convert("RGB")
            resized = image_rgb.resize((target_w, target_h), RESAMPLE_LANCZOS)
            image_tensors.append(to_tensor(resized))

    imgs = torch.stack(image_tensors, dim=0)[None].to(device=device, dtype=torch.float32)

    intrinsics_np = np.repeat(intrinsic_np[None, :, :], len(frame_paths), axis=0)
    intrinsics_np = intrinsics_np.copy()
    intrinsics_np[:, 0, 0] *= scale_x
    intrinsics_np[:, 0, 2] *= scale_x
    intrinsics_np[:, 1, 1] *= scale_y
    intrinsics_np[:, 1, 2] *= scale_y

    poses = torch.from_numpy(poses_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    intrinsics = torch.from_numpy(intrinsics_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    metadata = {
        "width_orig": int(width_orig),
        "height_orig": int(height_orig),
        "target_width": int(target_w),
        "target_height": int(target_h),
    }
    return imgs, poses, intrinsics, metadata


def load_model(device: torch.device) -> Pi3X:
    if LOCAL_PI3X_CKPT.exists():
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Local Pi3X checkpoint found, but `safetensors` is not installed. "
                "Install it or remove the local checkpoint so the script can fall back to Hugging Face."
            ) from exc
        model = Pi3X(use_multimodal=True).eval()
        state_dict = load_file(str(LOCAL_PI3X_CKPT))
        model.load_state_dict(state_dict, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
    return model.to(device)


def configure_model_for_kitti_odometry(model: Pi3X) -> Pi3X:
    # Keep Pi3X's original prior scale augmentation behavior at inference time.
    model.disable_prior_scale_aug_for_inference = False
    return model


def get_autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    major_cc, _ = torch.cuda.get_device_capability(device=device)
    return torch.bfloat16 if major_cc >= 8 else torch.float16


def infer_sequence(
    model: Pi3X,
    imgs: torch.Tensor,
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    use_gt_pose_prior: bool,
) -> dict[str, torch.Tensor]:
    _, num_frames, _, _, _ = imgs.shape
    device = imgs.device

    model_kwargs: dict[str, torch.Tensor | bool] = {
        "imgs": imgs,
        "intrinsics": intrinsics,
        "with_prior": True,
        "mask_add_ray": torch.ones((1, num_frames), dtype=torch.bool, device=device),
        "pose_only": True,
    }
    if use_gt_pose_prior:
        model_kwargs["poses"] = poses
        model_kwargs["mask_add_pose"] = torch.ones((1, num_frames), dtype=torch.bool, device=device)

    autocast_dtype = get_autocast_dtype(device)
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    with torch.inference_mode():
        with tqdm(total=1, desc="Inference", dynamic_ncols=True) as progress_bar:
            with autocast_ctx:
                outputs = model(**model_kwargs)
            progress_bar.update(1)

    return outputs


def align_predicted_poses_to_gt(pred_poses: np.ndarray, gt_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alignment = gt_poses[0] @ np.linalg.inv(pred_poses[0])
    aligned_pred = np.einsum("ij,njk->nik", alignment, pred_poses)
    mapped_pred_origin = alignment[:3, 3].copy()
    return aligned_pred, mapped_pred_origin


def umeyama_alignment(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    if src_xyz.shape != dst_xyz.shape:
        raise ValueError(f"Shape mismatch for Umeyama alignment: {src_xyz.shape} vs {dst_xyz.shape}")
    if src_xyz.ndim != 2 or src_xyz.shape[1] != 3:
        raise ValueError(f"Umeyama alignment expects shape (N, 3), got {src_xyz.shape}")
    if src_xyz.shape[0] < 2:
        raise ValueError("Umeyama alignment needs at least 2 points.")

    src_mean = src_xyz.mean(axis=0)
    dst_mean = dst_xyz.mean(axis=0)
    src_centered = src_xyz - src_mean
    dst_centered = dst_xyz - dst_mean

    covariance = (dst_centered.T @ src_centered) / src_xyz.shape[0]
    u, singular_values, vh = np.linalg.svd(covariance)

    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vh

    if with_scale:
        src_var = np.mean(np.sum(src_centered ** 2, axis=1))
        if src_var <= 0:
            raise ValueError("Degenerate source trajectory for Umeyama alignment.")
        scale = np.sum(singular_values * np.diag(correction)) / src_var
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    return float(scale), rotation, translation


def align_predicted_poses_umeyama(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_xyz = pred_poses[:, :3, 3]
    gt_xyz = gt_poses[:, :3, 3]
    scale, rotation, translation = umeyama_alignment(pred_xyz, gt_xyz, with_scale=True)

    aligned_pred = pred_poses.copy()
    aligned_pred[:, :3, :3] = np.einsum("ij,njk->nik", rotation, pred_poses[:, :3, :3])
    aligned_pred[:, :3, 3] = scale * np.einsum("ij,nj->ni", rotation, pred_xyz) + translation
    mapped_pred_origin = translation.copy()
    return aligned_pred, mapped_pred_origin


def build_visualization_stem(
    sequence: str,
    num_frames: int,
    use_gt_pose_prior: bool,
    alignment_name: str,
) -> str:
    pose_tag = "withpose" if use_gt_pose_prior else "withoutpose"
    return f"sequence_{sequence}_{num_frames}_{pose_tag}_contiguous_{alignment_name}"


def build_run_stem(num_samples: int, use_gt_pose_prior: bool) -> str:
    pose_tag = "withpose" if use_gt_pose_prior else "withoutpose"
    return f"kitti_odometry_{num_samples}_{pose_tag}_contiguous"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def pose_inverse_batch(poses: np.ndarray) -> np.ndarray:
    inv_poses = np.repeat(np.eye(4, dtype=np.float64)[None], poses.shape[0], axis=0)
    rotation = poses[:, :3, :3]
    translation = poses[:, :3, 3]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    inv_poses[:, :3, :3] = rotation_t
    inv_poses[:, :3, 3] = -np.einsum("nij,nj->ni", rotation_t, translation)
    return inv_poses


def rotation_angle_deg(rotation_delta: np.ndarray) -> np.ndarray:
    traces = np.trace(rotation_delta, axis1=1, axis2=2)
    cos_theta = np.clip((traces - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def summarize_error(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "rmse": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compute_pose_metrics(pred_poses: np.ndarray, gt_poses: np.ndarray) -> dict[str, object]:
    pred_translation = pred_poses[:, :3, 3]
    gt_translation = gt_poses[:, :3, 3]
    ape_translation = np.linalg.norm(pred_translation - gt_translation, axis=1)

    pred_rotation = pred_poses[:, :3, :3]
    gt_rotation = gt_poses[:, :3, :3]
    ape_rotation_delta = np.einsum("nij,njk->nik", np.transpose(gt_rotation, (0, 2, 1)), pred_rotation)
    ape_rotation = rotation_angle_deg(ape_rotation_delta)

    if pred_poses.shape[0] >= 2:
        pred_rel = np.einsum("nij,njk->nik", pose_inverse_batch(pred_poses[:-1]), pred_poses[1:])
        gt_rel = np.einsum("nij,njk->nik", pose_inverse_batch(gt_poses[:-1]), gt_poses[1:])
        rpe_translation = np.linalg.norm(pred_rel[:, :3, 3] - gt_rel[:, :3, 3], axis=1)
        rpe_rotation_delta = np.einsum(
            "nij,njk->nik",
            np.transpose(gt_rel[:, :3, :3], (0, 2, 1)),
            pred_rel[:, :3, :3],
        )
        rpe_rotation = rotation_angle_deg(rpe_rotation_delta)
    else:
        rpe_translation = np.empty((0,), dtype=np.float64)
        rpe_rotation = np.empty((0,), dtype=np.float64)

    return {
        "ape_translation_m": summarize_error(ape_translation),
        "ape_rotation_deg": summarize_error(ape_rotation),
        "rpe_translation_m": summarize_error(rpe_translation),
        "rpe_rotation_deg": summarize_error(rpe_rotation),
        "per_frame": {
            "ape_translation_m": ape_translation.tolist(),
            "ape_rotation_deg": ape_rotation.tolist(),
        },
        "per_step": {
            "rpe_translation_m": rpe_translation.tolist(),
            "rpe_rotation_deg": rpe_rotation.tolist(),
        },
    }


def save_bird_eye_plot(
    sequence: str,
    gt_poses: np.ndarray,
    pred_poses_aligned: np.ndarray,
    mapped_pred_origin: np.ndarray,
    output_dir: Path,
    file_stem: str,
    title_suffix: str = "Start-Aligned",
    pred_label: str = "Pi3 Predicted",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{file_stem}.png"

    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz = pred_poses_aligned[:, :3, 3]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], color="green", linewidth=2.5, label="Ground Truth")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 2], color="red", linewidth=2.5, label=pred_label)
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 2], color="black", s=60, marker="o", label="Start", zorder=5)
    ax.scatter(0.0, 0.0, color="blue", s=140, marker="+", linewidths=2.0, label="World Origin", zorder=6)
    ax.scatter(
        mapped_pred_origin[0],
        mapped_pred_origin[2],
        color="purple",
        s=100,
        marker="x",
        linewidths=2.0,
        label="Mapped Pred Origin",
        zorder=6,
    )
    ax.annotate(
        "Mapped Pred Origin",
        xy=(mapped_pred_origin[0], mapped_pred_origin[2]),
        xytext=(8, 8),
        textcoords="offset points",
        color="purple",
        fontsize=11,
    )

    ax.set_title(f"Sequence {sequence} — Bird-Eye View ({title_suffix})", fontsize=16)
    ax.set_xlabel("X (m)", fontsize=13)
    ax.set_ylabel("Z (m)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, color="#bfbfbf", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=11, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    kitti_root = resolve_kitti_root(args.kitti_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = get_autocast_dtype(device)

    print(f"Using KITTI root: {kitti_root}")
    print(f"Using device: {device}")
    print("Loading Pi3X model...")
    model = configure_model_for_kitti_odometry(load_model(device))
    total_runs = len(args.num_samples)

    for run_idx, num_samples in enumerate(args.num_samples, start=1):
        run_stem = build_run_stem(num_samples, args.use_gt_pose_prior)
        output_root = Path("results") / run_stem
        print(f"Starting run {run_idx}/{total_runs}: num_samples={num_samples}")
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(
            output_root / "run_config.json",
            {
                "run_name": run_stem,
                "kitti_root": str(kitti_root),
                "sequences": args.sequences,
                "num_samples_requested": int(num_samples),
                "num_samples_requested_all": [int(value) for value in args.num_samples],
                "frame_selection": "first_n_contiguous",
                "use_gt_pose_prior": bool(args.use_gt_pose_prior),
                "pose_only": True,
                "disable_prior_scale_aug_for_inference": bool(model.disable_prior_scale_aug_for_inference),
                "pixel_limit": int(PIXEL_LIMIT),
                "device": str(device),
                "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
                "command": " ".join(sys.argv),
            },
        )
        sequence_summaries: list[dict[str, object]] = []

        for sequence in args.sequences:
            print(f"Evaluating sequence {sequence}...")
            image_dir, pose_file, calib_file, camera_key = find_sequence_assets(kitti_root, sequence)
            frame_paths_all = list_frame_paths(image_dir)
            gt_poses_all = load_pose_file(pose_file)
            intrinsic_np = load_calibration_matrix(calib_file, camera_key)

            num_frames = min(num_samples, len(frame_paths_all), gt_poses_all.shape[0])
            if num_frames <= 0:
                raise RuntimeError(f"Sequence {sequence} has no usable frames.")

            frame_paths = frame_paths_all[:num_frames]
            gt_poses_np = gt_poses_all[:num_frames]
            imgs, gt_poses_t, intrinsics_t, tensor_metadata = load_sequence_tensors(frame_paths, gt_poses_np, intrinsic_np, device)
            sequence_dir = output_root / f"sequence_{sequence}"
            visualization_dir = sequence_dir / "visualizations"
            visualization_dir.mkdir(parents=True, exist_ok=True)

            outputs = infer_sequence(
                model=model,
                imgs=imgs,
                poses=gt_poses_t,
                intrinsics=intrinsics_t,
                use_gt_pose_prior=args.use_gt_pose_prior,
            )

            pred_poses_np = outputs["camera_poses"][0].detach().float().cpu().numpy().astype(np.float64)
            pred_poses_aligned, mapped_pred_origin = align_predicted_poses_to_gt(pred_poses_np, gt_poses_np)
            origin_metrics = compute_pose_metrics(pred_poses_aligned, gt_poses_np)
            save_bird_eye_plot(
                sequence=sequence,
                gt_poses=gt_poses_np,
                pred_poses_aligned=pred_poses_aligned,
                mapped_pred_origin=mapped_pred_origin,
                output_dir=visualization_dir,
                file_stem=build_visualization_stem(
                    sequence=sequence,
                    num_frames=num_frames,
                    use_gt_pose_prior=args.use_gt_pose_prior,
                    alignment_name="origin",
                ),
                title_suffix="Start-Aligned",
                pred_label="Pi3 Predicted (Start-Aligned)",
            )

            pred_poses_umeyama, mapped_pred_origin_umeyama = align_predicted_poses_umeyama(pred_poses_np, gt_poses_np)
            umeyama_metrics = compute_pose_metrics(pred_poses_umeyama, gt_poses_np)
            save_bird_eye_plot(
                sequence=sequence,
                gt_poses=gt_poses_np,
                pred_poses_aligned=pred_poses_umeyama,
                mapped_pred_origin=mapped_pred_origin_umeyama,
                output_dir=visualization_dir,
                file_stem=build_visualization_stem(
                    sequence=sequence,
                    num_frames=num_frames,
                    use_gt_pose_prior=args.use_gt_pose_prior,
                    alignment_name="umeyama",
                ),
                title_suffix="Umeyama-Aligned",
                pred_label="Pi3 Predicted (Umeyama)",
            )

            np.save(sequence_dir / "pred_poses_raw.npy", pred_poses_np)
            np.save(sequence_dir / "pred_poses_origin.npy", pred_poses_aligned)
            np.save(sequence_dir / "pred_poses_umeyama.npy", pred_poses_umeyama)
            np.save(sequence_dir / "gt_poses.npy", gt_poses_np)

            write_json(
                sequence_dir / "sequence_config.json",
                {
                    "sequence": sequence,
                    "num_frames_used": int(num_frames),
                    "num_frames_available": int(len(frame_paths_all)),
                    "num_gt_poses_available": int(gt_poses_all.shape[0]),
                    "frame_selection": "first_n_contiguous",
                    "use_gt_pose_prior": bool(args.use_gt_pose_prior),
                    "camera_key": camera_key,
                    "image_dir": str(image_dir),
                    "pose_file": str(pose_file),
                    "calib_file": str(calib_file),
                    "input_image_size": {
                        "width": tensor_metadata["width_orig"],
                        "height": tensor_metadata["height_orig"],
                    },
                    "model_image_size": {
                        "width": tensor_metadata["target_width"],
                        "height": tensor_metadata["target_height"],
                    },
                    "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
                    "pose_only": True,
                    "disable_prior_scale_aug_for_inference": bool(model.disable_prior_scale_aug_for_inference),
                    "visualizations": [
                        f"visualizations/{build_visualization_stem(sequence, num_frames, args.use_gt_pose_prior, 'origin')}.png",
                        f"visualizations/{build_visualization_stem(sequence, num_frames, args.use_gt_pose_prior, 'umeyama')}.png",
                    ],
                    "frame_names": [path.name for path in frame_paths],
                },
            )
            write_json(sequence_dir / "metrics_origin.json", origin_metrics)
            write_json(sequence_dir / "metrics_umeyama.json", umeyama_metrics)

            sequence_summaries.append(
                {
                    "sequence": sequence,
                    "num_frames_used": int(num_frames),
                    "origin": {
                        "ape_translation_rmse_m": origin_metrics["ape_translation_m"]["rmse"],
                        "ape_rotation_rmse_deg": origin_metrics["ape_rotation_deg"]["rmse"],
                        "rpe_translation_rmse_m": origin_metrics["rpe_translation_m"]["rmse"],
                        "rpe_rotation_rmse_deg": origin_metrics["rpe_rotation_deg"]["rmse"],
                    },
                    "umeyama": {
                        "ape_translation_rmse_m": umeyama_metrics["ape_translation_m"]["rmse"],
                        "ape_rotation_rmse_deg": umeyama_metrics["ape_rotation_deg"]["rmse"],
                        "rpe_translation_rmse_m": umeyama_metrics["rpe_translation_m"]["rmse"],
                        "rpe_rotation_rmse_deg": umeyama_metrics["rpe_rotation_deg"]["rmse"],
                    },
                }
            )

            print(f"Sequence {sequence} done, {num_frames} frames processed")

            del imgs, gt_poses_t, intrinsics_t, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

        write_json(output_root / "summary.json", {"run_name": run_stem, "sequences": sequence_summaries})
        print(f"Completed run {run_idx}/{total_runs}: num_samples={num_samples}")


if __name__ == "__main__":
    main()
