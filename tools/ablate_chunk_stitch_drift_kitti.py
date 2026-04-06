from __future__ import annotations

import argparse
import json
import math
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
PIXEL_LIMIT = 255000
RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate long-sequence chunk drift for Pi3X on KITTI. "
            "The script runs the canonical chunk inference once, then re-stitches "
            "the saved raw chunk poses with multiple boundary alignment policies."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/Pi3/dataset/kitti-od",
        help="Path to KITTI odometry root.",
    )
    parser.add_argument("--seq", type=str, default="00", help="KITTI sequence id.")
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
        "--output_dir",
        type=str,
        default="debug_outputs/chunk_stitch_ablation",
        help="Directory to save json/npy outputs.",
    )
    parser.add_argument(
        "--visualize_only_summary",
        type=str,
        default=None,
        help="Only render visualization figures from an existing chunk_stitch_ablation_summary.json.",
    )
    return parser.parse_args()


def get_runtime_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("Requested `cuda` but CUDA is unavailable, falling back to `cpu`.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(device: torch.device, ckpt: str | None):
    try:
        import huggingface_hub  # noqa: F401
    except ModuleNotFoundError:
        hub_module = types.ModuleType("huggingface_hub")

        class PyTorchModelHubMixin:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise ModuleNotFoundError(
                    "`huggingface_hub` is required for from_pretrained(). "
                    "Install it or pass a local checkpoint with `--ckpt`."
                )

        hub_module.PyTorchModelHubMixin = PyTorchModelHubMixin
        sys.modules["huggingface_hub"] = hub_module

    from pi3.models.pi3x import Pi3X

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


def get_autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    major_cc, _ = torch.cuda.get_device_capability(device=device)
    return torch.bfloat16 if major_cc >= 8 else torch.float16


def build_gt_rays(intrinsics: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, num_frames = intrinsics.shape[:2]
    device = intrinsics.device
    dtype = intrinsics.dtype
    pixels = torch.from_numpy(get_pixel(height, width).T.reshape(height, width, 3)).to(device=device, dtype=dtype)
    pixels = pixels.view(1, 1, height, width, 3).expand(batch_size, num_frames, height, width, 3)
    inv_intrinsics = torch.linalg.inv(intrinsics)
    return torch.einsum("bnij,bnhwj->bnhwi", inv_intrinsics, pixels)


def get_pixel(height: int, width: int) -> np.ndarray:
    u_a, v_a = np.meshgrid(np.arange(width), np.arange(height))
    pixels_a = np.stack(
        [
            u_a.flatten() + 0.5,
            v_a.flatten() + 0.5,
            np.ones_like(u_a.flatten()),
        ],
        axis=0,
    )
    return pixels_a


def depth_edge(
    depth: torch.Tensor,
    atol: float | None = None,
    rtol: float | None = None,
    kernel_size: int = 3,
    mask: torch.Tensor | None = None,
) -> torch.BoolTensor:
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (
            F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2)
            + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
        )
    else:
        diff = (
            F.max_pool2d(
                torch.where(mask, depth, -torch.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
            + F.max_pool2d(
                torch.where(mask, -depth, -torch.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        )

    if atol is not None and rtol is not None:
        edge = diff > (atol + rtol * depth)
    elif atol is not None:
        edge = diff > atol
    elif rtol is not None:
        edge = diff > (rtol * depth)
    else:
        raise ValueError("Either `atol` or `rtol` must be provided.")

    return edge.reshape(*shape)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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


def to_first_frame_relative(poses_np: np.ndarray) -> np.ndarray:
    first_inv = np.linalg.inv(poses_np[0])
    return np.einsum("ij,njk->nik", first_inv, poses_np)


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix)
    rot = u @ vh
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vh
    return rot


def normalize_pose_rotations(poses_np: np.ndarray) -> np.ndarray:
    normalized = poses_np.copy()
    for idx in range(normalized.shape[0]):
        normalized[idx, :3, :3] = project_to_rotation(normalized[idx, :3, :3])
        normalized[idx, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=normalized.dtype)
    return normalized


def compute_point_mask(local_depth: torch.Tensor, conf_logit: torch.Tensor, conf_thre: float) -> torch.Tensor:
    conf = torch.sigmoid(conf_logit)[..., 0]
    edge = depth_edge(local_depth, rtol=0.03)
    conf = conf.clone()
    conf[edge] = 0
    mask = conf > conf_thre

    if int(mask.sum().item()) < 10:
        flat_conf = conf.view(conf.shape[0], conf.shape[1], -1)
        k = max(1, int(flat_conf.shape[-1] * 0.1))
        topk_vals, _ = torch.topk(flat_conf, k, dim=-1)
        min_vals = topk_vals[..., -1].unsqueeze(-1).unsqueeze(-1)
        mask = conf >= min_vals

    return mask


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
        return torch.eye(4, device=device, dtype=src_points.dtype).repeat(batch_size, 1, 1)

    src_mean = (src * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(batch_size, 1, 1) + eps)
    tgt_mean = (tgt * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(batch_size, 1, 1) + eps)
    src_centered = (src - src_mean) * mask
    tgt_centered = (tgt - tgt_mean) * mask

    cov = torch.bmm(src_centered.transpose(1, 2), tgt_centered)
    u, s, v = torch.svd(cov)
    r = torch.bmm(v, u.transpose(1, 2))

    det = torch.det(r)
    diag = torch.ones(batch_size, 3, device=device, dtype=src_points.dtype)
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
    sim3 = torch.eye(4, device=device, dtype=src_points.dtype).repeat(batch_size, 1, 1)
    sim3[:, :3, :3] = scale * r
    sim3[:, :3, 3] = t.squeeze(2)

    if bad_mask.any():
        sim3[bad_mask] = torch.eye(4, device=device, dtype=src_points.dtype).repeat(batch_size, 1, 1)[bad_mask]
    return sim3


def compute_center_alignment_transform(
    src_centers: torch.Tensor,
    tgt_centers: torch.Tensor,
    with_scale: bool,
) -> torch.Tensor:
    if src_centers.shape != tgt_centers.shape:
        raise ValueError(f"Shape mismatch: {tuple(src_centers.shape)} vs {tuple(tgt_centers.shape)}")
    if src_centers.ndim != 3 or src_centers.shape[-1] != 3:
        raise ValueError(f"Expected shape [B, K, 3], got {tuple(src_centers.shape)}")

    batch_size = src_centers.shape[0]
    device = src_centers.device
    dtype = src_centers.dtype
    eps = 1e-8

    src_mean = src_centers.mean(dim=1, keepdim=True)
    tgt_mean = tgt_centers.mean(dim=1, keepdim=True)
    src_zero = src_centers - src_mean
    tgt_zero = tgt_centers - tgt_mean

    cov = torch.bmm(src_zero.transpose(1, 2), tgt_zero) / max(src_centers.shape[1], 1)
    u, s, v = torch.linalg.svd(cov)
    r = torch.bmm(v.transpose(1, 2), u.transpose(1, 2))
    det = torch.det(r)
    fix = torch.ones(batch_size, 3, device=device, dtype=dtype)
    fix[:, 2] = torch.where(det < 0, torch.tensor(-1.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype))
    correction = torch.diag_embed(fix)
    r = torch.bmm(torch.bmm(v.transpose(1, 2), correction), u.transpose(1, 2))

    if with_scale:
        src_var = torch.mean(torch.sum(src_zero ** 2, dim=-1), dim=1)
        scale = (s * fix).sum(dim=1) / (src_var + eps)
    else:
        scale = torch.ones(batch_size, device=device, dtype=dtype)

    t = tgt_mean[:, 0] - scale.view(batch_size, 1) * torch.einsum("bij,bj->bi", r, src_mean[:, 0])

    sim3 = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
    sim3[:, :3, :3] = scale.view(batch_size, 1, 1) * r
    sim3[:, :3, 3] = t
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


def center_alignment_rmse(aligned_poses: torch.Tensor, tgt_centers: torch.Tensor, overlap_len: int) -> float | None:
    if overlap_len <= 0:
        return None
    aligned_centers = aligned_poses[:, :overlap_len, :3, 3]
    diff = aligned_centers - tgt_centers
    return float(torch.sqrt((diff ** 2).mean()).item())


def canonical_online_chunk_pass(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    chunk_size: int,
    overlap: int,
    conf_thre: float,
    inject_condition: list[str],
    dtype: torch.dtype | None,
    poses: torch.Tensor | None,
    rays: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
) -> tuple[torch.Tensor, list[dict[str, object]], dict[str, object]]:
    batch_size, total_frames, _, height, width = imgs.shape
    if chunk_size <= 1:
        raise RuntimeError("`--chunk_size` must be > 1.")
    if overlap < 0 or overlap >= chunk_size:
        raise RuntimeError("Need `0 <= overlap < chunk_size`.")

    step = chunk_size - overlap
    chunk_records: list[dict[str, object]] = []
    merged_poses: list[torch.Tensor] = []
    chunk_ranges: list[list[int]] = []
    overlap_scales: list[float] = []
    overlap_valid_points: list[int] = []
    overlap_alignment_rmse_m: list[float | None] = []

    prev_global_pts_overlap = None
    prev_global_mask_overlap = None
    prev_aligned_poses_overlap = None
    prev_local_depth_overlap = None
    prev_local_conf_overlap = None
    prev_rays_overlap = None

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if imgs.device.type == "cuda" and dtype is not None
        else nullcontext()
    )

    for start_idx in range(0, total_frames, step):
        end_idx = min(start_idx + chunk_size, total_frames)
        current_len = end_idx - start_idx
        overlap_len = min(overlap, current_len)
        if current_len <= overlap and start_idx > 0:
            break

        chunk_ranges.append([int(start_idx), int(end_idx)])
        chunk_imgs = imgs[:, start_idx:end_idx]

        model_kwargs: dict[str, object] = {"with_prior": False}
        chunk_poses = None if poses is None else poses[:, start_idx:end_idx]
        chunk_rays = None if rays is None else rays[:, start_idx:end_idx]
        chunk_intrinsics = None if intrinsics is None else intrinsics[:, start_idx:end_idx]

        if chunk_poses is not None:
            model_kwargs["poses"] = chunk_poses
            model_kwargs["mask_add_pose"] = torch.ones((batch_size, current_len), dtype=torch.bool, device=imgs.device)
            model_kwargs["with_prior"] = True

        if chunk_rays is not None:
            model_kwargs["rays"] = chunk_rays
            model_kwargs["mask_add_ray"] = torch.ones((batch_size, current_len), dtype=torch.bool, device=imgs.device)
            model_kwargs["with_prior"] = True
        elif chunk_intrinsics is not None:
            model_kwargs["intrinsics"] = chunk_intrinsics
            model_kwargs["mask_add_ray"] = torch.ones((batch_size, current_len), dtype=torch.bool, device=imgs.device)

        if start_idx > 0 and overlap_len > 0:
            if (
                chunk_poses is None
                and "pose" in inject_condition
                and prev_aligned_poses_overlap is not None
            ):
                prior_poses = torch.eye(4, device=imgs.device, dtype=imgs.dtype).repeat(batch_size, current_len, 1, 1)
                prior_poses[:, :overlap_len] = prev_aligned_poses_overlap[:, :overlap_len]
                mask_pose = torch.zeros((batch_size, current_len), dtype=torch.bool, device=imgs.device)
                mask_pose[:, :overlap_len] = True
                model_kwargs["poses"] = prior_poses
                model_kwargs["mask_add_pose"] = mask_pose
                model_kwargs["with_prior"] = True

            if (
                "depth" in inject_condition
                and prev_local_depth_overlap is not None
            ):
                prior_depths = torch.zeros((batch_size, current_len, height, width), device=imgs.device)
                prior_depths[:, :overlap_len] = prev_local_depth_overlap[:, :overlap_len]
                mask_depth = torch.zeros((batch_size, current_len), dtype=torch.bool, device=imgs.device)
                mask_depth[:, :overlap_len] = True
                if prev_local_conf_overlap is not None:
                    valid_mask = prev_local_conf_overlap[:, :overlap_len] > conf_thre
                    prior_depths[:, :overlap_len][~valid_mask] = 0
                model_kwargs["depths"] = prior_depths
                model_kwargs["mask_add_depth"] = mask_depth
                model_kwargs["with_prior"] = True

            if (
                chunk_rays is None
                and chunk_intrinsics is None
                and ("ray" in inject_condition or "intrinsic" in inject_condition)
                and prev_rays_overlap is not None
            ):
                prior_rays = torch.zeros((batch_size, current_len, height, width, 3), device=imgs.device)
                prior_rays[:, :overlap_len] = prev_rays_overlap[:, :overlap_len]
                mask_ray = torch.zeros((batch_size, current_len), dtype=torch.bool, device=imgs.device)
                mask_ray[:, :overlap_len] = True
                model_kwargs["rays"] = prior_rays
                model_kwargs["mask_add_ray"] = mask_ray
                model_kwargs["with_prior"] = True

        with autocast_ctx:
            pred = model(chunk_imgs, **model_kwargs)

        curr_pts = pred["points"]
        curr_poses = pred["camera_poses"]
        curr_conf = pred["conf"]
        curr_local_depth = pred["local_points"][..., 2]
        curr_rays = pred["rays"]
        curr_mask = compute_point_mask(curr_local_depth, curr_conf, conf_thre=conf_thre)

        if start_idx == 0:
            aligned_pts = curr_pts
            aligned_poses = curr_poses
            overlap_scales.append(1.0)
            overlap_valid_points.append(0)
            overlap_alignment_rmse_m.append(None)
        else:
            assert prev_global_pts_overlap is not None
            assert prev_global_mask_overlap is not None
            src_pts = curr_pts[:, :overlap_len]
            src_mask = curr_mask[:, :overlap_len]
            tgt_pts = prev_global_pts_overlap[:, :overlap_len]
            tgt_mask = prev_global_mask_overlap[:, :overlap_len]

            sim3 = compute_sim3_umeyama_masked(src_pts, tgt_pts, src_mask, tgt_mask)
            aligned_pts = apply_sim3_to_points(curr_pts, sim3)
            aligned_poses = apply_sim3_to_poses(curr_poses, sim3)
            overlap_scales.extend(sim3_scale_values(sim3))

            overlap_mask = src_mask & tgt_mask
            valid_count = int(overlap_mask.sum().item())
            overlap_valid_points.append(valid_count)
            if valid_count > 0:
                diff = aligned_pts[:, :overlap_len] - tgt_pts
                rmse = torch.sqrt((diff[overlap_mask] ** 2).mean()).item()
                overlap_alignment_rmse_m.append(float(rmse))
            else:
                overlap_alignment_rmse_m.append(None)

        chunk_records.append(
            {
                "chunk_idx": len(chunk_records),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "overlap_len": int(overlap_len),
                "raw_poses": curr_poses.detach().float().cpu(),
                "canonical_aligned_poses": aligned_poses.detach().float().cpu(),
            }
        )

        if start_idx == 0:
            merged_poses.append(aligned_poses.detach().float().cpu())
        else:
            merged_poses.append(aligned_poses[:, overlap_len:].detach().float().cpu())

        prev_global_pts_overlap = aligned_pts[:, -overlap_len:] if overlap_len > 0 else aligned_pts[:, :0]
        prev_global_mask_overlap = curr_mask[:, -overlap_len:] if overlap_len > 0 else curr_mask[:, :0]
        prev_aligned_poses_overlap = aligned_poses[:, -overlap_len:] if overlap_len > 0 else aligned_poses[:, :0]
        prev_local_depth_overlap = curr_local_depth[:, -overlap_len:] if overlap_len > 0 else curr_local_depth[:, :0]
        prev_local_conf_overlap = torch.sigmoid(curr_conf[:, -overlap_len:, ..., 0]) if overlap_len > 0 else curr_conf[:, :0, ..., 0]
        prev_rays_overlap = curr_rays[:, -overlap_len:] if overlap_len > 0 else curr_rays[:, :0]

        del pred, curr_pts, curr_poses, curr_conf, curr_local_depth, curr_rays, curr_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if end_idx == total_frames:
            break

    diagnostics = {
        "chunk_ranges_0based_half_open": chunk_ranges,
        "overlap_sim3_scales": overlap_scales,
        "overlap_valid_points": overlap_valid_points,
        "overlap_alignment_rmse_m": overlap_alignment_rmse_m,
    }
    return torch.cat(merged_poses, dim=1), chunk_records, diagnostics


def build_gt_reference_poses(chunk_records: list[dict[str, object]], gt_poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not chunk_records:
        raise RuntimeError("No chunk records found.")
    first_chunk = chunk_records[0]
    raw_first = first_chunk["raw_poses"].to(device=gt_poses.device, dtype=gt_poses.dtype)
    first_len = raw_first.shape[1]
    gt_first = gt_poses[:, :first_len]
    gt_to_ref_sim3 = compute_center_alignment_transform(
        src_centers=gt_first[:, :, :3, 3],
        tgt_centers=raw_first[:, :, :3, 3],
        with_scale=True,
    )
    gt_ref = apply_sim3_to_poses(gt_poses, gt_to_ref_sim3)
    return gt_ref, gt_to_ref_sim3


def stitch_chunks_with_gt_overlap_centers(
    chunk_records: list[dict[str, object]],
    gt_ref_poses: torch.Tensor,
    overlap: int,
    with_scale: bool,
) -> tuple[torch.Tensor, dict[str, object]]:
    merged_poses: list[torch.Tensor] = []
    boundary_scales: list[float] = []
    overlap_center_rmse_m: list[float | None] = []
    chunk_ranges: list[list[int]] = []

    for chunk_idx, record in enumerate(chunk_records):
        start_idx = int(record["start_idx"])
        end_idx = int(record["end_idx"])
        raw_poses = record["raw_poses"].to(device=gt_ref_poses.device, dtype=gt_ref_poses.dtype)
        current_len = raw_poses.shape[1]
        overlap_len = min(overlap, current_len)
        chunk_ranges.append([start_idx, end_idx])

        if chunk_idx == 0:
            aligned_poses = raw_poses
            boundary_scales.append(1.0)
            overlap_center_rmse_m.append(None)
        else:
            src_centers = raw_poses[:, :overlap_len, :3, 3]
            tgt_centers = gt_ref_poses[:, start_idx : start_idx + overlap_len, :3, 3]
            sim3 = compute_center_alignment_transform(src_centers, tgt_centers, with_scale=with_scale)
            aligned_poses = apply_sim3_to_poses(raw_poses, sim3)
            boundary_scales.extend(sim3_scale_values(sim3))
            overlap_center_rmse_m.append(center_alignment_rmse(aligned_poses, tgt_centers, overlap_len))

        if chunk_idx == 0:
            merged_poses.append(aligned_poses.detach().float().cpu())
        else:
            merged_poses.append(aligned_poses[:, overlap_len:].detach().float().cpu())

    diagnostics = {
        "chunk_ranges_0based_half_open": chunk_ranges,
        "boundary_scales": boundary_scales,
        "overlap_center_rmse_m": overlap_center_rmse_m,
        "target_space": "gt_mapped_to_first_chunk_reference",
        "alignment_source": "overlap_camera_centers",
        "with_scale": bool(with_scale),
    }
    return torch.cat(merged_poses, dim=1), diagnostics


def summarize_mode(pred_poses: torch.Tensor, gt_ref_poses: torch.Tensor, diagnostics: dict[str, object]) -> dict[str, object]:
    pred_np_raw = pred_poses[0].detach().cpu().numpy().astype(np.float64)
    gt_ref_np_raw = gt_ref_poses[0].detach().cpu().numpy().astype(np.float64)
    return summarize_mode_from_numpy(pred_np_raw, gt_ref_np_raw, diagnostics)


def summarize_mode_from_numpy(
    pred_np_raw: np.ndarray,
    gt_ref_np_raw: np.ndarray,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    pred_np = normalize_pose_rotations(pred_np_raw)
    gt_ref_np = normalize_pose_rotations(gt_ref_np_raw)

    pred_origin_np, _ = align_predicted_poses_to_gt(pred_np, gt_ref_np)
    pred_umeyama_np, _ = align_predicted_poses_umeyama(pred_np, gt_ref_np)
    pred_rel_np = to_first_frame_relative(pred_np)
    gt_rel_np = to_first_frame_relative(gt_ref_np)

    metrics_raw = compute_pose_metrics(pred_np, gt_ref_np)
    metrics_origin = compute_pose_metrics(pred_origin_np, gt_ref_np)
    metrics_umeyama = compute_pose_metrics(pred_umeyama_np, gt_ref_np)
    metrics_first_frame_relative = compute_pose_metrics(pred_rel_np, gt_rel_np)

    last_frame_translation_err = float(metrics_first_frame_relative["per_frame"]["ape_translation_m"][-1])
    last_frame_rotation_err = float(metrics_first_frame_relative["per_frame"]["ape_rotation_deg"][-1])

    return {
        "metrics": {
            "raw": metrics_raw,
            "origin_aligned_to_ref_gt": metrics_origin,
            "umeyama_aligned_to_ref_gt": metrics_umeyama,
            "first_frame_relative_to_ref_gt": metrics_first_frame_relative,
        },
        "diagnostics": diagnostics,
        "metric_evaluation": {
            "rotation_block_projection": "Each pose 3x3 block is projected to the nearest SO(3) rotation before pose metrics are computed.",
        },
        "error_scalars": {
            "first_frame_relative_ape_translation_rmse_m": float(metrics_first_frame_relative["ape_translation_m"]["rmse"]),
            "first_frame_relative_ape_rotation_rmse_deg": float(metrics_first_frame_relative["ape_rotation_deg"]["rmse"]),
            "last_frame_translation_error_m": last_frame_translation_err,
            "last_frame_rotation_error_deg": last_frame_rotation_err,
        },
        "pred_poses_raw": pred_np_raw.tolist(),
        "pred_poses_rotation_normalized_for_metrics": pred_np.tolist(),
    }


def safe_share(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def build_attribution(
    current_mode: dict[str, object],
    oracle_se3_mode: dict[str, object],
    oracle_sim3_mode: dict[str, object],
) -> dict[str, object]:
    keys = [
        "first_frame_relative_ape_translation_rmse_m",
        "last_frame_translation_error_m",
    ]
    outputs: dict[str, object] = {}
    for key in keys:
        current_err = float(current_mode["error_scalars"][key])
        oracle_se3_err = float(oracle_se3_mode["error_scalars"][key])
        oracle_sim3_err = float(oracle_sim3_mode["error_scalars"][key])

        scale_component = max(0.0, oracle_se3_err - oracle_sim3_err)
        pose_component = max(0.0, oracle_sim3_err)
        other_boundary_component = max(0.0, current_err - oracle_se3_err)

        outputs[key] = {
            "current_pred_overlap_sim3_error": current_err,
            "oracle_gt_overlap_centers_se3_error": oracle_se3_err,
            "oracle_gt_overlap_centers_sim3_error": oracle_sim3_err,
            "controlled_decomposition": {
                "scale_component": scale_component,
                "pose_component": pose_component,
                "scale_share_within_se3_oracle_error": safe_share(scale_component, oracle_se3_err),
                "pose_share_within_se3_oracle_error": safe_share(pose_component, oracle_se3_err),
            },
            "production_gap_decomposition": {
                "other_boundary_component": other_boundary_component,
                "scale_component": scale_component,
                "pose_component": pose_component,
                "other_boundary_share_of_current": safe_share(other_boundary_component, current_err),
                "scale_share_of_current": safe_share(scale_component, current_err),
                "pose_share_of_current": safe_share(pose_component, current_err),
            },
        }
    return outputs


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_float_array(values: list[float | int | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)


def _mode_plot_items(summary: dict[str, object]) -> list[tuple[str, str, str]]:
    return [
        ("current_pred_overlap_sim3", "Current pred-overlap Sim3", "#d62728"),
        ("oracle_gt_overlap_centers_se3", "Oracle GT-overlap SE3", "#1f77b4"),
        ("oracle_gt_overlap_centers_sim3", "Oracle GT-overlap Sim3", "#2ca02c"),
    ]


def refresh_summary_metrics_from_saved_poses(summary: dict[str, object], output_dir: Path) -> dict[str, object]:
    gt_ref_path = output_dir / "gt_poses_reference.npy"
    if not gt_ref_path.exists():
        return summary

    gt_ref_np = np.load(gt_ref_path).astype(np.float64)
    pose_files = {
        "current_pred_overlap_sim3": output_dir / "pred_poses_current_pred_overlap_sim3.npy",
        "oracle_gt_overlap_centers_se3": output_dir / "pred_poses_oracle_gt_overlap_centers_se3.npy",
        "oracle_gt_overlap_centers_sim3": output_dir / "pred_poses_oracle_gt_overlap_centers_sim3.npy",
    }

    for mode_key, pose_path in pose_files.items():
        if not pose_path.exists() or mode_key not in summary["modes"]:
            continue
        pred_np = np.load(pose_path).astype(np.float64)
        diagnostics = summary["modes"][mode_key]["diagnostics"]
        summary["modes"][mode_key] = summarize_mode_from_numpy(pred_np, gt_ref_np, diagnostics)

    if all(mode_key in summary["modes"] for mode_key, _ in pose_files.items()):
        summary["attribution"] = build_attribution(
            summary["modes"]["current_pred_overlap_sim3"],
            summary["modes"]["oracle_gt_overlap_centers_se3"],
            summary["modes"]["oracle_gt_overlap_centers_sim3"],
        )

    summary.setdefault("notes", [])
    rotation_note = (
        "Pose metrics project each pose 3x3 block to the nearest SO(3) rotation before evaluating rotation errors, "
        "so Sim3-stitched trajectories are compared with valid rotations."
    )
    if rotation_note not in summary["notes"]:
        summary["notes"].append(rotation_note)
    return summary


def render_visualizations(summary: dict[str, object], output_dir: Path) -> list[Path]:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    deprecated_attr_path = output_dir / "viz_error_attribution.png"
    if deprecated_attr_path.exists():
        deprecated_attr_path.unlink()

    chunk_ranges = summary["modes"]["current_pred_overlap_sim3"]["diagnostics"]["chunk_ranges_0based_half_open"]
    chunk_starts = [int(chunk[0]) for chunk in chunk_ranges]
    chunk_labels = [f"{int(start)}-{int(end)}" for start, end in chunk_ranges]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for mode_key, label, color in _mode_plot_items(summary):
        metrics = summary["modes"][mode_key]["metrics"]["first_frame_relative_to_ref_gt"]["per_frame"]
        translation = _to_float_array(metrics["ape_translation_m"])
        rotation = _to_float_array(metrics["ape_rotation_deg"])
        rmse_t = float(summary["modes"][mode_key]["error_scalars"]["first_frame_relative_ape_translation_rmse_m"])
        rmse_r = float(summary["modes"][mode_key]["error_scalars"]["first_frame_relative_ape_rotation_rmse_deg"])
        frames = np.arange(translation.shape[0], dtype=np.int32)
        axes[0].plot(frames, translation, label=f"{label} (RMSE {rmse_t:.2f} m)", color=color, linewidth=2)
        axes[1].plot(frames, rotation, label=f"{label} (RMSE {rmse_r:.2f} deg)", color=color, linewidth=2)

    for ax in axes:
        for chunk_start in chunk_starts[1:]:
            ax.axvline(chunk_start, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[0].set_title("First-frame-relative trajectory error over time")
    axes[0].set_ylabel("Translation error (m)")
    axes[1].set_ylabel("Rotation error (deg, SO(3)-projected)")
    axes[1].set_xlabel("Frame index")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    path = output_dir / "viz_error_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    boundary_x = np.arange(len(chunk_labels), dtype=np.int32)
    for mode_key, label, color in _mode_plot_items(summary):
        diag = summary["modes"][mode_key]["diagnostics"]
        if "overlap_sim3_scales" in diag:
            scales = _to_float_array(diag["overlap_sim3_scales"])
        else:
            scales = _to_float_array(diag["boundary_scales"])
        if "overlap_alignment_rmse_m" in diag:
            rmse = _to_float_array(diag["overlap_alignment_rmse_m"])
        else:
            rmse = _to_float_array(diag["overlap_center_rmse_m"])
        axes[0].plot(boundary_x, scales, marker="o", color=color, linewidth=2, label=label)
        axes[1].plot(boundary_x, rmse, marker="o", color=color, linewidth=2, label=label)

    axes[0].axhline(1.0, color="#444444", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].set_title("Chunk-boundary stitching diagnostics")
    axes[0].set_ylabel("Boundary scale")
    axes[1].set_ylabel("Local overlap-center RMSE (m)")
    axes[1].set_xlabel("Chunk range")
    axes[1].set_xticks(boundary_x)
    axes[1].set_xticklabels(chunk_labels, rotation=20, ha="right")
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend()
    fig.tight_layout()
    path = output_dir / "viz_boundary_diagnostics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(path)

    config = summary["config"]
    notes = [
        f"Sequence: {config['seq']} | Frames: {config['num_frames']} | Chunk: {config['chunk_size']} | Overlap: {config['overlap']}",
        "Files:",
    ]
    notes.extend([f"- {path.name}" for path in saved_paths])
    notes.extend(
        [
            "",
            "Interpretation:",
            "- viz_error_curves.png: full-sequence drift trends for all three stitching policies.",
            "- viz_boundary_diagnostics.png: local overlap-center fit quality and boundary scale, useful for locating bad chunk transitions.",
            "",
            "Metric note:",
            "- Rotation curves use pose rotations projected to the nearest valid SO(3) matrix before evaluation.",
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
        summary = refresh_summary_metrics_from_saved_poses(load_json(summary_path), summary_path.parent)
        save_json(summary_path, summary)
        saved_paths = render_visualizations(summary, summary_path.parent)
        print("Saved visualizations:")
        for path in saved_paths:
            print(path)
        return

    device = get_runtime_device(args.device)
    seq = str(args.seq).zfill(2)

    output_dir = Path(args.output_dir) / (
        f"seq{seq}_n{args.num_frames}_chunk{args.chunk_size}_overlap{args.overlap}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    model.disable_prior_scale_aug_for_inference = bool(args.disable_prior_scale_aug_for_inference)
    autocast_dtype = get_autocast_dtype(device)

    print("Running canonical online chunk inference once...")
    with torch.inference_mode():
        baseline_poses, chunk_records, baseline_diag = canonical_online_chunk_pass(
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

    gt_ref_poses, gt_to_ref_sim3 = build_gt_reference_poses(chunk_records, gt_poses_t)

    print("Re-stitching saved raw chunk poses with GT-overlap oracle SE3...")
    oracle_se3_poses, oracle_se3_diag = stitch_chunks_with_gt_overlap_centers(
        chunk_records=chunk_records,
        gt_ref_poses=gt_ref_poses,
        overlap=args.overlap,
        with_scale=False,
    )

    print("Re-stitching saved raw chunk poses with GT-overlap oracle Sim3...")
    oracle_sim3_poses, oracle_sim3_diag = stitch_chunks_with_gt_overlap_centers(
        chunk_records=chunk_records,
        gt_ref_poses=gt_ref_poses,
        overlap=args.overlap,
        with_scale=True,
    )

    current_summary = summarize_mode(baseline_poses, gt_ref_poses, baseline_diag)
    oracle_se3_summary = summarize_mode(oracle_se3_poses, gt_ref_poses, oracle_se3_diag)
    oracle_sim3_summary = summarize_mode(oracle_sim3_poses, gt_ref_poses, oracle_sim3_diag)
    attribution = build_attribution(current_summary, oracle_se3_summary, oracle_sim3_summary)

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
            "disable_prior_scale_aug_for_inference": bool(args.disable_prior_scale_aug_for_inference),
            "device": str(device),
            "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
            "camera_key": camera_key,
            "model_image_size": {
                "width": int(tensor_metadata["target_width"]),
                "height": int(tensor_metadata["target_height"]),
            },
        },
        "reference_world": {
            "definition": (
                "All chunk trajectories are compared against GT poses mapped into the "
                "first chunk's raw predicted reference by a single Sim3 estimated from "
                "the whole first chunk camera-center trajectory."
            ),
            "gt_to_first_chunk_reference_sim3": gt_to_ref_sim3[0].detach().cpu().numpy().tolist(),
        },
        "modes": {
            "current_pred_overlap_sim3": current_summary,
            "oracle_gt_overlap_centers_se3": oracle_se3_summary,
            "oracle_gt_overlap_centers_sim3": oracle_sim3_summary,
        },
        "attribution": attribution,
        "notes": [
            "The canonical online pass matches the existing Pi3X chunk pipeline: later chunks may consume overlap priors from the baseline stitching path.",
            "The offline oracle modes keep the same raw chunk predictions and only change the boundary alignment policy, so they isolate stitching effects from chunk content prediction.",
            "Use `oracle_gt_overlap_centers_se3` vs `oracle_gt_overlap_centers_sim3` to estimate how much of the long-sequence drift comes from boundary scale mismatch versus residual intra-chunk pose drift.",
            "Use `current_pred_overlap_sim3` vs `oracle_gt_overlap_centers_se3` to gauge additional production-gap error from overlap point matching, mask quality, and error accumulation through chained chunk stitching.",
            "Pose metrics project each pose 3x3 block to the nearest SO(3) rotation before evaluating rotation errors, so Sim3-stitched trajectories are compared with valid rotations.",
        ],
    }

    save_json(output_dir / "chunk_stitch_ablation_summary.json", summary)
    np.save(output_dir / "gt_poses_world.npy", gt_poses_t[0].detach().cpu().numpy().astype(np.float64))
    np.save(output_dir / "gt_poses_reference.npy", gt_ref_poses[0].detach().cpu().numpy().astype(np.float64))
    np.save(output_dir / "pred_poses_current_pred_overlap_sim3.npy", baseline_poses[0].detach().cpu().numpy().astype(np.float64))
    np.save(output_dir / "pred_poses_oracle_gt_overlap_centers_se3.npy", oracle_se3_poses[0].detach().cpu().numpy().astype(np.float64))
    np.save(output_dir / "pred_poses_oracle_gt_overlap_centers_sim3.npy", oracle_sim3_poses[0].detach().cpu().numpy().astype(np.float64))
    saved_visualizations = render_visualizations(summary, output_dir)

    print("Saved outputs:")
    print(output_dir / "chunk_stitch_ablation_summary.json")
    print(output_dir / "gt_poses_world.npy")
    print(output_dir / "gt_poses_reference.npy")
    print(output_dir / "pred_poses_current_pred_overlap_sim3.npy")
    print(output_dir / "pred_poses_oracle_gt_overlap_centers_se3.npy")
    print(output_dir / "pred_poses_oracle_gt_overlap_centers_sim3.npy")
    for path in saved_visualizations:
        print(path)


if __name__ == "__main__":
    main()
