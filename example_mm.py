import argparse
import os
from pathlib import Path

import numpy as np
import torch

from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_edge
from tool.ply_to_colmap_bin import (
    align_points_to_condition_frame,
    build_colmap_records,
    load_conditions,
    load_images,
    maybe_downsample,
    print_debug_stats,
    read_ply_xyz_rgb,
    write_colmap_binary_model,
)


def export_colmap_bin_from_ply(
    ply_path,
    out_dir,
    conditions_npz,
    image_dir,
    pred_poses_npz=None,
    frame_interval=1,
    camera_model="PINHOLE",
    shared_camera=False,
    min_track_len=2,
    max_points=200000,
    seed=0,
    debug_stats=False,
    debug_sample_points=50000,
    debug_frame_step=1,
):
    points_xyz_full, points_rgb_full = read_ply_xyz_rgb(ply_path)
    poses, intrinsics = load_conditions(conditions_npz)
    image_paths, width, height = load_images(image_dir)

    step = max(1, int(frame_interval))
    if step > 1:
        image_paths = image_paths[::step]
        poses = poses[::step]
        intrinsics = intrinsics[::step]
        print(f"[INFO] COLMAP export uses interval={step} sampled frames.")

    image_names = [p.name for p in image_paths]
    n = min(len(image_names), poses.shape[0], intrinsics.shape[0])
    if n <= 0:
        raise RuntimeError("No overlapping frames among images/poses/intrinsics for COLMAP export.")
    if n < len(image_names) or n < poses.shape[0] or n < intrinsics.shape[0]:
        print(
            f"[WARN] Frame count mismatch, trimming to {n}: "
            f"images={len(image_names)}, poses={poses.shape[0]}, intrinsics={intrinsics.shape[0]}"
        )
    image_names = image_names[:n]
    poses = poses[:n]
    intrinsics = intrinsics[:n]

    if pred_poses_npz is not None:
        points_xyz_full = align_points_to_condition_frame(
            points_xyz=points_xyz_full,
            condition_poses_c2w=poses,
            pred_poses_npz=pred_poses_npz,
        )

    if debug_stats:
        print_debug_stats(
            points_xyz=points_xyz_full,
            poses=poses,
            intrinsics=intrinsics,
            width=width,
            height=height,
            sample_points=debug_sample_points,
            frame_step=max(1, int(debug_frame_step)),
            seed=seed,
        )

    points_xyz, points_rgb = maybe_downsample(points_xyz_full, points_rgb_full, max_points, seed)
    cameras, images, points = build_colmap_records(
        points_xyz=points_xyz,
        points_rgb=points_rgb,
        poses=poses,
        intrinsics=intrinsics,
        image_names=image_names,
        width=width,
        height=height,
        camera_model=camera_model,
        shared_camera=shared_camera,
        min_track_len=min_track_len,
    )
    write_colmap_binary_model(out_dir, cameras, images, points)

    print(f"Saved COLMAP model to: {out_dir}")
    print(f"images={len(images)}, cameras={len(cameras)}, points3D={len(points)}")
    print("Generated files: cameras.bin, images.bin, points3D.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pi3X inference and export PLY/COLMAP.")

    parser.add_argument(
        "--data_path",
        type=str,
        default="examples/skating.mp4",
        help="Path to the input image directory or a video file.",
    )
    parser.add_argument(
        "--conditions_path",
        type=str,
        default=None,
        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="examples/result.ply",
        help="Path to save the output .ply file.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=-1,
        help="Interval to sample image. Default: 1 for images dir, 10 for video.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to the model checkpoint file. Default: None.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'.",
    )

    # COLMAP export (default ON)
    parser.add_argument(
        "--disable_colmap_bin",
        action="store_true",
        help="Disable auto-export of COLMAP bin files after PLY is saved.",
    )
    parser.add_argument(
        "--colmap_out_dir",
        type=str,
        default=None,
        help="Output directory for COLMAP bin files. Default: <save_path_stem>_colmap_sparse.",
    )
    parser.add_argument(
        "--colmap_pred_poses_npz",
        type=str,
        default=None,
        help="Path to save predicted camera poses npz. Default: <save_path_stem>_pred_camera_poses.npz.",
    )
    parser.add_argument(
        "--no_colmap_pred_alignment",
        action="store_true",
        help="Disable Sim3 alignment using predicted camera poses.",
    )
    parser.add_argument(
        "--colmap_camera_model",
        type=str,
        default="PINHOLE",
        choices=["SIMPLE_PINHOLE", "PINHOLE"],
        help="COLMAP camera model.",
    )
    parser.add_argument(
        "--colmap_shared_camera",
        action="store_true",
        help="Use one shared camera for all images when exporting COLMAP.",
    )
    parser.add_argument(
        "--colmap_min_track_len",
        type=int,
        default=2,
        help="Minimum visible frames required per 3D point for COLMAP.",
    )
    parser.add_argument(
        "--colmap_max_points",
        type=int,
        default=200000,
        help="Randomly downsample points before COLMAP export. <=0 means keep all.",
    )
    parser.add_argument(
        "--colmap_seed",
        type=int,
        default=0,
        help="Random seed for COLMAP point downsampling.",
    )
    parser.add_argument(
        "--colmap_debug_stats",
        action="store_true",
        help="Print geometry diagnostics during COLMAP export.",
    )
    parser.add_argument(
        "--colmap_debug_sample_points",
        type=int,
        default=50000,
        help="Sample point count for --colmap_debug_stats.",
    )
    parser.add_argument(
        "--colmap_debug_frame_step",
        type=int,
        default=1,
        help="Frame step for --colmap_debug_stats.",
    )

    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith(".mp4") else 1
    print(f"Sampling interval: {args.interval}")

    print("Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3X().to(device).eval()
        if args.ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()

    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)
        poses = data_npz["poses"]
        depths = data_npz["depths"]
        intrinsics = data_npz["intrinsics"]

    conditions = dict(intrinsics=intrinsics, poses=poses, depths=depths)
    imgs, conditions = load_multimodal_data(
        args.data_path, conditions, interval=args.interval, device=device
    )

    print("Running model inference...")
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    with torch.no_grad():
        with torch.amp.autocast(
            "cuda", dtype=dtype, enabled=(device.type == "cuda")
        ):
            res = model(imgs=imgs, **conditions)

    masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    print(f"Saving point cloud to: {args.save_path}")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(res["points"][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], str(save_path))
    print("PLY saved.")

    if args.disable_colmap_bin:
        print("COLMAP export disabled by --disable_colmap_bin.")
    else:
        if args.conditions_path is None or not os.path.exists(args.conditions_path):
            print("[WARN] Skip COLMAP export: --conditions_path is required.")
        elif not os.path.isdir(args.data_path):
            print("[WARN] Skip COLMAP export: --data_path must be an image directory.")
        else:
            pred_poses_npz = (
                Path(args.colmap_pred_poses_npz)
                if args.colmap_pred_poses_npz is not None
                else save_path.with_name(f"{save_path.stem}_pred_camera_poses.npz")
            )
            np.savez_compressed(
                str(pred_poses_npz),
                camera_poses=res["camera_poses"][0].detach().cpu().numpy().astype(np.float32),
            )
            print(f"Saved predicted camera poses: {pred_poses_npz}")

            colmap_out_dir = (
                Path(args.colmap_out_dir)
                if args.colmap_out_dir is not None
                else save_path.with_name(f"{save_path.stem}_colmap_sparse")
            )
            pred_alignment_npz = None if args.no_colmap_pred_alignment else str(pred_poses_npz)

            export_colmap_bin_from_ply(
                ply_path=str(save_path),
                out_dir=str(colmap_out_dir),
                conditions_npz=args.conditions_path,
                image_dir=args.data_path,
                pred_poses_npz=pred_alignment_npz,
                frame_interval=args.interval,
                camera_model=args.colmap_camera_model,
                shared_camera=args.colmap_shared_camera,
                min_track_len=args.colmap_min_track_len,
                max_points=args.colmap_max_points,
                seed=args.colmap_seed,
                debug_stats=args.colmap_debug_stats,
                debug_sample_points=args.colmap_debug_sample_points,
                debug_frame_step=args.colmap_debug_frame_step,
            )

    print("Done.")
