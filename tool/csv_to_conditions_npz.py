#!/usr/bin/env python3
'''
python /root/autodl-tmp/Pi3/tool/csv_to_conditions_npz.py \
  --camera_matrix_csv /root/autodl-tmp/Pi3/data/sofa/camera_matrix.csv \
  --odometry_csv /root/autodl-tmp/Pi3/data/sofa/odometry.csv \
  --image_dir /root/autodl-tmp/Pi3/data/sofa/images \
  --out_npz /root/autodl-tmp/Pi3/data/sofa/condition.npz \
  --input_pose_type c2w
'''
import argparse
import csv
from pathlib import Path

import numpy as np


def quat_xyzw_to_rotmat(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Invalid quaternion with near-zero norm.")
    qx, qy, qz, qw = q / n

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    xw, yw, zw = qx * qw, qy * qw, qz * qw

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def read_intrinsics(camera_matrix_csv):
    k = np.loadtxt(camera_matrix_csv, delimiter=",", dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got shape={k.shape}")
    return k


def read_odometry_as_c2w(odometry_csv, input_pose_type):
    """
    Parse CSV columns: timestamp, frame, x, y, z, qx, qy, qz, qw
    and convert to camera-to-world (OpenCV) 4x4 matrices.
    """
    pose_by_frame = {}
    with open(odometry_csv, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        required = {"frame", "x", "y", "z", "qx", "qy", "qz", "qw"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"odometry CSV missing required columns. got={reader.fieldnames}")

        for row in reader:
            frame_idx = int(row["frame"])
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            qx = float(row["qx"])
            qy = float(row["qy"])
            qz = float(row["qz"])
            qw = float(row["qw"])

            r = quat_xyzw_to_rotmat(qx, qy, qz, qw)
            t = np.array([x, y, z], dtype=np.float64)

            t_mat = np.eye(4, dtype=np.float64)
            t_mat[:3, :3] = r
            t_mat[:3, 3] = t

            if input_pose_type == "c2w":
                c2w = t_mat
            elif input_pose_type == "w2c":
                c2w = np.linalg.inv(t_mat)
            else:
                raise ValueError(f"Unsupported input_pose_type={input_pose_type}")

            pose_by_frame[frame_idx] = c2w

    return pose_by_frame


def collect_image_indices(image_dir):
    image_dir = Path(image_dir)
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    files = []
    for pat in patterns:
        files.extend(image_dir.glob(pat))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    indices = []
    for p in files:
        try:
            indices.append(int(p.stem))
        except ValueError as e:
            raise ValueError(f"Image filename stem must be integer-like, got: {p.name}") from e
    return indices, files


def main():
    parser = argparse.ArgumentParser(
        description="Convert camera_matrix.csv + odometry.csv to Pi3/Pi3X condition.npz"
    )
    parser.add_argument("--camera_matrix_csv", type=str, required=True, help="Path to 3x3 camera matrix CSV.")
    parser.add_argument("--odometry_csv", type=str, required=True, help="Path to odometry CSV.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory used by --data_path.")
    parser.add_argument("--out_npz", type=str, required=True, help="Output .npz path.")
    parser.add_argument(
        "--input_pose_type",
        type=str,
        default="c2w",
        choices=["c2w", "w2c"],
        help="Interpret odometry as camera-to-world or world-to-camera before conversion.",
    )
    parser.add_argument(
        "--allow_missing_pose",
        action="store_true",
        help="Skip images that do not have a matching frame pose in odometry.",
    )
    args = parser.parse_args()

    k = read_intrinsics(args.camera_matrix_csv)
    pose_by_frame = read_odometry_as_c2w(args.odometry_csv, args.input_pose_type)
    image_indices, files = collect_image_indices(args.image_dir)

    selected_poses = []
    used_files = []
    for idx, f in zip(image_indices, files):
        if idx in pose_by_frame:
            selected_poses.append(pose_by_frame[idx])
            used_files.append(f.name)
        elif not args.allow_missing_pose:
            raise KeyError(
                f"Missing pose for image frame {idx:06d} ({f.name}). "
                "Use --allow_missing_pose to skip unmatched images."
            )

    if not selected_poses:
        raise RuntimeError("No matched image/pose pairs found.")

    poses = np.stack(selected_poses, axis=0).astype(np.float32)  # (N, 4, 4), c2w
    intrinsics = np.repeat(k[None, :, :], repeats=poses.shape[0], axis=0).astype(np.float32)  # (N, 3, 3)

    # Keep a placeholder key to satisfy example_mm.py's unconditional data_npz['depths'] access.
    depths = np.empty((0,), dtype=np.float32)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, poses=poses, intrinsics=intrinsics, depths=depths)

    print(f"Saved: {out_npz}")
    print(f"Matched frames: {poses.shape[0]}")
    print(f"poses shape: {poses.shape}, intrinsics shape: {intrinsics.shape}, depths shape: {depths.shape}")
    print(f"First image: {used_files[0]}, Last image: {used_files[-1]}")


if __name__ == "__main__":
    main()
