#!/usr/bin/env python3
"""Visualize KITTI odometry ground-truth trajectories (bird's-eye X–Z), first frame at world origin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_kitti_root(kitti_root: str) -> Path:
    candidate = Path(kitti_root).expanduser().resolve()
    if candidate.exists():
        return candidate
    fallback = REPO_ROOT / "dataset" / "kitti-od"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        f"KITTI root does not exist: {candidate}\nFallback root also not found: {fallback}"
    )


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_pose_file(kitti_root: Path, sequence: str) -> Path | None:
    seq = sequence.zfill(2)
    return first_existing_path(
        [
            kitti_root / "poses" / f"{seq}.txt",
            kitti_root / "data_odometry_poses" / "poses" / f"{seq}.txt",
            kitti_root / "data_odometry_poses" / "dataset" / "poses" / f"{seq}.txt",
        ]
    )


def load_pose_file(pose_file: Path) -> np.ndarray:
    poses: list[np.ndarray] = []
    with pose_file.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = np.fromstring(stripped, sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(
                    f"Pose file {pose_file} line {line_idx}: expected 12 values, got {values.size}."
                )
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :] = values.reshape(3, 4)
            poses.append(pose)
    if not poses:
        raise ValueError(f"Pose file is empty: {pose_file}")
    return np.stack(poses, axis=0)


def to_first_frame_relative(poses_np: np.ndarray) -> np.ndarray:
    first_inv = np.linalg.inv(poses_np[0])
    return np.einsum("ij,njk->nik", first_inv, poses_np)


def discover_sequences_with_poses(kitti_root: Path) -> list[str]:
    found: set[str] = set()
    for sub in ("data_odometry_poses/poses", "poses"):
        d = kitti_root / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.txt"):
            if p.stem.isdigit():
                found.add(p.stem.zfill(2))
    return sorted(found, key=lambda s: int(s))


def plot_sequence_bev(
    poses_world_first: np.ndarray,
    sequence: str,
    marker_stride: int,
    out_path: Path,
    dpi: int,
) -> None:
    # KITTI camera trajectory: translation column is position in the (original) world frame;
    # after to_first_frame_relative, first pose is identity → origin.
    x = poses_world_first[:, 0, 3]
    z = poses_world_first[:, 2, 3]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.plot(x, z, color="#2d6a2d", linewidth=1.8, label="Ground Truth", zorder=2)

    ax.scatter(
        [0.0],
        [0.0],
        c="black",
        s=45,
        zorder=6,
        label="Start",
    )
    ax.plot(
        [0.0],
        [0.0],
        marker="+",
        color="tab:blue",
        markersize=12,
        markeredgewidth=2.0,
        linestyle="none",
        label="World origin (0,0,0)",
        zorder=5,
    )

    idx = np.arange(0, len(x), marker_stride, dtype=np.int64)
    ax.scatter(
        x[idx],
        z[idx],
        c="red",
        s=22,
        zorder=4,
        label=f"Every {marker_stride} frames",
    )
    for i in idx:
        ax.annotate(
            str(int(i)),
            (float(x[i]), float(z[i])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="black",
            zorder=5,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="-", alpha=0.35)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Sequence {sequence} — Bird-Eye View (World)")
    ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot KITTI GT trajectories (BEV X–Z), first frame as world origin."
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=str(REPO_ROOT / "dataset" / "kitti-od"),
        help="KITTI odometry root (contains data_odometry_poses/poses/ etc.).",
    )
    p.add_argument(
        "--seq",
        type=str,
        nargs="*",
        default=None,
        help="Sequence ids (e.g. 00 03). Default: all sequences that have a pose file.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "debug_outputs" / "kitti_gt_trajectory_bev"),
        help="Directory to save PNG files.",
    )
    p.add_argument(
        "--marker_stride",
        type=int,
        default=100,
        help="Place a red marker every N frames (0, N, 2N, ...).",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_kitti_root(args.data_root)
    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.seq is None or len(args.seq) == 0:
        sequences = discover_sequences_with_poses(root)
        if not sequences:
            raise FileNotFoundError(f"No pose *.txt found under {root}.")
    else:
        sequences = [s.zfill(2) for s in args.seq]

    for seq in sequences:
        pose_path = find_pose_file(root, seq)
        if pose_path is None:
            print(f"[skip] No pose file for sequence {seq}")
            continue
        poses = load_pose_file(pose_path)
        poses_rel = to_first_frame_relative(poses)
        out_path = out_dir / f"seq{seq}_gt_bev.png"
        plot_sequence_bev(
            poses_rel,
            seq,
            max(1, args.marker_stride),
            out_path,
            args.dpi,
        )
        print(f"Wrote {out_path} ({poses.shape[0]} poses)")


if __name__ == "__main__":
    main()
