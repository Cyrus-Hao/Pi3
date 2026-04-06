from contextlib import nullcontext

from ..utils.geometry import depth_edge
import numpy as np
import torch


class Pi3XVO:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        imgs,
        chunk_size=16,
        overlap=6,
        conf_thre=0.05,
        inject_condition=None,
        dtype=torch.bfloat16,
        poses=None,
        rays=None,
        intrinsics=None,
        depths=None,
    ):
        """
        Args:
            inject_condition: list of strings, e.g. ['pose', 'depth']
            poses: optional full-sequence pose priors, shape [B, T, 4, 4]
            rays: optional full-sequence ray priors, shape [B, T, H, W, 3]
            intrinsics: optional full-sequence intrinsics, shape [B, T, 3, 3]
            depths: optional full-sequence depth priors, shape [B, T, H, W]
        """
        if inject_condition is None:
            inject_condition = []

        B, T, _, H, W = imgs.shape
        print(f"[PiXVO] Total frames: {T}, Chunk size: {chunk_size}, Overlap: {overlap}")

        self._validate_optional_sequence_shape("poses", poses, (B, T))
        self._validate_optional_sequence_shape("rays", rays, (B, T))
        self._validate_optional_sequence_shape("intrinsics", intrinsics, (B, T))
        self._validate_optional_sequence_shape("depths", depths, (B, T))

        merged_points, merged_poses, merged_confs = [], [], []
        chunk_metrics = []
        chunk_ranges = []
        overlap_scales = []
        overlap_valid_points = []

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

        for start_idx in range(0, T, chunk_size - overlap):
            end_idx = min(start_idx + chunk_size, T)
            chunk_imgs = imgs[:, start_idx:end_idx]
            current_len = end_idx - start_idx
            overlap_len = min(overlap, current_len)
            chunk_ranges.append([int(start_idx), int(end_idx)])

            print(f"  > Inference chunk: [{start_idx} : {end_idx}] (Length: {current_len})")

            if current_len <= overlap and start_idx > 0:
                break

            model_kwargs = {"with_prior": False}

            chunk_depths = None if depths is None else depths[:, start_idx:end_idx]
            chunk_poses = None if poses is None else poses[:, start_idx:end_idx]
            chunk_rays = None if rays is None else rays[:, start_idx:end_idx]
            chunk_intrinsics = None if intrinsics is None else intrinsics[:, start_idx:end_idx]

            if chunk_depths is not None:
                model_kwargs["depths"] = chunk_depths
                model_kwargs["mask_add_depth"] = torch.ones((B, current_len), dtype=torch.bool, device=imgs.device)
                model_kwargs["with_prior"] = True

            if chunk_poses is not None:
                model_kwargs["poses"] = chunk_poses
                model_kwargs["mask_add_pose"] = torch.ones((B, current_len), dtype=torch.bool, device=imgs.device)
                model_kwargs["with_prior"] = True

            if chunk_rays is not None:
                model_kwargs["rays"] = chunk_rays
                model_kwargs["mask_add_ray"] = torch.ones((B, current_len), dtype=torch.bool, device=imgs.device)
                model_kwargs["with_prior"] = True
            elif chunk_intrinsics is not None:
                model_kwargs["intrinsics"] = chunk_intrinsics
                model_kwargs["mask_add_ray"] = torch.ones((B, current_len), dtype=torch.bool, device=imgs.device)

            if start_idx > 0:
                if (
                    chunk_poses is None
                    and "pose" in inject_condition
                    and prev_aligned_poses_overlap is not None
                    and overlap_len > 0
                ):
                    prior_poses = torch.eye(4, device=imgs.device).repeat(B, current_len, 1, 1)
                    prior_poses[:, :overlap_len] = prev_aligned_poses_overlap[:, :overlap_len]
                    mask_pose = torch.zeros((B, current_len), dtype=torch.bool, device=imgs.device)
                    mask_pose[:, :overlap_len] = True
                    model_kwargs["poses"] = prior_poses
                    model_kwargs["mask_add_pose"] = mask_pose
                    model_kwargs["with_prior"] = True

                if (
                    chunk_depths is None
                    and "depth" in inject_condition
                    and prev_local_depth_overlap is not None
                    and overlap_len > 0
                ):
                    prior_depths = torch.zeros((B, current_len, H, W), device=imgs.device)
                    prior_depths[:, :overlap_len] = prev_local_depth_overlap[:, :overlap_len]
                    mask_depth = torch.zeros((B, current_len), dtype=torch.bool, device=imgs.device)
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
                    and overlap_len > 0
                ):
                    prior_rays = torch.zeros((B, current_len, H, W, 3), device=imgs.device)
                    prior_rays[:, :overlap_len] = prev_rays_overlap[:, :overlap_len]
                    mask_ray = torch.zeros((B, current_len), dtype=torch.bool, device=imgs.device)
                    mask_ray[:, :overlap_len] = True
                    model_kwargs["rays"] = prior_rays
                    model_kwargs["mask_add_ray"] = mask_ray
                    model_kwargs["with_prior"] = True

            with autocast_ctx:
                pred = self.model(chunk_imgs, **model_kwargs)

            curr_local_depth = pred["local_points"][..., 2]
            curr_pts = pred["points"]
            curr_poses = pred["camera_poses"]
            curr_conf = torch.sigmoid(pred["conf"])[..., 0]
            curr_rays = pred["rays"]
            curr_metric = pred["metric"]
            chunk_metrics.append(curr_metric.unsqueeze(1).detach().cpu())

            edge = depth_edge(curr_local_depth, rtol=0.03)
            curr_conf[edge] = 0
            curr_mask = curr_conf > conf_thre

            if int(curr_mask.sum().item()) < 10:
                flat_conf = curr_conf.view(B, current_len, -1)
                k = max(1, int(flat_conf.shape[-1] * 0.1))
                topk_vals, _ = torch.topk(flat_conf, k, dim=-1)
                min_vals = topk_vals[..., -1].unsqueeze(-1).unsqueeze(-1)
                curr_mask = curr_conf >= min_vals

            if start_idx == 0:
                aligned_pts = curr_pts
                aligned_poses = curr_poses
                overlap_scales.append(1.0)
                overlap_valid_points.append(0)
            else:
                src_pts = curr_pts[:, :overlap_len]
                src_mask = curr_mask[:, :overlap_len]
                tgt_pts = prev_global_pts_overlap[:, :overlap_len]
                tgt_mask = prev_global_mask_overlap[:, :overlap_len]

                transform_matrix = self._compute_sim3_umeyama_masked(
                    src_pts, tgt_pts, src_mask, tgt_mask
                )
                aligned_pts = self._apply_sim3_to_points(curr_pts, transform_matrix)
                aligned_poses = self._apply_sim3_to_poses(curr_poses, transform_matrix)
                overlap_scales.extend(self._sim3_scale_values(transform_matrix))
                overlap_valid_points.append(int((src_mask & tgt_mask).sum().item()))

            if start_idx == 0:
                merged_points.append(aligned_pts)
                merged_poses.append(aligned_poses)
                merged_confs.append(curr_conf)
            else:
                merged_points.append(aligned_pts[:, overlap_len:])
                merged_poses.append(aligned_poses[:, overlap_len:])
                merged_confs.append(curr_conf[:, overlap_len:])

            prev_global_pts_overlap = aligned_pts[:, -overlap_len:] if overlap_len > 0 else aligned_pts[:, :0]
            prev_global_mask_overlap = curr_mask[:, -overlap_len:] if overlap_len > 0 else curr_mask[:, :0]
            prev_aligned_poses_overlap = aligned_poses[:, -overlap_len:] if overlap_len > 0 else aligned_poses[:, :0]
            prev_local_depth_overlap = curr_local_depth[:, -overlap_len:] if overlap_len > 0 else curr_local_depth[:, :0]
            prev_local_conf_overlap = curr_conf[:, -overlap_len:] if overlap_len > 0 else curr_conf[:, :0]
            prev_rays_overlap = curr_rays[:, -overlap_len:] if overlap_len > 0 else curr_rays[:, :0]

            del pred, curr_pts, curr_poses, curr_mask, curr_local_depth, curr_conf, curr_rays, curr_metric
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if end_idx == T:
                break

        chunk_metrics_tensor = torch.cat(chunk_metrics, dim=1) if chunk_metrics else torch.empty((B, 0))
        return {
            "points": torch.cat(merged_points, dim=1),
            "camera_poses": torch.cat(merged_poses, dim=1),
            "conf": torch.cat(merged_confs, dim=1),
            "metric": chunk_metrics_tensor.mean(dim=1) if chunk_metrics else torch.empty((B,)),
            "chunk_metrics": chunk_metrics_tensor,
            "diagnostics": {
                "chunk_ranges_0based_half_open": chunk_ranges,
                "overlap_sim3_scales": overlap_scales,
                "overlap_valid_points": overlap_valid_points,
            },
        }

    def _compute_sim3_umeyama_masked(self, src_points, tgt_points, src_mask, tgt_mask):
        B = src_points.shape[0]
        device = src_points.device
        
        src = src_points.reshape(B, -1, 3)
        tgt = tgt_points.reshape(B, -1, 3)
        
        mask = (src_mask.reshape(B, -1) & tgt_mask.reshape(B, -1)).float().unsqueeze(-1)
        valid_cnt = mask.sum(dim=1).squeeze(-1)
        eps = 1e-6
        
        bad_mask = valid_cnt < 10
        if bad_mask.all():
            return torch.eye(4, device=device).repeat(B, 1, 1)

        src_mean = (src * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)
        tgt_mean = (tgt * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)
        
        src_centered = (src - src_mean) * mask
        tgt_centered = (tgt - tgt_mean) * mask
        
        H = torch.bmm(src_centered.transpose(1, 2), tgt_centered)
        U, S, V = torch.svd(H)
        
        R = torch.bmm(V, U.transpose(1, 2))
        
        det = torch.det(R)
        diag = torch.ones(B, 3, device=device)
        diag[:, 2] = torch.sign(det)
        R = torch.bmm(torch.bmm(V, torch.diag_embed(diag)), U.transpose(1, 2))
        
        src_var = (src_centered ** 2).sum(dim=2) * mask.squeeze(-1)
        src_var = src_var.sum(dim=1) / (valid_cnt + eps)
        
        corrected_S = S.clone()
        corrected_S[:, 2] *= diag[:, 2]
        trace_S = corrected_S.sum(dim=1)
        
        scale = trace_S / (src_var * valid_cnt + eps)
        scale = scale.view(B, 1, 1)
        
        t = tgt_mean.transpose(1, 2) - scale * torch.bmm(R, src_mean.transpose(1, 2))
        
        sim3 = torch.eye(4, device=device).repeat(B, 1, 1)
        sim3[:, :3, :3] = scale * R
        sim3[:, :3, 3] = t.squeeze(2)
        
        if bad_mask.any():
            identity = torch.eye(4, device=device).repeat(B, 1, 1)
            sim3[bad_mask] = identity[bad_mask]
            
        return sim3
    
    def _apply_sim3_to_points(self, points, sim3):
        B, T, H, W, C = points.shape
        flat_pts = points.reshape(B, -1, 3)
        R_s = sim3[:, :3, :3]
        t = sim3[:, :3, 3].unsqueeze(1)
        out_pts = torch.bmm(flat_pts, R_s.transpose(1, 2)) + t
        return out_pts.reshape(B, T, H, W, 3)

    def _apply_sim3_to_poses(self, poses, sim3):
        sim3_expanded = sim3.unsqueeze(1)
        out_poses = torch.matmul(sim3_expanded, poses)
        return out_poses

    def _sim3_scale_values(self, sim3):
        rot_scale = sim3[:, :3, :3]
        det_val = torch.det(rot_scale).detach().float().cpu().numpy()
        return (np.cbrt(np.abs(det_val)) * np.sign(det_val)).tolist()

    def _validate_optional_sequence_shape(self, name, tensor, expected_prefix):
        if tensor is None:
            return
        if tensor.shape[:2] != expected_prefix:
            raise ValueError(
                f"`{name}` should have leading shape {expected_prefix}, got {tuple(tensor.shape)}"
            )