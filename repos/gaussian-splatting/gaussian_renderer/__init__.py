#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, pose_refiner=None, cam_idx=None, drop_prob=0.0, dd_drop_params=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Resolve Gaussian means and camera position (possibly perturbed by pose refiner)
    means3D = pc.get_xyz
    campos = viewpoint_camera.camera_center
    if pose_refiner is not None and cam_idx is not None:
        means3D = pose_refiner.transform_gaussians_for_camera(
            means3D, cam_idx, viewpoint_camera.world_view_transform_init
        )
        campos = pose_refiner.get_perturbed_campos(
            cam_idx, viewpoint_camera.world_view_transform_init
        )

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc.get_opacity

    # DropGaussian: inverted dropout on opacity (training only)
    drop_mask = None
    if drop_prob > 0.0 and opacity.requires_grad:
        compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
        compensation = torch.nn.functional.dropout(compensation, p=drop_prob, training=True)
        drop_mask = (compensation > 0.0)  # True for surviving Gaussians
        opacity = opacity * compensation.unsqueeze(1)

    # DD-Drop .
    # Score each Gaussian by (depth-from-camera, local density, VGGT confidence) and
    # drop near+dense+confident Gaussians more aggressively than far+sparse+unconfident.
    # Replaces the uniform DropGaussian above when dd_drop_params is provided.
    if dd_drop_params is not None and opacity.requires_grad:
        depth_w = dd_drop_params.get("depth_weight", 0.5)
        dens_w = dd_drop_params.get("density_weight", 0.5)
        conf_w = dd_drop_params.get("conf_weight", 0.0)
        drop_min = dd_drop_params.get("drop_min", 0.05)
        drop_max = dd_drop_params.get("drop_max", 0.3)
        iteration = int(dd_drop_params.get("iteration", 0))
        max_iter = int(dd_drop_params.get("max_iter", 10000))
        with torch.no_grad():
            gaussian_positions = pc.get_xyz
            ones = torch.ones((gaussian_positions.shape[0], 1), device=gaussian_positions.device)
            gaussian_positions_homo = torch.cat([gaussian_positions, ones], dim=1)
            camera_coordinates = torch.matmul(gaussian_positions_homo, viewpoint_camera.world_view_transform.T)
            camera_depths = camera_coordinates[:, 2]
            depth_min, depth_max = camera_depths.min(), camera_depths.max()
            depth_score = (1.0 - (camera_depths - depth_min) / (depth_max - depth_min + 1e-6)).float()
            sorted_depths, _ = torch.sort(camera_depths)
            n = sorted_depths.shape[0]
            idx_33 = int(n * 0.33)
            idx_67 = int(n * 0.67)
            d33 = sorted_depths[idx_33].float()
            d67 = sorted_depths[idx_67].float()
            near_field = (camera_depths <= d33).float()
            mid_field = ((camera_depths > d33) & (camera_depths <= d67)).float()
            far_field = (camera_depths > d67).float()
            density_norm = torch.ones_like(depth_score) * 0.5
            if hasattr(pc, "density_score") and pc.density_score.numel() >= opacity.shape[0]:
                ds = pc.density_score[:opacity.shape[0]].float()
                density_norm = ((ds - ds.min()) / (ds.max() - ds.min() + 1e-6)).float()
            conf_norm = torch.zeros_like(depth_score)
            if conf_w > 0 and pc.vggt_confidence is not None and pc.vggt_confidence.numel() >= opacity.shape[0]:
                cn = pc.vggt_confidence[:opacity.shape[0]].float()
                if cn.max() > cn.min():
                    conf_norm = ((cn - cn.min()) / (cn.max() - cn.min() + 1e-6)).float()
            combined = (depth_w * depth_score + dens_w * density_norm + conf_w * conf_norm).float()
            progress = min(1.0, iteration / max_iter)
            drop_rate = float(drop_min + (drop_max - drop_min) * progress)
            drop_prob_pg = (near_field * combined * drop_rate +
                            mid_field * combined * drop_rate * 0.7 +
                            far_field * combined * drop_rate * 0.3)
            keep_prob = 1.0 - drop_prob_pg.clamp(0.0, 1.0)
            mask = (torch.rand_like(keep_prob) < keep_prob).float()
        # Inverted dropout: zero out dropped Gaussians' opacity (NO scale-up since
        # drop_rate is small)
        opacity = opacity * mask[:, None]
        drop_mask = (mask > 0.5)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - campos.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        "drop_mask": drop_mask,
        }
    
    return out
