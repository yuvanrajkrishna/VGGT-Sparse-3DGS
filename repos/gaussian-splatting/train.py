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

import os
import json
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, weighted_l1_loss, weighted_ssim, huber_loss, pearson_depth_loss, pearson_depth_loss_weighted, wavelet_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.pose_utils import PoseRefiner, compute_camp_preconditioner
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def get_drop_schedule(iteration, init_rate, peak_rate, peak_iter, end_iter):
    """Progressive DropGaussian schedule: ramp up to peak, then anneal to 0."""
    if peak_rate <= 0:
        return 0.0
    if iteration <= 0:
        return init_rate
    if iteration <= peak_iter:
        # Linear ramp: init_rate -> peak_rate
        t = iteration / max(peak_iter, 1)
        return init_rate + (peak_rate - init_rate) * t
    elif iteration <= end_iter:
        # Linear decay: peak_rate -> 0
        t = (iteration - peak_iter) / max(end_iter - peak_iter, 1)
        return peak_rate * (1.0 - t)
    else:
        return 0.0


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # init-time confidence filter percentile (before Scene loads pcd)
    gaussians.conf_percentile_filter = float(opt.conf_percentile_filter)
    # Confidence-gated densification flag (Patched)
    gaussians._use_conf_densify_gate = bool(opt.use_conf_densify_gate)
    gaussians._conf_densify_threshold = float(opt.conf_densify_threshold)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Pose refinement setup
    pose_refiner = None
    pose_optimizer = None
    pose_lr_func = None
    if opt.refine_poses:
        n_cams = len(scene.getTrainCameras())
        pose_refiner = PoseRefiner(n_cams).cuda()
        pose_lr_func = get_expon_lr_func(opt.pose_lr_init, opt.pose_lr_final, max_steps=opt.pose_lr_max_steps)
        pose_optimizer = torch.optim.Adam([pose_refiner.delta_xi], lr=opt.pose_lr_init, eps=1e-15)
        print(f"[POSE] Initialized pose refinement for {n_cams} cameras")

        if opt.pose_use_camp and not opt.pose_no_camp:
            cam_centers = torch.stack([c.camera_center for c in scene.getTrainCameras()])
            P = compute_camp_preconditioner(gaussians.get_xyz.detach(), cam_centers)
            pose_refiner.set_preconditioner(P)
            print("[POSE] CamP preconditioning enabled")

    # LPIPS for pseudo-view loss
    lpips_fn = None
    if opt.use_pseudo_views:
        from lpipsPyTorch.modules.lpips import LPIPS
        lpips_fn = LPIPS(net_type='alex').cuda().eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
        print("[PSEUDO] LPIPS loss initialized")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Determine pose phase
        pose_active = (opt.refine_poses and opt.pose_start_iter <= iteration <= opt.pose_end_iter)

        # DD-Drop: refresh per-Gaussian local density score periodically
        if opt.use_dd_drop and iteration % opt.dd_density_update_interval == 0:
            gaussians.update_density_score()

        # Every 1000 iterations we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        use_pseudo_this_iter = False
        pseudo_cams = scene.getPseudoCameras() if opt.use_pseudo_views else []
        in_mixed_phase = (opt.pseudo_phase1_end < iteration <= opt.pseudo_phase2_end)

        if pseudo_cams and in_mixed_phase and random.random() < opt.pseudo_sample_prob:
            use_pseudo_this_iter = True
            viewpoint_cam = pseudo_cams[randint(0, len(pseudo_cams) - 1)]
            vind = -1  # Sentinel: no pose refinement for pseudo
        else:
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        drop_prob = get_drop_schedule(iteration, opt.drop_prob_init, opt.drop_prob_peak,
                                      opt.drop_prob_peak_iter, opt.drop_prob_end_iter)
        # DD-Drop kwargs 
        dd_drop_params = None
        if opt.use_dd_drop and not use_pseudo_this_iter:
            dd_drop_params = {
                "depth_weight": opt.dd_depth_weight,
                "density_weight": opt.dd_density_weight,
                "conf_weight": opt.dd_conf_weight,
                "drop_min": opt.dd_drop_min,
                "drop_max": opt.dd_drop_max,
                "iteration": iteration,
                "max_iter": opt.iterations,
            }
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE,
                            pose_refiner=pose_refiner if (pose_active and not use_pseudo_this_iter) else None,
                            cam_idx=vind if (pose_active and not use_pseudo_this_iter) else None,
                            drop_prob=drop_prob if not opt.use_dd_drop else 0.0,
                            dd_drop_params=dd_drop_params)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss — always use vanilla formula (pose gradients flow via SE(3) chain rule)
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        if use_pseudo_this_iter and lpips_fn is not None:
            # Triangular annealing weight
            t_frac = (iteration - opt.pseudo_phase1_end) / max(opt.pseudo_phase2_end - opt.pseudo_phase1_end, 1)
            pseudo_w = opt.pseudo_weight_peak * (2 * t_frac if t_frac < 0.5 else 2 * (1 - t_frac))
            pseudo_w = max(pseudo_w, opt.pseudo_weight_init)

            lpips_val = lpips_fn(image.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).squeeze()
            loss = pseudo_w * (0.8 * Ll1 + 0.2 * (1 - ssim_value) + opt.pseudo_lpips_weight * lpips_val)
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Wavelet frequency regularization (skip for pseudo-views)
        if opt.wavelet_weight > 0 and not use_pseudo_this_iter:
            loss = loss + opt.wavelet_weight * wavelet_loss(image, gt_image, opt.wavelet_sparse_lambda)

        # DAFE: Distance-Aware Fidelity Enhancement (D2GS).
        # Boost L1 loss on far-field pixels (where Gaussians underfit). Far mask
        # derived from the loaded mono inverse-depth (low invdepth = far).
        if opt.lambda_far > 0 and viewpoint_cam.depth_reliable and not use_pseudo_this_iter:
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            # Far-field mask: select the bottom (far_mask_quantile)-fraction of
            # invdepth pixels. Small invdepth = far away. With default 0.33 this
            # selects the farthest 33% of pixels (matches D2GS DAFE intent).
            valid_vals = mono_invdepth[depth_mask > 0.5]
            if valid_vals.numel() > 100:
                thr = torch.quantile(valid_vals.flatten(), opt.far_mask_quantile)
                far_mask = (mono_invdepth < thr).float() * depth_mask
                # Expand to RGB and apply
                far_mask_rgb = far_mask.unsqueeze(0).expand_as(image) if far_mask.dim() == 2 else far_mask.expand_as(image)
                far_loss = torch.abs((image - gt_image) * far_mask_rgb).mean()
                loss = loss + opt.lambda_far * far_loss

        # Depth regularization (skip for pseudo-views)
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable and not use_pseudo_this_iter:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            if opt.use_pearson_depth:
                if opt.use_depth_conf and viewpoint_cam.depth_conf is not None:
                    # Confidence-weighted Pearson loss .
                    # depth_conf is already sigmoid-passed in Camera; apply optional
                    # temperature to sharpen/dampen the weighting.
                    conf_w = viewpoint_cam.depth_conf.cuda()
                    if abs(opt.depth_conf_temperature - 1.0) > 1e-6:
                        # Reapply sigmoid-style sharpening: w' = sigmoid(T * logit(w))
                        # but cheaper: just raise to a power
                        conf_w = conf_w ** opt.depth_conf_temperature
                    Ll1depth_pure = pearson_depth_loss_weighted(invDepth, mono_invdepth, conf_w, depth_mask)
                else:
                    Ll1depth_pure = pearson_depth_loss(invDepth, mono_invdepth, depth_mask)
            else:
                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Pose regularization (skip for pseudo-views)
        if pose_active and not use_pseudo_this_iter:
            loss = loss + opt.pose_reg_lambda * (pose_refiner.delta_xi[vind] ** 2).sum()

        loss.backward()

        # Pose optimizer step (before torch.no_grad block)
        if pose_active and pose_optimizer is not None:
            torch.nn.utils.clip_grad_norm_([pose_refiner.delta_xi], max_norm=opt.pose_grad_clip)
            if opt.pose_use_camp and not opt.pose_no_camp and pose_refiner._preconditioner is not None:
                g = pose_refiner.delta_xi.grad
                if g is not None:
                    pose_refiner.delta_xi.grad.data = g @ pose_refiner._preconditioner.T
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
            for pg in pose_optimizer.param_groups:
                pg['lr'] = pose_lr_func(iteration)
        elif opt.refine_poses and pose_optimizer is not None:
            pose_optimizer.zero_grad(set_to_none=True)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            # Pose refinement logging
            if opt.refine_poses and tb_writer and iteration % 100 == 0:
                with torch.no_grad():
                    delta_norms = pose_refiner.delta_xi.detach().norm(dim=1)
                    tb_writer.add_scalar('pose/mean_delta_norm', delta_norms.mean().item(), iteration)
                    tb_writer.add_scalar('pose/max_delta_norm', delta_norms.max().item(), iteration)
                    tb_writer.add_scalar('pose/lr', pose_lr_func(iteration), iteration)
                    reg_loss = opt.pose_reg_lambda * (pose_refiner.delta_xi ** 2).sum().item()
                    tb_writer.add_scalar('pose/reg_loss', reg_loss, iteration)
                    tb_writer.add_scalar('pose/active', float(pose_active), iteration)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification / MCMC (skip stats for pseudo-views to avoid spurious splits)
            if opt.use_mcmc:
                # MCMC-3DGS: stochastic relocalization (no heuristic clone/split)
                if not use_pseudo_this_iter:
                    if iteration > opt.densify_from_iter and iteration % opt.mcmc_relocate_interval == 0 and iteration < opt.densify_until_iter:
                        n_relocated = gaussians.relocate_dead_gaussians(opt.mcmc_dead_threshold)
                        if n_relocated > 0 and iteration % 1000 == 0:
                            print(f"\n[MCMC] Relocated {n_relocated} dead Gaussians at iter {iteration}")
                    if iteration % opt.mcmc_add_interval == 0 and iteration < opt.densify_until_iter:
                        n_added = gaussians.add_new_gaussians(opt.mcmc_cap_max, opt.mcmc_add_ratio)
                        if n_added > 0 and iteration % 1000 == 0:
                            print(f"\n[MCMC] Added {n_added} Gaussians at iter {iteration} (total: {gaussians.get_xyz.shape[0]})")
                    if iteration < opt.densify_until_iter:
                        gaussians.mcmc_noise_step(opt.mcmc_noise_lr, gaussians.spatial_lr_scale)
            elif iteration < opt.densify_until_iter and not use_pseudo_this_iter:
                # Original heuristic densification
                # Exclude dropped Gaussians from densification stats — they contribute
                # nothing to the rendered image and would poison split/clone decisions.
                vis_for_densify = visibility_filter
                drop_mask = render_pkg.get("drop_mask")
                if drop_mask is not None:
                    # visibility_filter is indices from .nonzero(); drop_mask is bool (N,)
                    survived = drop_mask[visibility_filter.squeeze()]
                    vis_for_densify = visibility_filter[survived]
                gaussians.max_radii2D[vis_for_densify] = torch.max(gaussians.max_radii2D[vis_for_densify], radii[vis_for_densify])
                gaussians.add_densification_stats(viewspace_point_tensor, vis_for_densify)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Floater pruning: once after densification ends (not for MCMC)
            if iteration == opt.densify_until_iter + 500 and not opt.use_mcmc:
                n_before = gaussians.get_xyz.shape[0]
                gaussians.prune_floaters(scene.getTrainCameras())
                print(f"\n[Floater pruning] {n_before} -> {gaussians.get_xyz.shape[0]}")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                # Two-stage training: skip gaussian update during pose-only phase
                if opt.use_two_stage_pose and iteration <= opt.two_stage_split_iter:
                    gaussians.optimizer.zero_grad(set_to_none = True)
                elif use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # Save refined poses after training
    if opt.refine_poses and pose_refiner is not None:
        refined_poses = []
        train_cams = scene.getTrainCameras()
        with torch.no_grad():
            for idx, cam in enumerate(train_cams):
                delta = pose_refiner.delta_xi[idx].detach().cpu()
                T_init = cam.world_view_transform_init.cpu()
                # Compute refined W2C
                from utils.pose_utils import se3_exp_map
                T_w2v = T_init.T
                exp_delta = se3_exp_map(delta)
                T_refined = exp_delta @ T_w2v
                refined_poses.append({
                    "image_name": cam.image_name,
                    "initial_w2c": T_w2v.numpy().tolist(),
                    "refined_w2c": T_refined.numpy().tolist(),
                    "delta_xi": delta.numpy().tolist(),
                    "delta_norm": float(delta.norm().item()),
                })
        poses_path = os.path.join(scene.model_path, "refined_poses.json")
        with open(poses_path, "w") as f:
            json.dump(refined_poses, f, indent=2)
        print(f"[POSE] Saved refined poses to {poses_path}")
        mean_delta = sum(p["delta_norm"] for p in refined_poses) / len(refined_poses)
        max_delta = max(p["delta_norm"] for p in refined_poses)
        print(f"[POSE] Mean delta_xi norm: {mean_delta:.6f}, Max: {max_delta:.6f}")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
