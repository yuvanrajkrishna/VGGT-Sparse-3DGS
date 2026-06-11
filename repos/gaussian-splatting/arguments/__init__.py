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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.llffhold = 8
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        # Pose refinement parameters
        self.refine_poses = False
        self.pose_lr_init = 1e-4
        self.pose_lr_final = 1e-6
        self.pose_lr_max_steps = 30000
        self.pose_reg_lambda = 0.01
        self.pose_grad_clip = 0.01
        self.pose_start_iter = 500
        self.pose_end_iter = 25000
        self.pose_huber_delta = 0.05          # deprecated: vanilla loss used for all params
        self.pose_sh_promote_interval = 2000  # deprecated: standard 1000-iter interval always used
        self.pose_use_camp = True
        self.pose_no_camp = False
        # Pseudo-view parameters
        self.use_pseudo_views = False
        self.pseudo_phase1_end = 500
        self.pseudo_phase2_end = 25000
        self.pseudo_weight_init = 0.05
        self.pseudo_weight_peak = 0.3
        self.pseudo_lpips_weight = 0.1
        self.pseudo_sample_prob = 0.15
        # DropGaussian: progressive schedule (0 = disabled)
        self.drop_prob_init = 0.0
        self.drop_prob_peak = 0.0
        self.drop_prob_peak_iter = 5000
        self.drop_prob_end_iter = 9000
        # Pearson correlation depth loss
        self.use_pearson_depth = False
        # Confidence-weighted Pearson depth loss .
        # When use_depth_conf is True AND viewpoint has depth_conf, weight each
        # pixel by VGGT confidence in the Pearson correlation.
        self.use_depth_conf = False
        self.depth_conf_temperature = 1.0  # sharpness multiplier for sigmoid
        # DD-Drop 
        self.use_dd_drop = False
        self.dd_depth_weight = 0.5
        self.dd_density_weight = 0.5
        self.dd_conf_weight = 0.0
        self.dd_drop_min = 0.05
        self.dd_drop_max = 0.3
        self.dd_density_update_interval = 500  # recompute kNN density every N iters
        # DAFE (distance-aware fidelity enhancement, from D2GS)
        self.lambda_far = 0.0  # 0 = off; typical 0.5-1.0
        self.far_mask_quantile = 0.33  # bottom-33% of invdepth = farthest 33% of pixels
        # Init-time VGGT-confidence percentile filter .
        # Drops the bottom-X% of init points by VGGT confidence BEFORE training.
        self.conf_percentile_filter = 0.0  # e.g. 0.2 drops bottom 20%
        # Two-stage training : break joint optimization basin trap.
        # Stage 1 (iter 0..two_stage_split_iter): pose-only, gaussians frozen.
        # Stage 2 (iter two_stage_split_iter+1..end): gaussian-only, poses frozen.
        # Requires --refine_poses; pose_start/end_iter should be set to (0, split).
        self.use_two_stage_pose = False
        self.two_stage_split_iter = 1000
        # Confidence-gated densification : only allow split/clone
        # of Gaussians whose VGGT confidence exceeds threshold. Prevents
        # bad-pose-induced wrong densification.
        self.use_conf_densify_gate = False
        self.conf_densify_threshold = 0.5
        # Wavelet frequency regularization
        self.wavelet_weight = 0.0
        self.wavelet_sparse_lambda = 0.01
        # MCMC-3DGS (mutually exclusive with DropGaussian)
        self.use_mcmc = False
        self.mcmc_cap_max = 200000
        self.mcmc_noise_lr = 5e5
        self.mcmc_dead_threshold = 0.005
        self.mcmc_relocate_interval = 100
        self.mcmc_add_interval = 300
        self.mcmc_add_ratio = 0.05
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
