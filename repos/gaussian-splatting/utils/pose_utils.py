import torch
import torch.nn as nn


def skew_symmetric(v):
    """Convert a 3-vector to a 3x3 skew-symmetric matrix."""
    zero = torch.zeros(1, device=v.device, dtype=v.dtype)
    return torch.stack([
        torch.cat([zero, -v[2:3], v[1:2]]),
        torch.cat([v[2:3], zero, -v[0:1]]),
        torch.cat([-v[1:2], v[0:1], zero]),
    ], dim=0)


def se3_exp_map(xi):
    """Map a 6-vector (omega, v) to a 4x4 SE(3) matrix via Rodrigues formula.

    Args:
        xi: (6,) tensor — first 3 are rotation (omega), last 3 are translation (v)

    Returns:
        (4, 4) SE(3) matrix
    """
    omega = xi[:3]
    v = xi[3:]
    theta = omega.norm()

    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    omega_hat = skew_symmetric(omega)

    if theta < 1e-8:
        # First-order Taylor: R = I + [omega]_x, t = v
        T[:3, :3] = T[:3, :3] + omega_hat
        T[:3, 3] = v
    else:
        omega_hat_sq = omega_hat @ omega_hat
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        theta_sq = theta * theta
        theta_cu = theta_sq * theta

        # Rodrigues rotation
        R = torch.eye(3, device=xi.device, dtype=xi.dtype) + \
            (sin_t / theta) * omega_hat + \
            ((1.0 - cos_t) / theta_sq) * omega_hat_sq

        # V matrix for translation
        V = torch.eye(3, device=xi.device, dtype=xi.dtype) + \
            ((1.0 - cos_t) / theta_sq) * omega_hat + \
            ((theta - sin_t) / theta_cu) * omega_hat_sq

        T[:3, :3] = R
        T[:3, 3] = V @ v

    return T


class PoseRefiner(nn.Module):
    """Learnable SE(3) perturbations for each camera.

    Stores a (num_cameras, 6) parameter of Lie algebra deltas initialized to
    zero (identity perturbation).
    """

    def __init__(self, num_cameras):
        super().__init__()
        self.delta_xi = nn.Parameter(torch.zeros(num_cameras, 6))
        self._preconditioner = None

    def set_preconditioner(self, P):
        """Store a (6, 6) CamP preconditioning matrix."""
        if not isinstance(P, torch.Tensor):
            P = torch.tensor(P, dtype=torch.float32)
        self._preconditioner = P.cuda()

    def transform_gaussians_for_camera(self, means3D, cam_idx, T_init_w2v):
        """Transform Gaussian means to account for the learned pose perturbation.

        Convention: 3DGS stores world_view_transform as W2V **transposed**
        (row-major), so T_init_w2v.T gives the standard W2V matrix.

        The world-space perturbation is:
            M = T_v2w @ exp(delta_xi) @ T_w2v
        and we transform: means3D' = (cat([means3D, ones]) @ M.T)[:, :3]

        Args:
            means3D: (N, 3) Gaussian centers in world space
            cam_idx: integer camera index
            T_init_w2v: (4, 4) initial world_view_transform (stored transposed)

        Returns:
            (N, 3) transformed Gaussian centers
        """
        delta = self.delta_xi[cam_idx]
        # Standard W2V (column-major)
        T_w2v = T_init_w2v.T
        T_v2w = torch.inverse(T_w2v)

        exp_delta = se3_exp_map(delta)
        # World-space perturbation
        M = T_v2w @ exp_delta @ T_w2v

        N = means3D.shape[0]
        ones = torch.ones(N, 1, device=means3D.device, dtype=means3D.dtype)
        means_h = torch.cat([means3D, ones], dim=1)  # (N, 4)
        means_transformed = (means_h @ M.T)[:, :3]
        return means_transformed

    def get_perturbed_campos(self, cam_idx, T_init_w2v):
        """Compute the perturbed camera position for SH view-direction evaluation.

        Args:
            cam_idx: integer camera index
            T_init_w2v: (4, 4) initial world_view_transform (stored transposed)

        Returns:
            (3,) camera center in world space
        """
        delta = self.delta_xi[cam_idx]
        T_w2v = T_init_w2v.T
        exp_delta = se3_exp_map(delta)
        T_perturbed = exp_delta @ T_w2v
        T_perturbed_inv = torch.inverse(T_perturbed)
        return T_perturbed_inv[:3, 3]


def compute_camp_preconditioner(point_cloud_xyz, camera_centers):
    """Compute CamP (Camera-space Preconditioning) matrix.

    ZCA whitening of camera center covariance for translation components,
    scene-scale normalization for rotation components.

    Args:
        point_cloud_xyz: (M, 3) point cloud positions
        camera_centers: (N, 3) camera centers

    Returns:
        (6, 6) block-diagonal preconditioning matrix
    """
    # Scene scale from point cloud
    scene_scale = point_cloud_xyz.detach().cpu().norm(dim=1).median().item()
    if scene_scale < 1e-6:
        scene_scale = 1.0

    # Camera center covariance
    centers = camera_centers.detach().cpu().float()
    mean_c = centers.mean(dim=0, keepdim=True)
    centered = centers - mean_c
    cov = (centered.T @ centered) / max(len(centers) - 1, 1)

    # ZCA whitening: W = cov^{-1/2}
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=1e-6)
    W_trans = eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.T

    # Rotation preconditioning: scale by 1/scene_scale
    W_rot = torch.eye(3) / scene_scale

    # Block-diagonal (6x6): [rotation | translation]
    P = torch.zeros(6, 6)
    P[:3, :3] = W_rot
    P[3:, 3:] = W_trans

    return P
