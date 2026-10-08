"""Type stubs for msplat._core (compiled nanobind extension)."""

import numpy as np
from numpy.typing import NDArray

class TrainingConfig:
    iterations: int
    sh_degree: int
    sh_degree_interval: int
    ssim_weight: float
    num_downscales: int
    resolution_schedule: int
    refine_every: int
    warmup_length: int
    reset_alpha_every: int
    densify_grad_thresh: float
    densify_size_thresh: float
    stop_screen_size_at: int
    split_screen_size: float
    keep_crs: bool
    downscale_factor: float
    output: str
    save_every: int
    bg_color: list[float]
    """Background color as [R, G, B] floats in [0, 1]. Default magenta [0.613, 0.010, 0.398]."""
    depth_weight: float
    """Weight of the log-depth prior loss (0 = off). Needs priors/depth."""
    depth_weight_final: float
    """Depth prior weight at the last step, log-linear schedule (<0: constant, 0: linear to 0)."""
    depth_huber_delta: float
    """Huber transition of the depth loss, in log depth (~ relative error)."""
    depth_min_alpha: float
    """Depth is only supervised where accumulated alpha exceeds this."""
    sky_alpha_weight: float
    """Push sky-mask pixels transparent so the learned sky shows. Needs priors/sky."""
    fill_weight: float
    """Push non-sky pixels opaque (cameras with a sky mask)."""
    use_masks: bool
    """Drop all gradients where priors/mask is 0 (moving objects)."""
    learn_sky: bool
    """Learn a direction-dependent sky (equirect texture) behind the gaussians."""
    sky_width: int
    sky_height: int
    sky_lr: float
    exposure_compensation: bool
    """Per-image affine color transform, absorbing auto-exposure/white balance."""
    exposure_lr: float
    exposure_reg: float
    max_scale_ratio: float
    """Cap each gaussian's largest/median scale ratio (needle suppression; <=1 = off)."""

    def __init__(
        self,
        iterations: int = 30000,
        sh_degree: int = 3,
        sh_degree_interval: int = 1000,
        ssim_weight: float = 0.2,
        num_downscales: int = 2,
        resolution_schedule: int = 3000,
        refine_every: int = 100,
        warmup_length: int = 500,
        reset_alpha_every: int = 30,
        densify_grad_thresh: float = 0.0002,
        densify_size_thresh: float = 0.01,
        stop_screen_size_at: int = 4000,
        split_screen_size: float = 0.05,
        keep_crs: bool = False,
        downscale_factor: float = 1.0,
        output: str = "splat.ply",
        save_every: int = -1,
        bg_color: list[float] = ...,
        depth_weight: float = 0.0,
        depth_weight_final: float = -1.0,
        depth_huber_delta: float = 0.05,
        depth_min_alpha: float = 0.25,
        sky_alpha_weight: float = 0.0,
        fill_weight: float = 0.0,
        use_masks: bool = False,
        learn_sky: bool = False,
        sky_width: int = 512,
        sky_height: int = 128,
        sky_lr: float = 0.01,
        exposure_compensation: bool = False,
        exposure_lr: float = 0.005,
        exposure_reg: float = 0.01,
        max_scale_ratio: float = 0.0,
    ) -> None: ...

class TrainingStats:
    """Per-step training statistics returned by GaussianTrainer.step()."""

    @property
    def iteration(self) -> int:
        """Current training iteration."""
        ...

    @property
    def splat_count(self) -> int:
        """Number of active Gaussians."""
        ...

    @property
    def ms_per_step(self) -> float:
        """Wall-clock time for this step in milliseconds."""
        ...

class Dataset:
    """A loaded dataset of camera images. Auto-detects COLMAP, Nerfstudio, and Polycam formats."""

    def __init__(
        self,
        path: str,
        downscale_factor: float = 1.0,
        eval_mode: bool = False,
        test_every: int = 8,
        prior_dir: str = "",
    ) -> None:
        """prior_dir: directory of geometric priors (depth/, confidence/, sky/, mask/).

        Defaults to <path>/priors when present; transforms.json frame keys take precedence.
        """
        ...

    @property
    def num_train(self) -> int:
        """Number of training cameras."""
        ...

    @property
    def num_test(self) -> int:
        """Number of test cameras (0 unless eval_mode=True)."""
        ...

    def camera_pose(self, index: int) -> NDArray[np.float32]:
        """Get camera-to-world pose (4x4 row-major, OpenGL convention) as numpy array."""
        ...

    def prior_counts(self) -> dict[str, int]:
        """Number of training cameras with each kind of prior file attached."""
        ...

class GaussianTrainer:
    """3D Gaussian Splatting trainer. All computation runs on the Metal GPU."""

    def __init__(self, dataset: Dataset, config: TrainingConfig) -> None: ...

    def step(self) -> TrainingStats:
        """Run a single training iteration. Returns TrainingStats."""
        ...

    def train(
        self,
        callback: object,
        callback_every: int = 100,
    ) -> None:
        """Run training to completion, calling callback(stats) every callback_every steps."""
        ...

    def evaluate(self) -> dict[str, float | int]:
        """Evaluate on held-out test cameras. Returns dict with psnr, ssim, l1 keys.

        Requires the dataset to have been loaded with eval_mode=True.
        """
        ...

    def render(
        self,
        cam_idx: int,
        use_test: bool = False,
    ) -> NDArray[np.float32]:
        """Render a camera view. Returns a numpy array of shape (H, W, 3), float32, RGB [0,1]."""
        ...

    def render_from_pose(
        self,
        cam_to_world: NDArray[np.float32],
        ref_cam_idx: int = 0,
    ) -> NDArray[np.float32]:
        """Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention).

        Uses intrinsics from ref_cam_idx. Returns numpy (H, W, 3) float32.
        """
        ...

    def render_depth(
        self,
        cam_idx: int,
        use_test: bool = False,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Render expected depth (dataset units, 0 where empty) and accumulated alpha.

        Returns a tuple of two numpy (H, W) float32 arrays.
        """
        ...

    def prior_losses(self) -> dict[str, float]:
        """Mean prior losses of the last step: dict with depth, sky, fill. Syncs the GPU."""
        ...

    def export_sky(self, path: str) -> None:
        """Save the learned sky as an equirect PNG (requires learn_sky=True)."""
        ...

    def export_ply(self, path: str) -> None:
        """Export the current Gaussians as a PLY file."""
        ...

    def export_splat(self, path: str) -> None:
        """Export the current Gaussians as a .splat file."""
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save a training checkpoint."""
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint and resume from the saved iteration."""
        ...

    @property
    def splat_count(self) -> int:
        """Current number of active Gaussians."""
        ...

    @property
    def iteration(self) -> int:
        """Current training iteration."""
        ...

def sync() -> None:
    """Synchronize GPU (wait for all commands to complete)."""
    ...

def cleanup() -> None:
    """Release all cached GPU resources."""
    ...
