"""msplat: Metal-accelerated 3D Gaussian Splatting."""

import atexit

from msplat._core import (
    TrainingConfig,
    TrainingStats,
    Dataset,
    GaussianTrainer,
    sync,
    cleanup as _cleanup_raw,
)

_cleaned_up = False


def cleanup():
    """Release all cached GPU resources. Safe to call multiple times."""
    global _cleaned_up
    if not _cleaned_up:
        _cleaned_up = True
        _cleanup_raw()


atexit.register(cleanup)

__all__ = [
    "TrainingConfig",
    "TrainingStats",
    "Dataset",
    "GaussianTrainer",
    "sync",
    "cleanup",
    "load_dataset",
    "street_config",
    "STREET_PRESET",
]

__version__ = "1.1.4"


def load_dataset(
    path: str,
    downscale_factor: float = 1.0,
    eval_mode: bool = False,
    test_every: int = 8,
    prior_dir: str = "",
) -> Dataset:
    """Load a dataset (auto-detects COLMAP, Nerfstudio, Polycam).

    Geometric priors (see ``msplat-prior``) are attached from ``prior_dir``, or from
    ``<path>/priors`` when that directory exists.
    """
    return Dataset(path, downscale_factor, eval_mode, test_every, prior_dir)


# Prior-guided settings for street-view and other sparse, forward-moving captures.
# They need priors from msplat-prior (depth, confidence, sky masks): the depth prior
# pins gaussians to surfaces, the sky terms move the sky onto a learned background,
# exposure compensation absorbs auto-exposure, and the scale cap removes the needle
# gaussians forward motion produces. SH degree 1: view dependence is poorly
# constrained when every camera looks the same way.
STREET_PRESET = dict(
    sh_degree=1,
    depth_weight=0.2,
    depth_weight_final=0.05,
    sky_alpha_weight=0.05,
    fill_weight=0.01,
    use_masks=True,
    learn_sky=True,
    exposure_compensation=True,
    max_scale_ratio=10.0,
)


def street_config(**overrides) -> TrainingConfig:
    """TrainingConfig with STREET_PRESET applied; keyword arguments override it."""
    kwargs = dict(STREET_PRESET)
    kwargs.update(overrides)
    return TrainingConfig(**kwargs)
