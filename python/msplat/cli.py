"""msplat-train CLI entry point."""

import sys


def main():
    try:
        import tyro
    except ImportError:
        print("Install tyro for CLI support: pip install msplat[cli]", file=sys.stderr)
        sys.exit(1)

    from dataclasses import dataclass, field
    from typing import Literal, Optional

    @dataclass
    class Args:
        """Train 3D Gaussian Splatting on a dataset."""

        input: str
        """Path to dataset (COLMAP, Nerfstudio, etc.)"""

        output: str = "splat.ply"
        """Output PLY file path"""

        num_iters: int = 30000
        """Number of training iterations"""

        downscale_factor: float = 1.0
        """Image downscale factor"""

        num_downscales: int = 2
        """Number of progressive downscales"""

        resolution_schedule: int = 3000
        """Double resolution every N steps"""

        sh_degree: int = 3
        """Max spherical harmonics degree"""

        sh_degree_interval: int = 1000
        """Steps between SH degree increases"""

        ssim_weight: float = 0.2
        """SSIM loss weight"""

        refine_every: int = 100
        """Densification interval"""

        warmup_length: int = 500
        """Steps before densification starts"""

        reset_alpha_every: int = 30
        """Reset opacity every N refinements"""

        densify_grad_thresh: float = 0.0002
        """Gradient threshold for densification"""

        densify_size_thresh: float = 0.01
        """Size threshold for split vs clone"""

        stop_screen_size_at: int = 4000
        """Stop screen-size split after this step"""

        split_screen_size: float = 0.05
        """Screen-space split threshold"""

        keep_crs: bool = False
        """Keep input coordinate reference system"""

        save_every: int = -1
        """Save every N steps (-1 to disable)"""

        eval: bool = False
        """Evaluate on held-out test views"""

        test_every: int = 8
        """Hold out every Nth image for eval"""

        preset: Optional[Literal["street"]] = None
        """'street': prior-guided settings for street-view / sparse forward captures (needs msplat-prior output)"""

        prior_dir: str = ""
        """Prior directory (default: <input>/priors)"""

        depth_weight: Optional[float] = None
        """Depth prior loss weight (0 = off)"""

        depth_weight_final: Optional[float] = None
        """Depth prior weight at the last step (<0: constant)"""

        sky_alpha_weight: Optional[float] = None
        """Push sky-mask pixels transparent"""

        fill_weight: Optional[float] = None
        """Push non-sky pixels opaque (needs sky masks)"""

        use_masks: Optional[bool] = None
        """Ignore pixels where priors/mask is 0 (moving objects)"""

        learn_sky: Optional[bool] = None
        """Learn a direction-dependent sky behind the gaussians"""

        exposure_compensation: Optional[bool] = None
        """Per-image affine exposure compensation"""

        max_scale_ratio: Optional[float] = None
        """Cap largest/median gaussian scale (0 = off)"""

    args = tyro.cli(Args)

    from msplat import TrainingConfig, Dataset, GaussianTrainer, sync, cleanup, STREET_PRESET

    # Preset first, then every prior option given explicitly on the command line
    prior_kwargs = dict(STREET_PRESET) if args.preset == "street" else {}
    sh_degree_given = any(a.startswith(("--sh-degree", "--sh_degree")) for a in sys.argv[1:])
    if args.preset == "street" and not sh_degree_given:
        args.sh_degree = prior_kwargs.pop("sh_degree")
    else:
        prior_kwargs.pop("sh_degree", None)
    for name in ("depth_weight", "depth_weight_final", "sky_alpha_weight", "fill_weight",
                 "use_masks", "learn_sky", "exposure_compensation", "max_scale_ratio"):
        value = getattr(args, name)
        if value is not None:
            prior_kwargs[name] = value

    config = TrainingConfig(
        iterations=args.num_iters,
        sh_degree=args.sh_degree,
        sh_degree_interval=args.sh_degree_interval,
        ssim_weight=args.ssim_weight,
        num_downscales=args.num_downscales,
        resolution_schedule=args.resolution_schedule,
        refine_every=args.refine_every,
        warmup_length=args.warmup_length,
        reset_alpha_every=args.reset_alpha_every,
        densify_grad_thresh=args.densify_grad_thresh,
        densify_size_thresh=args.densify_size_thresh,
        stop_screen_size_at=args.stop_screen_size_at,
        split_screen_size=args.split_screen_size,
        keep_crs=args.keep_crs,
        downscale_factor=args.downscale_factor,
        output=args.output,
        save_every=args.save_every,
        **prior_kwargs,
    )

    dataset = Dataset(
        args.input,
        downscale_factor=args.downscale_factor,
        eval_mode=args.eval,
        test_every=args.test_every,
        prior_dir=args.prior_dir,
    )
    if prior_kwargs:
        print(f"Priors attached: {dataset.prior_counts()}")
    print(f"Loaded {dataset.num_train} train cameras", end="")
    if args.eval:
        print(f", {dataset.num_test} test cameras")
    else:
        print()

    trainer = GaussianTrainer(dataset, config)

    def on_step(stats):
        print(
            f"step={stats.iteration:>6}  "
            f"splats={stats.splat_count:>8,}  "
            f"ms={stats.ms_per_step:.1f}"
        )

    trainer.train(on_step, callback_every=100)

    trainer.export_ply(args.output)
    print(f"Saved {args.output}")
    if config.learn_sky:
        sky_path = args.output.rsplit(".", 1)[0] + "_sky.png"
        trainer.export_sky(sky_path)
        print(f"Saved {sky_path}")

    if args.eval:
        metrics = trainer.evaluate()
        print(f"\n=== Evaluation ({metrics['num_test']} test views) ===")
        print(f"  PSNR:  {metrics['psnr']:.4f}")
        print(f"  SSIM:  {metrics['ssim']:.4f}")
        print(f"  L1:    {metrics['l1']:.4f}")
        print(f"  Gaussians: {metrics['num_gaussians']:,}")

    cleanup()


if __name__ == "__main__":
    main()
