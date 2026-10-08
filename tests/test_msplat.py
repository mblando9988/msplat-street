"""msplat test suite."""

import pytest
import numpy as np
import tempfile
import os

GARDEN = os.path.join(os.path.dirname(__file__), "..", "datasets", "mipnerf360", "garden")
HAS_GARDEN = os.path.isdir(GARDEN)


# ── Import tests ─────────────────────────────────────────────────────────────


def test_import():
    import msplat
    assert hasattr(msplat, "GaussianTrainer")
    assert hasattr(msplat, "TrainingConfig")
    assert hasattr(msplat, "Dataset")
    assert hasattr(msplat, "load_dataset")


def test_training_config_defaults():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.iterations == 30000
    assert cfg.sh_degree == 3
    assert cfg.ssim_weight == pytest.approx(0.2)
    assert cfg.refine_every == 100
    assert cfg.warmup_length == 500


def test_training_config_custom():
    from msplat import TrainingConfig

    cfg = TrainingConfig(iterations=100, sh_degree=1, ssim_weight=0.0)
    assert cfg.iterations == 100
    assert cfg.sh_degree == 1
    assert cfg.ssim_weight == 0.0


def test_training_config_mutable():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    cfg.iterations = 500
    assert cfg.iterations == 500


# ── Dataset tests ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    assert ds.num_train > 0
    assert ds.num_test > 0
    assert ds.num_train + ds.num_test > 100


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset_no_eval():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=False)
    assert ds.num_train > 0
    assert ds.num_test == 0


# ── Training tests ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_train_short():
    """Train 50 steps at 4x downscale — verify it runs without error."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    steps_seen = []
    trainer.train(lambda s: steps_seen.append(s.iteration), callback_every=10)

    assert trainer.iteration == 50
    assert trainer.splat_count > 100000
    assert steps_seen == [10, 20, 30, 40, 50]


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_step_by_step():
    """Manual step loop works."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        stats = trainer.step()

    assert stats.iteration == 10
    assert stats.splat_count > 0
    assert stats.ms_per_step > 0


# ── Render tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_render():
    """Render produces valid image array."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer, sync

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    img = trainer.render(0)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0
    # Values should be in [0, 1] range (approximately)
    assert img.min() >= -0.1
    assert img.max() <= 1.5


# ── Export tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_ply():
    """PLY export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        trainer.export_ply(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000  # non-trivial file
    finally:
        os.unlink(path)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_splat():
    """Splat export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        path = f.name

    try:
        trainer.export_splat(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000
    finally:
        os.unlink(path)


# ── Eval tests ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_evaluate():
    """Evaluation returns valid metrics dict."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    trainer.train(lambda s: None, callback_every=50)
    metrics = trainer.evaluate()

    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "l1" in metrics
    assert "num_test" in metrics
    assert metrics["num_test"] > 0
    assert metrics["psnr"] > 10  # sanity — should be at least somewhat trained
    assert 0 < metrics["ssim"] < 1
    assert metrics["l1"] > 0


# ── Checkpoint tests ────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_save_load():
    """Save checkpoint, load it, verify state is preserved."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    splats_at_50 = trainer.splat_count

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)
        assert os.path.getsize(ckpt_path) > 1000

        # Load into a fresh trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.iteration == 50
        assert trainer2.splat_count == splats_at_50
    finally:
        os.unlink(ckpt_path)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_resume_training():
    """Train 50 → save → load → train 50 more. Verify it completes."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)

        # Resume in a new trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        for _ in range(50):
            stats = trainer2.step()

        assert trainer2.iteration == 100
        assert stats.splat_count > 0
        assert stats.ms_per_step > 0
    finally:
        os.unlink(ckpt_path)


# ── Prior-guided training ───────────────────────────────────────────────────


def _write_png_gray(path, img):
    """Minimal 8-bit grayscale PNG writer (stdlib only; CI has no Pillow)."""
    import struct
    import zlib

    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw)))
        f.write(chunk(b"IEND", b""))


def _make_priors(tmpdir, n_steps=30):
    """Depth priors from a short warm-up render, plus synthetic sky (top rows) and keep
    masks, keyed by image name like msplat-prior writes them."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    trainer = GaussianTrainer(ds, TrainingConfig(iterations=n_steps, num_downscales=0))
    for _ in range(n_steps):
        trainer.step()

    names = sorted(os.listdir(os.path.join(GARDEN, "images")))
    assert len(names) == ds.num_train
    for sub in ("depth", "confidence", "sky", "mask"):
        os.makedirs(os.path.join(tmpdir, sub), exist_ok=True)
    for i, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        depth, alpha = trainer.render_depth(i)
        assert depth.ndim == 2 and depth.shape == alpha.shape
        # Half resolution: priors are sampled by normalized position, not pixel index
        np.save(os.path.join(tmpdir, "depth", stem + ".npy"), depth[::2, ::2].astype(np.float32))
        np.save(os.path.join(tmpdir, "confidence", stem + ".npy"),
                np.clip(alpha[::2, ::2], 0, 1).astype(np.float16))
        h, w = depth.shape
        sky = np.zeros((h, w), np.uint8)
        sky[: h // 10] = 255
        _write_png_gray(os.path.join(tmpdir, "sky", stem + ".png"), sky)
        keep = np.full((h, w), 255, np.uint8)
        keep[h // 2: h // 2 + 8, w // 2: w // 2 + 8] = 0
        _write_png_gray(os.path.join(tmpdir, "mask", stem + ".png"), keep)
    return names


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_render_depth():
    """render_depth returns finite expected depth where alpha is high."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    trainer = GaussianTrainer(ds, TrainingConfig(iterations=20, num_downscales=0))
    for _ in range(20):
        trainer.step()
    depth, alpha = trainer.render_depth(0)
    img = trainer.render(0)
    assert depth.shape == alpha.shape == img.shape[:2]
    assert np.isfinite(depth).all() and np.isfinite(alpha).all()
    assert alpha.min() >= -1e-4 and alpha.max() <= 1 + 1e-4
    covered = alpha > 0.5
    assert covered.mean() > 0.5
    assert (depth[covered] > 0).all()


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_street_preset_trains():
    """Prior-guided training (depth, sky, masks, exposure, scale cap) runs end to end."""
    import msplat
    from msplat import Dataset, GaussianTrainer

    with tempfile.TemporaryDirectory() as tmp:
        _make_priors(tmp)
        ds = Dataset(GARDEN, downscale_factor=4.0, prior_dir=tmp)
        counts = ds.prior_counts()
        assert counts["depth"] == counts["sky"] == counts["mask"] == ds.num_train

        cfg = msplat.street_config(iterations=120, num_downscales=1, resolution_schedule=60,
                                   refine_every=50, warmup_length=40, sky_width=64, sky_height=16)
        assert cfg.learn_sky and cfg.exposure_compensation and cfg.sh_degree == 1
        trainer = GaussianTrainer(ds, cfg)
        losses = []
        for _ in range(120):
            stats = trainer.step()
            if stats.iteration % 20 == 0:
                losses.append(trainer.prior_losses())
        assert trainer.splat_count > 0
        for l in losses:
            assert all(np.isfinite(v) and v >= 0 for v in l.values()), l
        assert losses[-1]["depth"] > 0  # depth prior active on every camera

        img = trainer.render(0)
        assert np.isfinite(img).all()
        depth, alpha = trainer.render_depth(0)
        assert np.isfinite(depth).all()

        sky_png = os.path.join(tmp, "sky_out.png")
        trainer.export_sky(sky_png)
        assert os.path.getsize(sky_png) > 0

        ckpt = os.path.join(tmp, "street.msplat")
        trainer.save_checkpoint(ckpt)
        trainer2 = GaussianTrainer(ds, cfg)
        trainer2.load_checkpoint(ckpt)
        assert trainer2.iteration == 120
        np.testing.assert_allclose(trainer2.render(0), img, atol=1e-5)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_scale_ratio_cap_and_exposure_only():
    """Options that do not need the aux rasterizer work on their own."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=60, num_downscales=0, max_scale_ratio=4.0,
                         exposure_compensation=True)
    trainer = GaussianTrainer(ds, cfg)
    for _ in range(60):
        trainer.step()
    assert trainer.splat_count > 0
    assert np.isfinite(trainer.render(0)).all()


def test_street_config_overrides():
    import msplat

    cfg = msplat.street_config(depth_weight=0.5, sh_degree=2)
    assert cfg.depth_weight == pytest.approx(0.5)
    assert cfg.sh_degree == 2
    assert cfg.learn_sky
    assert msplat.TrainingConfig().depth_weight == 0.0
    assert not msplat.TrainingConfig().learn_sky
