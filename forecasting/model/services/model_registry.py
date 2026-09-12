import time
import logging
from pathlib import Path
from forecasting.model.tft_model import TFTPriceModel
from infra.s3 import download_run
from config import AUTOMATIC_DIR

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, TFTPriceModel] = {}
# zone -> (latest_run_id, checked_at) — re-check S3 every 5 minutes
_S3_RUN_CACHE: dict[str, tuple[str | None, float]] = {}
_S3_CHECK_TTL = 300


def get_model(zone: str) -> TFTPriceModel:
    """Load and cache the most recent trained model for a zone, checking S3 every 5 min."""
    latest = _latest_s3_run(zone)

    cached = _MODEL_CACHE.get(zone)
    if cached and (latest is None or cached.run_id == latest):
        return cached

    if cached and latest:
        logger.info("Newer model for zone=%s: %s → %s", zone, cached.run_id, latest)

    run_id = latest or _resolve_latest_local_run(AUTOMATIC_DIR / zone)
    run_dir = AUTOMATIC_DIR / zone / "runs" / run_id

    required = [
        run_dir / "model" / "tft.ckpt",
        run_dir / "training_dataset.pt",
        run_dir / "meta.json",
    ]
    if not all(p.exists() for p in required):
        logger.info("Downloading run=%s for zone=%s from S3", run_id, zone)
        download_run(zone, run_id, AUTOMATIC_DIR)

    model = TFTPriceModel.load(zone, run_id, base_dir=run_dir)
    _MODEL_CACHE[zone] = model
    return model


def _latest_s3_run(zone: str) -> str | None:
    """Return the most recent training run ID from S3, cached for _S3_CHECK_TTL seconds."""
    cached_id, checked_at = _S3_RUN_CACHE.get(zone, (None, 0.0))
    if time.time() - checked_at < _S3_CHECK_TTL:
        return cached_id
    try:
        from infra.s3 import list_runs
        runs = list_runs(zone)
        latest = runs[-1] if runs else None
        _S3_RUN_CACHE[zone] = (latest, time.time())
        return latest
    except Exception as e:
        logger.warning("Could not list S3 runs for zone=%s: %s", zone, e)
        _S3_RUN_CACHE[zone] = (cached_id, time.time())  # reset TTL, keep stale value
        return cached_id


def _resolve_latest_local_run(zone_dir: Path) -> str:
    """Return the most recent run ID from the local filesystem."""
    runs_dir = zone_dir / "runs"
    if not runs_dir.exists():
        raise RuntimeError(f"No local runs found for zone {zone_dir.name}")
    runs = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    if not runs:
        raise RuntimeError(f"No trained runs found for zone {zone_dir.name}")
    return runs[-1]
