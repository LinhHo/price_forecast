from pathlib import Path
from forecasting.model.tft_model import TFTPriceModel
from infra.s3 import download_zone

# import boto3
from config import AUTOMATIC_DIR  # , LOCAL_MODEL_CACHE, S3_BUCKET_NAME

_MODEL_CACHE: dict[str, TFTPriceModel] = {}


def get_model(zone: str) -> TFTPriceModel:
    if zone in _MODEL_CACHE:
        return _MODEL_CACHE[zone]

    zone_dir = AUTOMATIC_DIR / zone
    runs_dir = zone_dir / "runs"

    # Download if zone or runs are missing
    if not runs_dir.exists():
        download_zone(zone, AUTOMATIC_DIR)

    run_id = _resolve_latest_run(zone_dir)
    run_dir = runs_dir / run_id

    # Safety check: ensure required artifacts exist
    # Handle partial cache cases
    required = [
        run_dir / "model" / "tft.ckpt",
        run_dir / "training_dataset.pt",
        run_dir / "meta.json",
    ]

    if not all(p.exists() for p in required):
        # re-download to repair partial cache
        download_zone(zone, AUTOMATIC_DIR)

    model = TFTPriceModel.load(zone, run_id, base_dir=run_dir)
    _MODEL_CACHE[zone] = model
    return model


def _resolve_latest_run(zone_dir: Path) -> str:
    runs_dir = zone_dir / "runs"
    runs = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    if not runs:
        raise RuntimeError(f"No trained runs found for zone {zone_dir.name}")
    return runs[-1]


# def get_model(zone: str) -> TFTPriceModel:
#     if zone in _MODEL_CACHE:
#         return _MODEL_CACHE[zone]

#     zone_dir = AUTOMATIC_DIR / zone

#     # If model artifacts are not present locally → pull from S3
#     if not zone_dir.exists():
#         download_zone(zone, AUTOMATIC_DIR)

#     # Decide which run to load (for now: latest)
#     run_id = _resolve_latest_run(zone_dir)
#     run_dir = zone_dir / "runs" / run_id

#     model = TFTPriceModel.load(zone, run_id, base_dir=run_dir)
#     _MODEL_CACHE[zone] = model
#     return model


# def ensure_run_downloaded(zone: str, run_id: str) -> Path:
#     local_run_dir = LOCAL_MODEL_CACHE / zone / "runs" / run_id
#     if local_run_dir.exists():
#         return local_run_dir

#     local_run_dir.mkdir(parents=True, exist_ok=True)

#     s3 = boto3.client("s3")

#     prefix = f"{zone}/runs/{run_id}/"
#     objects = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)

#     for obj in objects.get("Contents", []):
#         key = obj["Key"]
#         rel = key.replace(prefix, "")
#         local_path = local_run_dir / rel
#         local_path.parent.mkdir(parents=True, exist_ok=True)

#         s3.download_file(S3_BUCKET_NAME, key, str(local_path))

#     return local_run_dir


# from forecasting.model.tft_model import TFTPriceModel
# from infra.s3 import download_run
# from pathlib import Path
# import threading

# _MODEL_CACHE: dict[str, TFTPriceModel] = {}
# _LOCK = threading.Lock()


# def get_model(zone: str, run_id: str | None = None) -> TFTPriceModel:
#     """
#     Load and cache a TFTPriceModel for a zone.
#     Thread-safe.
#     """

#     cache_key = f"{zone}:{run_id or 'latest'}"

#     if cache_key in _MODEL_CACHE:
#         return _MODEL_CACHE[cache_key]

#     with _LOCK:
#         if cache_key in _MODEL_CACHE:
#             return _MODEL_CACHE[cache_key]

#         run_path = download_run(zone, run_id)

#         model = TFTPriceModel.load(
#             zone=zone,
#             run_id=run_path.name,
#         )

#         _MODEL_CACHE[cache_key] = model
#         return model


# from forecasting.model.tft_model import TFTPriceModel
# from infra.s3 import download_dir
# from pathlib import Path

# LOCAL_MODEL_ROOT = Path("/tmp/models")


# def load_model_from_s3(zone: str, run_id: str):
#     local_dir = LOCAL_MODEL_ROOT / zone / run_id
#     if not local_dir.exists():
#         download_dir(f"{zone}/runs/{run_id}", local_dir)

#     return TFTPriceModel.load(zone=zone, run_id=run_id)
