from forecasting.model.tft_model import TFTPriceModel
from infra.s3 import download_run
from pathlib import Path
import threading

_MODEL_CACHE: dict[str, TFTPriceModel] = {}
_LOCK = threading.Lock()


def get_model(zone: str, run_id: str | None = None) -> TFTPriceModel:
    """
    Load and cache a TFTPriceModel for a zone.
    Thread-safe.
    """

    cache_key = f"{zone}:{run_id or 'latest'}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        run_path = download_run(zone, run_id)

        model = TFTPriceModel.load(
            zone=zone,
            run_id=run_path.name,
        )

        _MODEL_CACHE[cache_key] = model
        return model


from forecasting.model.tft_model import TFTPriceModel
from infra.s3 import download_dir
from pathlib import Path

LOCAL_MODEL_ROOT = Path("/tmp/models")


def load_model_from_s3(zone: str, run_id: str):
    local_dir = LOCAL_MODEL_ROOT / zone / run_id
    if not local_dir.exists():
        download_dir(f"{zone}/runs/{run_id}", local_dir)

    return TFTPriceModel.load(zone=zone, run_id=run_id)
