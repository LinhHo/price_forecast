from fastapi import APIRouter
from config import AUTOMATIC_DIR
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", operation_id="list_available_zones")
def list_zones():
    local_zones: set[str] = set()
    if AUTOMATIC_DIR.exists():
        local_zones = {d.name for d in AUTOMATIC_DIR.iterdir() if d.is_dir()}

    s3_zones: set[str] = set()
    try:
        from infra.s3 import list_zones as list_s3_zones
        s3_zones = set(list_s3_zones())
    except Exception as e:
        logger.warning("Could not list S3 zones: %s", e)

    return sorted(local_zones | s3_zones)
