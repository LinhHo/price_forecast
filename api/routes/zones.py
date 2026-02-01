from fastapi import APIRouter
from config import AUTOMATIC_DIR

router = APIRouter()


@router.get("/")
@router.get(
    "/zones",
    operation_id="list_available_zones",
)
# drop down list of zones
def list_zones():
    if not AUTOMATIC_DIR.exists():
        return []
    return [d.name for d in AUTOMATIC_DIR.iterdir() if d.is_dir()]

    # return sorted(
    #     d.name
    #     for d in AUTOMATIC_DIR.iterdir()
    #     if d.is_dir()
    # )
