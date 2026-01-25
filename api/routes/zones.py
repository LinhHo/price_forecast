from fastapi import APIRouter
from config import AUTOMATIC_DIR

router = APIRouter()

@router.get("/")
def list_zones():
    if not AUTOMATIC_DIR.exists():
        return []

    return sorted(
        d.name
        for d in AUTOMATIC_DIR.iterdir()
        if d.is_dir()
    )
