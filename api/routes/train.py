from fastapi import APIRouter
from price_forecast.api.authentication import router, require_admin


router = APIRouter()


# @router.post("/train/{zone}")
# def train(zone: str, token: str):
#     require_admin(token)
#     return {"status": "disabled", "reason": "Training runs in Colab only"}


# api/routes/train.py
from fastapi import APIRouter, Depends
from api.auth import require_admin

router = APIRouter()


@router.post("/{zone}")
def request_training(zone: str, _=Depends(require_admin)):
    # write a "train_request.json" to S3
    return {"status": "queued", "zone": zone}
