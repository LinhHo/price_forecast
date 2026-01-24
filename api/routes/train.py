from fastapi import APIRouter, Depends

# from api.authentication import router, require_admin


router = APIRouter()


@router.post("/train/{zone}")
def train(zone: str, token: str):
    # require_admin(token)
    return {"status": "disabled", "reason": "Training runs in Colab only"}


