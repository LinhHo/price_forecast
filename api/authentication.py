from fastapi import APIRouter, Depends, HTTPException
from fastapi.param_functions import Header
import os

from config import ENABLE_ADMIN_AUTH


router = APIRouter()

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")


def require_admin(token: str | None = Header(default=None)):
    if not ENABLE_ADMIN_AUTH:
        return
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401)


# from fastapi import Header, HTTPException
# import os

# ADMIN_TOKEN = os.environ["ADMIN_TOKEN"]


# def require_admin(x_admin_token: str = Header(...)):
#     if x_admin_token != ADMIN_TOKEN:
#         raise HTTPException(status_code=403, detail="Forbidden")
