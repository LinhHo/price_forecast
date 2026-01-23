from fastapi import APIRouter, Depends, HTTPException
import os


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")


def require_admin(token: str):
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "Forbidden")


from fastapi import Header, HTTPException
import os

ADMIN_TOKEN = os.environ["ADMIN_TOKEN"]


def require_admin(x_admin_token: str = Header(...)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
