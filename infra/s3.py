# infra/s3.py
import boto3
from pathlib import Path
from config import S3_BUCKET_NAME, S3_REGION

_s3 = boto3.client("s3", region_name=S3_REGION)


def download_zone(zone: str, local_dir: Path):
    """
    Download all model artifacts for a zone from S3 into local_dir
    """
    prefix = f"{zone}/"

    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            dest = local_dir / obj["Key"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            _s3.download_file(S3_BUCKET_NAME, obj["Key"], str(dest))


def upload_file(local_path: Path, s3_key: str):
    _s3.upload_file(str(local_path), S3_BUCKET_NAME, s3_key)