


# import boto3
# from pathlib import Path

# s3 = boto3.client("s3")

# def upload_run(zone: str, run_dir: Path):
#     for file in run_dir.rglob("*"):
#         if file.is_file():
#             s3.upload_file(
#                 str(file),
#                 "price-forecast-models",
#                 f"{zone}/{file.relative_to(run_dir)}",
#             )


# infra/s3.py
import boto3
from pathlib import Path

BUCKET = "price-forecast-models"

s3 = boto3.client("s3")


def upload_dir(local_dir: Path, s3_prefix: str):
    for path in local_dir.rglob("*"):
        if path.is_file():
            s3.upload_file(
                str(path),
                BUCKET,
                f"{s3_prefix}/{path.relative_to(local_dir)}",
            )


def download_dir(s3_prefix: str, local_dir: Path):
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_path = local_dir / Path(key).relative_to(s3_prefix)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, key, str(local_path))
