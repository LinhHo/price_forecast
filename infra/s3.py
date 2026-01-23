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


# # infra/s3.py
# import boto3
# from pathlib import Path

# from config import S3_BUCKET_NAME, S3_REGION


# s3 = boto3.client("s3", region_name=S3_REGION)


# def upload_dir(local_dir: Path, s3_prefix: str):
#     for path in local_dir.rglob("*"):
#         if path.is_file():
#             s3.upload_file(
#                 str(path),
#                 S3_BUCKET_NAME,
#                 f"{s3_prefix}/{path.relative_to(local_dir)}",
#             )


# def download_dir(s3_prefix: str, local_dir: Path):
#     paginator = s3.get_paginator("list_objects_v2")

#     for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=s3_prefix):
#         for obj in page.get("Contents", []):
#             key = obj["Key"]
#             local_path = local_dir / Path(key).relative_to(s3_prefix)
#             local_path.parent.mkdir(parents=True, exist_ok=True)
#             s3.download_file(S3_BUCKET_NAME, key, str(local_path))


# """
# Optional helper to save and load to AWS free tier S3
# """


# # Call this inside TFTPriceModel.load() if local files don’t exist.
# def download_zone(bucket, zone, local_dir):
#     prefix = f"{zone}/"
#     for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix)["Contents"]:
#         dest = Path(local_dir) / obj["Key"]
#         dest.parent.mkdir(parents=True, exist_ok=True)
#         s3.download_file(bucket, obj["Key"], str(dest))
