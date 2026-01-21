"""
Optional helper to save and load to AWS free tier S3
"""
import boto3
from pathlib import Path

s3 = boto3.client("s3")

# Call this inside TFTPriceModel.load() if local files don’t exist.
def download_zone(bucket, zone, local_dir):
    prefix = f"{zone}/"
    for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix)["Contents"]:
        dest = Path(local_dir) / obj["Key"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, obj["Key"], str(dest))
