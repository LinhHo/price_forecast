# infra/s3.py
import json
import boto3
from pathlib import Path
from config import S3_BUCKET_NAME, S3_REGION

_s3 = boto3.client("s3", region_name=S3_REGION)


def download_zone(zone: str, local_dir: Path):
    """Download all artifacts for a zone from S3 into local_dir."""
    prefix = f"{zone}/"
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            dest = local_dir / obj["Key"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            _s3.download_file(S3_BUCKET_NAME, obj["Key"], str(dest))


def download_run(zone: str, run_id: str, local_dir: Path):
    """Download artifacts for a single training run from S3."""
    prefix = f"{zone}/runs/{run_id}/"
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            dest = local_dir / obj["Key"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            _s3.download_file(S3_BUCKET_NAME, obj["Key"], str(dest))


def upload_file(local_path: Path, s3_key: str):
    _s3.upload_file(str(local_path), S3_BUCKET_NAME, s3_key)


def upload_dir(local_dir: Path, s3_prefix: str):
    """Recursively upload all files under local_dir to s3_prefix/."""
    for path in local_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(local_dir)
            upload_file(path, f"{s3_prefix}/{rel}")


def list_zones() -> list[str]:
    """List all zone prefixes available in the S3 bucket."""
    paginator = _s3.get_paginator("list_objects_v2")
    zones = []
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix="", Delimiter="/"):
        for prefix_info in page.get("CommonPrefixes", []):
            zone = prefix_info["Prefix"].rstrip("/")
            if zone:
                zones.append(zone)
    return sorted(zones)


def list_runs(zone: str) -> list[str]:
    """List all training run IDs for a zone from S3, sorted chronologically."""
    paginator = _s3.get_paginator("list_objects_v2")
    runs = []
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=f"{zone}/runs/", Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            run = p["Prefix"].rstrip("/").split("/")[-1]
            if run:
                runs.append(run)
    return sorted(runs)


def upload_inference_stats(zone: str, inference_id: str, stats: dict):
    """Save inference accuracy stats JSON to S3 under {zone}/inferences/{inference_id}/stats.json."""
    _s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=f"{zone}/inferences/{inference_id}/stats.json",
        Body=json.dumps(stats, indent=2).encode(),
        ContentType="application/json",
    )


def list_inferences(zone: str) -> list[str]:
    """List all inference IDs for a zone from S3, sorted chronologically."""
    paginator = _s3.get_paginator("list_objects_v2")
    ids = []
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=f"{zone}/inferences/", Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            iid = p["Prefix"].rstrip("/").split("/")[-1]
            if iid:
                ids.append(iid)
    return sorted(ids)


def get_inference_stats(zone: str, inference_id: str) -> dict:
    """Download stats JSON for one inference from S3."""
    obj = _s3.get_object(
        Bucket=S3_BUCKET_NAME,
        Key=f"{zone}/inferences/{inference_id}/stats.json",
    )
    return json.loads(obj["Body"].read())
