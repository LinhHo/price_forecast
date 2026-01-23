from infra.s3 import list_runs

def list_available_zones():
    return list_runs()

def latest_run_for_zone(zone: str) -> str:
    return list_runs(zone)[0]
